"""样本图 GCN —— 按前人工作（Wang 2026、Sun 2026，同组 RSSI+GNN 毕业论文）的结构移植到 CSI。

为什么要这个模型
----------------
前两篇论文的 GNN 是"样本图"：**每个样本是一个节点**，按特征向量的欧氏距离给每个样本连 k 个
最相似的样本，再用两层 GCNConv 做信息传递；新样本作为查询节点挂到图上，连向最相似的训练样本。
本质上是"会学习的 KNN"。把同样的结构用在 CSI 上，才能和前人的 RSSI+GNN 对比，并回答
"图上的信息传递比直接找邻居（KNN）多带来了什么"。
（注意：这与本项目 9 月的 HeteroGAT 不同——那是"链路图"，4 条链路是 4 个节点。）

与前人代码一致的部分（Sun 论文附录 A-1）
---------------------------------------
    两层图卷积，通道 128 → 64，每层 conv → BatchNorm → ReLU → Dropout(0.1)，最后一层全连接输出
    Adam(lr=1e-3, weight_decay=1e-4)，ReduceLROnPlateau(factor=0.5, patience=10)
    最多 300 轮，早停 patience=15（Wang 论文表 3.3）
    建图：特征向量的欧氏距离，k 近邻（两篇都没写 k；这里取 k=5，与 Sun 的 KNN 基线 n_neighbors=5 相同）

有意改动的部分（论文里要写明）
------------------------------
    1. 任务：15 类分类 + 交叉熵（前人是回归 + SmoothL1），与本项目其他模型一致。
    2. 早停看验证集平均定位误差（与本项目其他模型同一判据），不看损失。
    3. 训练是整图一次前向（full batch）。前人用 PyG 的数据加载器、batch=4，论文没写清楚批的含义。
    4. **防泄漏的推理方式**：训练节点的表示只在训练图上计算；每个查询（验证/测试）样本单独挂上去，
       只**接收**它的 k 个最近训练样本的信息，不向训练节点发送，查询样本之间也不相连。
       这样一个测试样本的预测与其他测试样本完全无关（等价于"逐个把新样本挂到图上"）。
    5. 节点特征：每个 16 帧窗口在时间上取平均后的预处理幅度+相位（四链路 4×2×55 = 440 维），
       用训练集的均值/标准差标准化。KNN 基线用完全相同的特征，两者的差别只剩"有没有信息传递"。

不依赖 torch_geometric：GCN 卷积按 Kipf & Welling 的对称归一化 Â = D^-1/2 (A+I) D^-1/2 手写。
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors

from backend import config


def window_features(X: np.ndarray) -> np.ndarray:
    """[N, L, C, T, S] -> [N, L*C*S]：每个窗口在时间上取平均（KNN 与 GCN 共用）。"""
    X = np.asarray(X)
    return X.mean(axis=3).reshape(len(X), -1).astype(np.float32)


class _SampleGCN(nn.Module):
    def __init__(self, in_dim: int, n_out: int, hidden=(128, 64), drop: float = 0.1):
        super().__init__()
        dims = [in_dim, *hidden]
        self.lins = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        self.bias = nn.ParameterList(nn.Parameter(torch.zeros(b)) for b in dims[1:])
        self.bns = nn.ModuleList(nn.BatchNorm1d(b) for b in dims[1:])
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(dims[-1], n_out)

    def _post(self, i, z):
        return self.drop(torch.relu(self.bns[i](z + self.bias[i])))

    def forward_train(self, x, A):
        """训练图整图前向。A：训练图归一化邻接（稀疏）。返回 logits 与每层输入（推理时要用）。"""
        hs = [x]
        h = x
        for i, lin in enumerate(self.lins):
            h = self._post(i, torch.sparse.mm(A, lin(h)))
            hs.append(h)
        return self.fc(h), hs

    def forward_query(self, xq, nbr, hs_train, deg_train):
        """查询节点：每层只聚合"自己 + k 个训练邻居"的上一层表示（邻居表示来自训练图）。"""
        k = nbr.shape[1]
        dq = float(k + 1)
        w_nb = 1.0 / torch.sqrt(dq * deg_train[nbr])                    # [M,k]
        h = xq
        for i, lin in enumerate(self.lins):
            m_self = lin(h) / dq
            m_nb = (lin(hs_train[i])[nbr] * w_nb.unsqueeze(-1)).sum(1)    # [M,out]
            h = self._post(i, m_self + m_nb)
        return self.fc(h)


class SampleGCNModel:
    """与 _TorchReg 相同的接口（fit / predict / save / load / best_epoch），可直接放进 train_d1。"""
    name = "GCN"
    MAX_EPOCHS, PATIENCE = 300, 15          # Wang 论文表 3.3

    def __init__(self, n_classes: int, grid_cols: int, seed: int = config.RANDOM_SEED,
                 k: int = 5, lr: float = 1e-3, weight_decay: float = 1e-4):
        self.n_classes, self.grid_cols, self.seed, self.k = n_classes, grid_cols, seed, k
        self.lr, self.wd = lr, weight_decay
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = None
        self.best_epoch = self.best_val_loss = None

    # ---- 图 ----
    def _graph(self, F: np.ndarray):
        """训练图：每个节点连 k 个最近邻，再对称化，加自环，做对称归一化。"""
        nn_ = NearestNeighbors(n_neighbors=self.k + 1).fit(F)
        idx = nn_.kneighbors(F, return_distance=False)[:, 1:]              # 去掉自己
        N = len(F)
        r = np.repeat(np.arange(N), self.k)
        c = idx.reshape(-1)
        edges = np.unique(np.concatenate([np.stack([r, c], 1), np.stack([c, r], 1),
                                          np.stack([np.arange(N)] * 2, 1)]), axis=0)
        deg = np.bincount(edges[:, 0], minlength=N).astype(np.float32)
        val = 1.0 / np.sqrt(deg[edges[:, 0]] * deg[edges[:, 1]])
        A = torch.sparse_coo_tensor(torch.tensor(edges.T), torch.tensor(val, dtype=torch.float32),
                                    (N, N), check_invariants=False).coalesce().to(self.device)
        return A, torch.tensor(deg, device=self.device)

    def _query_nbr(self, Fq: np.ndarray) -> torch.Tensor:
        idx = self._nn.kneighbors(Fq, n_neighbors=self.k, return_distance=False)
        return torch.tensor(idx, device=self.device)

    def _std(self, X) -> np.ndarray:
        return (window_features(X) - self.mu) / self.sd

    def _setup(self, F: np.ndarray):
        self.F_train = F
        self._nn = NearestNeighbors(n_neighbors=self.k).fit(F)
        self.A, self.deg = self._graph(F)
        self.x_train = torch.tensor(F, device=self.device)

    # ---- 训练 ----
    def fit(self, X, y, epochs: int | None = None, patience: int | None = None,
            X_val=None, y_val=None, early_stop_on: str = "error", **_):
        epochs = epochs or self.MAX_EPOCHS
        patience = patience or self.PATIENCE
        F = window_features(X)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        self._setup((F - self.mu) / self.sd)
        torch.manual_seed(self.seed)
        self.net = _SampleGCN(F.shape[1], self.n_classes).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        yt = torch.tensor(np.asarray(y), dtype=torch.long, device=self.device)
        lossf = nn.CrossEntropyLoss()
        has_val = X_val is not None
        if has_val:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
            yv = np.asarray(y_val)
            yv_xy = np.stack([yv % self.grid_cols, yv // self.grid_cols], 1)
            Xv = X_val
        best, best_state, stale = float("inf"), None, 0
        for ep in range(epochs):
            self.net.train()
            opt.zero_grad()
            logits, _ = self.net.forward_train(self.x_train, self.A)
            loss = lossf(logits, yt)
            loss.backward()
            opt.step()
            if has_val:
                err = float(np.linalg.norm(self.predict(Xv) - yv_xy, axis=1).mean())   # 网格单位
                sched.step(err)
                if err < best - 1e-4:
                    best, self.best_epoch, stale = err, ep, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    stale += 1
                    if stale >= patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.best_val_loss = best
        return self

    # ---- 推理 ----
    @torch.no_grad()
    def logits(self, X) -> np.ndarray:
        self.net.eval()
        _, hs = self.net.forward_train(self.x_train, self.A)
        Fq = self._std(X)
        out = []
        for s in range(0, len(Fq), 1024):
            q = Fq[s:s + 1024]
            out.append(self.net.forward_query(torch.tensor(q, device=self.device),
                                              self._query_nbr(q), hs, self.deg).cpu().numpy())
        return np.concatenate(out)

    def predict(self, X) -> np.ndarray:
        lab = self.logits(X).argmax(1)
        return np.stack([lab % self.grid_cols, lab // self.grid_cols], 1).astype(np.float32)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "F_train": self.F_train, "mu": self.mu, "sd": self.sd,
                    "k": self.k, "n_classes": self.n_classes}, f"{path}.pt")

    def load(self, path):
        d = torch.load(f"{path}.pt", map_location=self.device, weights_only=False)
        self.k, self.mu, self.sd = d["k"], d["mu"], d["sd"]
        self._setup(d["F_train"])
        self.net = _SampleGCN(d["F_train"].shape[1], d["n_classes"]).to(self.device)
        self.net.load_state_dict(d["state"])
        self.net.eval()
        return self
