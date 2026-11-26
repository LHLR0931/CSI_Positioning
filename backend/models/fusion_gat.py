"""阶段二主力模型：CNN 编码器 + GAT 多链路融合（PyTorch）。

对应方案第 8 节 & 附录 A.4：
  每条链路(2 Pi × 2 AP = 4 条) 的 [C=2, T, S] 先经共享 CNN 编码器 → 节点特征；
  4 个节点组成图，GAT 用注意力自动判断「此刻更该信任哪条链路」，融合后回归出
  连续坐标 (x, y)。输出连续坐标，便于送入卡尔曼跟踪层。

GAT 层是手写的稠密实现（4 节点小图，无需安装 PyTorch-Geometric，只依赖 torch）。

依赖 PyTorch（本机 `pip install torch`）。无 torch 环境用 backend.models.fusion_numpy
做「多链路 vs 单链路」的快速对照。

本机运行：python -m backend.models.fusion_gat
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from backend import config

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    class GATLayer(nn.Module):
        """稠密多头图注意力层（适用于小的全连接图）。"""

        def __init__(self, in_dim: int, out_dim: int, heads: int = 4, concat: bool = True):
            super().__init__()
            self.heads, self.out, self.concat = heads, out_dim, concat
            self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
            self.a_src = nn.Parameter(torch.randn(heads, out_dim) * 0.1)
            self.a_dst = nn.Parameter(torch.randn(heads, out_dim) * 0.1)
            self.leaky = nn.LeakyReLU(0.2)

        def forward(self, h, adj, return_attention: bool = False):
            # h:[B,N,in], adj:[N,N] (0/1)
            B, N, _ = h.shape
            Wh = self.W(h).view(B, N, self.heads, self.out)     # [B,N,H,O]
            e_src = (Wh * self.a_src).sum(-1)                   # [B,N,H]
            e_dst = (Wh * self.a_dst).sum(-1)                   # [B,N,H]
            e = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))  # [B,N,N,H]
            e = e.permute(0, 3, 1, 2)                           # [B,H,N,N]
            mask = (adj == 0).unsqueeze(0).unsqueeze(0)
            e = e.masked_fill(mask, float("-inf"))
            alpha = torch.softmax(e, dim=-1)                    # 对 j 归一化
            Wh_h = Wh.permute(0, 2, 1, 3)                       # [B,H,N,O]
            out = alpha @ Wh_h                                  # [B,H,N,O]
            out = out.permute(0, 2, 1, 3)                       # [B,N,H,O]
            z = out.reshape(B, N, self.heads * self.out) if self.concat else out.mean(2)
            return (z, alpha) if return_attention else z

    class CNNEncoder(nn.Module):
        """共享的单链路 CNN 编码器：[B,C,T,S] -> [B,D]。

        freq_bins=0（默认，旧 pilot 模型的结构）：卷积后对时间和子载波**一起**全局平均池化。
        freq_bins>0（2026-10-08 起 D1 使用）：只对时间平均，子载波轴保留，再平均分成
        freq_bins 段。原因：位置指纹就是"哪些子载波被衰落掉"，全局平均会把"凹陷在第 10 个
        子载波"和"凹陷在第 40 个子载波"变成同一个特征。D1 实测：同一份预处理输入，
        保留子载波轴的线性模型 r7 准确率 86%，只保留全局均值的线性模型 15%。
        """

        def __init__(self, in_ch: int = 2, d: int = 64, freq_bins: int = 0):
            super().__init__()
            self.freq_bins = int(freq_bins)
            conv = [nn.Conv2d(in_ch, 16, (3, 5), padding=(1, 2)), nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, (3, 5), padding=(1, 2)), nn.ReLU(inplace=True)]
            if self.freq_bins > 0:
                self.body = nn.Sequential(*conv)
                self.freq_pool = nn.AdaptiveAvgPool1d(self.freq_bins)
                self.fc = nn.Linear(32 * self.freq_bins, d)
            else:
                self.body = nn.Sequential(*conv, nn.AdaptiveAvgPool2d(1))
                self.fc = nn.Linear(32, d)

        def forward(self, x):
            h = self.body(x)                                   # [B,32,T,S] 或 [B,32,1,1]
            if self.freq_bins > 0:
                h = self.freq_pool(h.mean(dim=2))              # 只对时间平均 -> [B,32,bins]
            return self.fc(h.flatten(1))

    class FusionGAT(nn.Module):
        def __init__(self, num_nodes: int = config.NUM_LINKS_STAGE2, in_ch: int = 2, d: int = 64):
            super().__init__()
            self.encoder = CNNEncoder(in_ch, d)
            self.gat1 = GATLayer(d, 32, heads=4, concat=True)   # -> 128
            self.gat2 = GATLayer(128, 32, heads=1, concat=False)  # -> 32
            self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(inplace=True), nn.Linear(32, 2))
            adj = torch.ones(num_nodes, num_nodes)              # 全连接(含自环)
            self.register_buffer("adj", adj)

        def forward(self, x, return_attention: bool = False):   # x:[B,L,C,T,S]
            B, L, C, T, S = x.shape
            f = self.encoder(x.reshape(B * L, C, T, S)).reshape(B, L, -1)  # [B,L,D]
            h1, a1 = self.gat1(f, self.adj, return_attention=True)
            h2, a2 = self.gat2(F.elu(h1), self.adj, return_attention=True)
            g = h2.mean(dim=1)                      # 图级池化(对节点求均值)
            y = self.head(g)                        # [B,2]
            if return_attention:
                return y, {"layer1": {"alpha": a1}, "layer2": {"alpha": a2}}
            return y

    class _TorchReg:
        """通用回归训练器（MSE + Adam），供 GAT 与基线（含单链路）复用。

        改这一个类一次，所有继承它的模型（GAT/ConcatMLP/MeanFusion/单链路）自动
        获得同一套 early-stopping 逻辑，不存在"只给某个模型加"的不一致风险。
        """

        def __init__(self, net, lr: float = 1e-3, seed: int = config.RANDOM_SEED,
                     task: str = "reg", grid_cols: int | None = None):
            """task="reg"（默认，旧行为）：输出 [N,2] 连续坐标，MSE 训练。
            task="cls"（2026-10-08 起 D1 使用）：输出 num_classes 个 logits，交叉熵训练，
            fit 的 y 传格点 label；predict 仍返回 [N,2] 坐标（取概率最大格点的 x,y），
            所以评价代码不用改。改分类的原因：D1 r7 上同一 CNN 回归 0.43–0.51 m、
            分类 0.24–0.30 m（3 个种子）。回归在两点之间犹豫时会输出中点，那里不是格点。
            """
            if task not in ("reg", "cls"):
                raise ValueError(f"task 必须是 reg 或 cls，收到 {task}")
            if task == "cls" and not grid_cols:
                raise ValueError("task='cls' 需要 grid_cols，用于把 label 换回 (x,y)")
            self.task, self.grid_cols = task, grid_cols
            torch.manual_seed(seed)
            self.net = net
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.net.to(self.device)
            self.lr = lr
            self.history: list[dict] = []
            self.best_epoch: int | None = None
            self.best_val_loss: float | None = None
            self.epochs_run = 0

        def fit(self, X, y_xy, epochs: int = 150, batch: int = 32, verbose: bool = False,
                X_val=None, y_val=None, patience: int = 20, min_delta: float = 1e-4,
                weight_decay: float = 1e-4, early_stop_on: str = "loss"):
            """训练模型，提供验证集时执行 early stopping + 恢复最佳权重。

            early_stop_on="loss"（默认，旧行为）：看验证集的训练损失（回归 MSE / 分类交叉熵）。
            early_stop_on="error"（2026-10-08 起 D1 使用）：看验证集的平均定位误差（网格单位，
            越小越好）。原因：分类时交叉熵会因模型"过度自信"而早早上升，D1 上分类模型在
            第 2–4 轮就被停掉，而定位误差其实还在下降；回归与分类用同一判据才能公平比较。
            不传 X_val/y_val 时跑满 epochs，不早停。
            """
            if early_stop_on not in ("loss", "error"):
                raise ValueError(f"early_stop_on 必须是 loss 或 error，收到 {early_stop_on}")
            if (X_val is None) != (y_val is None):
                raise ValueError("X_val and y_val must be provided together")

            opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=weight_decay)
            cls = self.task == "cls"
            lossf = nn.CrossEntropyLoss() if cls else nn.MSELoss()
            ydt = torch.long if cls else torch.float32       # 分类时 y_xy 实际是格点 label
            Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
            yt = torch.tensor(np.asarray(y_xy), dtype=ydt)
            dl = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(Xt, yt), batch_size=batch, shuffle=True)

            has_val = X_val is not None
            if has_val:
                Xvt = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(self.device)
                yvt = torch.tensor(np.asarray(y_val), dtype=ydt).to(self.device)
                if cls:      # 验证集真实坐标（网格单位），给 error 判据用
                    yv_np = np.asarray(y_val)
                    yv_xy = torch.tensor(np.stack([yv_np % self.grid_cols, yv_np // self.grid_cols], 1),
                                         dtype=torch.float32).to(self.device)
                else:
                    yv_xy = yvt
                lr_patience = max(2, patience // 3)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=lr_patience, min_lr=1e-5)
            else:
                Xvt = yvt = scheduler = None

            self.history = []
            best_state = None
            best_loss = float("inf")
            best_epoch = None
            stale = 0

            for ep in range(epochs):
                self.net.train()
                tot = 0.0
                for xb, yb in dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    loss = lossf(self.net(xb), yb)
                    loss.backward()
                    opt.step()
                    tot += loss.item() * len(xb)
                train_loss = tot / len(Xt)
                row = {"epoch": ep, "train_loss": float(train_loss),
                       "lr": float(opt.param_groups[0]["lr"])}

                if has_val:
                    self.net.eval()
                    with torch.no_grad():
                        out_v = self.net(Xvt)
                        val_loss = float(lossf(out_v, yvt).item())
                        if cls:
                            lab_v = out_v.argmax(1)
                            pxy = torch.stack([lab_v % self.grid_cols, lab_v // self.grid_cols], 1).float()
                        else:
                            pxy = out_v
                        val_err = float(torch.linalg.norm(pxy - yv_xy, dim=1).mean().item())
                    row["val_loss"], row["val_error"] = val_loss, val_err
                    if early_stop_on == "error":
                        val_loss = val_err          # 下面的早停与学习率调度统一看这个数
                    scheduler.step(val_loss)

                    if val_loss < best_loss - min_delta:
                        best_loss, best_epoch = val_loss, ep
                        best_state = copy.deepcopy(self.net.state_dict())
                        stale = 0
                    else:
                        stale += 1

                    if verbose and (ep % 15 == 0 or ep == epochs - 1):
                        print(f"    epoch {ep:3d}  train_mse={train_loss:.4f}  "
                              f"val_mse={val_loss:.4f}  lr={opt.param_groups[0]['lr']:.2e}")
                    self.history.append(row)
                    self.epochs_run = ep + 1
                    if stale >= patience:
                        if verbose:
                            print(f"    early stopping: epoch={ep}, best_epoch={best_epoch}, "
                                  f"best_val_mse={best_loss:.4f}")
                        break
                else:
                    if verbose and (ep % 15 == 0 or ep == epochs - 1):
                        print(f"    epoch {ep:3d}  mse={train_loss:.4f}")
                    self.history.append(row)
                    self.epochs_run = ep + 1

            if best_state is not None:
                self.net.load_state_dict(best_state)
                self.best_epoch = best_epoch
                self.best_val_loss = best_loss
                if verbose:
                    print(f"    restored best weights: epoch={best_epoch}, val_mse={best_loss:.4f}")
            return self

        def _forward_np(self, X, batch: int = 512) -> np.ndarray:
            self.net.eval()
            X = np.asarray(X)
            outs = []
            with torch.no_grad():
                for s in range(0, len(X), batch):
                    xb = torch.tensor(X[s:s + batch], dtype=torch.float32).to(self.device)
                    outs.append(self.net(xb).cpu().numpy())
            return np.concatenate(outs, axis=0) if outs else np.empty((0, 2), np.float32)

        def predict(self, X) -> np.ndarray:
            """返回 [N,2] 坐标（网格单位）。分类模型取概率最大格点的 (x,y)。"""
            out = self._forward_np(X)
            if self.task == "reg":
                return out                               # [N,2] 连续坐标
            lab = out.argmax(axis=1)
            return np.stack([lab % self.grid_cols, lab // self.grid_cols], axis=1).astype(np.float32)

        def predict_proba(self, X) -> np.ndarray:
            """分类模型的 [N,num_classes] 概率。"""
            if self.task != "cls":
                raise ValueError("predict_proba 只适用于 task='cls'")
            z = self._forward_np(X)
            z = np.exp(z - z.max(axis=1, keepdims=True))
            return z / z.sum(axis=1, keepdims=True)

        def save(self, path: Path) -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.net.state_dict(), f"{path}.pt")

        def load(self, path: Path):
            self.net.load_state_dict(torch.load(f"{path}.pt", map_location=self.device))
            self.net.eval()
            return self

    class FusionGATModel(_TorchReg):
        name = "GAT"

        def __init__(self, lr: float = 1e-3, seed: int = config.RANDOM_SEED, **kw):
            # 种子必须在构造 net 之前设置——net 的随机初始化在这里发生，
            # _TorchReg.__init__ 里的 manual_seed 已经太晚了（net 作为参数
            # 在 super().__init__ 调用前就已经完成随机初始化）。
            torch.manual_seed(seed)
            super().__init__(FusionGAT(**kw), lr=lr, seed=seed)

        def attention(self, X, batch: int = 256) -> dict:
            """导出全连接 GAT 的 node-level 注意力，键名与 HeteroGAT 对齐（无 beta）。

            alpha[layer]: [N_samples, heads, L, L]，alpha[..., i, j] = 节点 i 分配给
            邻居 j 的权重，每行和为 1。
            """
            self.net.eval()
            X = np.asarray(X)
            acc = {"layer1": [], "layer2": []}
            with torch.no_grad():
                for s in range(0, len(X), batch):
                    xb = torch.tensor(X[s:s + batch], dtype=torch.float32).to(self.device)
                    _, att = self.net(xb, return_attention=True)
                    for layer in acc:
                        acc[layer].append(att[layer]["alpha"].cpu().numpy())
            return {"alpha": {l: np.concatenate(v, axis=0) for l, v in acc.items()},
                    "link_ids": config.all_link_ids()}


if __name__ == "__main__":
    if not _HAS_TORCH:
        print("本环境无 PyTorch。请在你本机运行；无 torch 时用 "
              "python -m backend.models.fusion_numpy 做多链路对照。")
        raise SystemExit(0)

    from backend.training.prepare import load_stage2_windows
    from backend.models.metrics import evaluate_xy, format_report

    data, _ = load_stage2_windows(T=16)
    Xtr, ytr, _ = data["train"]
    Xte, yte, _ = data["test"]
    print(f"输入形状 X={Xtr.shape}  (样本, 链路, 通道, 时间窗, 子载波)")
    m = FusionGATModel(lr=1e-3).fit(Xtr, ytr, epochs=60, verbose=True)
    from backend.data.sessions import session_grid
    metrics = evaluate_xy(m.predict(Xte), yte, grid=session_grid("session_phase2_001"))
    print(f"CNN+GAT 测试集: 平均 {metrics['mean_error_m']:.3f}m / "
          f"中位 {metrics['median_error_m']:.3f}m / 90% {metrics['p90_error_m']:.3f}m")
    print("CNN+GAT 自检通过。")
