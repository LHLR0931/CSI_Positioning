"""阶段二公平对照基线（PyTorch）—— 用于回答「GAT 的注意力融合是否真有增益」。

两个基线（都共享同一个 CNN 编码器，保证比较公平）：
  - ConcatMLP  : 4 条链路特征拼接后过 MLP 回归坐标（无注意力）。
  - MeanFusion : 每条链路各自回归坐标后取平均（最简单的融合）。

只有 CNN+GAT 明显优于这两者，才能说明「注意力融合」确有价值（见方案第 8 节）。

本机运行：python -m backend.models.fusion_baseline
"""

from __future__ import annotations

from backend import config
from backend.models.fusion_gat import _HAS_TORCH

if _HAS_TORCH:
    import torch
    from torch import nn
    from backend.models.fusion_gat import CNNEncoder, _TorchReg

    class ConcatMLP(nn.Module):
        """link_drop_p > 0 时为"随机链路遮挡"版本（D1 对照模型，2026-10-08）。

        训练时每个样本每条链路以概率 p 独立被遮挡：该链路的编码特征整体乘 0，且保证至少
        保留一条链路。推理时不遮挡；以后健康分门控通过 link_weights（[L] 或 [B,L]，取值
        0–1）按链路缩放编码特征。遮挡与加权作用在同一位置（编码特征），所以模型在训练时
        见过"某条链路权重为 0"的输入，推理时降权才不会落到训练分布之外。
        link_drop_p=0 且不传 link_weights 时，与原 ConcatMLP 逐位相同。
        """

        def __init__(self, num_nodes: int = config.NUM_LINKS_STAGE2, in_ch: int = 2, d: int = 64,
                     link_drop_p: float = 0.0, freq_bins: int = 0, out_dim: int = 2):
            super().__init__()
            self.encoder = CNNEncoder(in_ch, d, freq_bins=freq_bins)
            self.head = nn.Sequential(
                nn.Linear(num_nodes * d, 64), nn.ReLU(inplace=True), nn.Linear(64, out_dim))
            self.link_drop_p = float(link_drop_p)

        def forward(self, x, link_weights=None):   # x:[B,L,C,T,S]
            B, L, C, T, S = x.shape
            f = self.encoder(x.reshape(B * L, C, T, S)).reshape(B, L, -1)  # [B,L,D]
            if self.training and self.link_drop_p > 0:
                keep = (torch.rand(B, L, device=x.device) >= self.link_drop_p).float()
                none_left = keep.sum(1) == 0
                if none_left.any():                  # 全被遮掉的样本：随机放回一条
                    pick = torch.randint(0, L, (int(none_left.sum()),), device=x.device)
                    keep[none_left.nonzero(as_tuple=True)[0], pick] = 1.0
                f = f * keep.unsqueeze(-1)
            if link_weights is not None:
                w = torch.as_tensor(link_weights, dtype=f.dtype, device=f.device)
                f = f * (w.reshape(1, L, 1) if w.dim() == 1 else w.reshape(B, L, 1))
            return self.head(f.reshape(B, -1))

    class ConcatMLPSep(nn.Module):
        """E1（2026-10-14）：与 ConcatMLP 唯一的差别是 4 条链路**各用一个编码器、不共享权重**。

        用来区分冻结单链路编码器（F-MLP）为什么比 ConcatMLP 好：
          (a) 每条链路有自己的编码器（不共享权重），还是
          (b) 每条链路单独训练（再冻结）。
        本模型有 (a) 没有 (b)：结构与 F-MLP 完全相同（4 个 CNNEncoder + 同一个 MLP 头），
        但从零端到端联合训练。不加遮挡，与 ConcatMLP 对齐。
        """

        def __init__(self, num_nodes: int = config.NUM_LINKS_STAGE2, in_ch: int = 2, d: int = 64,
                     freq_bins: int = 0, out_dim: int = 2):
            super().__init__()
            self.encoders = nn.ModuleList(CNNEncoder(in_ch, d, freq_bins=freq_bins) for _ in range(num_nodes))
            self.head = nn.Sequential(
                nn.Linear(num_nodes * d, 64), nn.ReLU(inplace=True), nn.Linear(64, out_dim))

        def forward(self, x):                       # x:[B,L,C,T,S]
            f = torch.cat([enc(x[:, l]) for l, enc in enumerate(self.encoders)], dim=1)   # [B,L*D]
            return self.head(f)

    class MeanFusion(nn.Module):
        def __init__(self, in_ch: int = 2, d: int = 64, freq_bins: int = 0, out_dim: int = 2):
            super().__init__()
            self.encoder = CNNEncoder(in_ch, d, freq_bins=freq_bins)
            self.head = nn.Linear(d, out_dim)   # 分类时为每链路 logits，跨链路取平均

        def forward(self, x):                       # [B,L,C,T,S]
            B, L, C, T, S = x.shape
            f = self.encoder(x.reshape(B * L, C, T, S)).reshape(B, L, -1)  # [B,L,D]
            return self.head(f).mean(dim=1)         # 每链路回归后平均

    def _task_kw(task: str, n_classes: int | None, kw: dict) -> dict:
        """分类时把输出维度设成类别数；回归保持 2。"""
        if task == "cls":
            if not n_classes:
                raise ValueError("task='cls' 需要 n_classes")
            kw = {**kw, "out_dim": int(n_classes)}
        return kw

    class ConcatMLPModel(_TorchReg):
        name = "ConcatMLP"

        def __init__(self, lr: float = 1e-3, seed: int = config.RANDOM_SEED,
                     task: str = "reg", n_classes: int | None = None, grid_cols: int | None = None, **kw):
            torch.manual_seed(seed)   # 必须在构造 net 之前设置，见 fusion_gat.FusionGATModel 注释
            super().__init__(ConcatMLP(**_task_kw(task, n_classes, kw)), lr=lr, seed=seed,
                             task=task, grid_cols=grid_cols)

    class ConcatMLPMaskModel(_TorchReg):
        """对照模型：ConcatMLP + 训练时随机链路遮挡（默认 p=0.25）。"""
        name = "ConcatMLP-mask"

        def __init__(self, lr: float = 1e-3, seed: int = config.RANDOM_SEED,
                     link_drop_p: float = 0.25, task: str = "reg", n_classes: int | None = None,
                     grid_cols: int | None = None, **kw):
            torch.manual_seed(seed)
            super().__init__(ConcatMLP(link_drop_p=link_drop_p, **_task_kw(task, n_classes, kw)),
                             lr=lr, seed=seed, task=task, grid_cols=grid_cols)

    class ConcatMLPSepModel(_TorchReg):
        """E1：ConcatMLP，但 4 条链路的编码器不共享权重（见 ConcatMLPSep）。"""
        name = "ConcatMLP-sep"

        def __init__(self, lr: float = 1e-3, seed: int = config.RANDOM_SEED,
                     task: str = "reg", n_classes: int | None = None, grid_cols: int | None = None, **kw):
            torch.manual_seed(seed)
            super().__init__(ConcatMLPSep(**_task_kw(task, n_classes, kw)), lr=lr, seed=seed,
                             task=task, grid_cols=grid_cols)

    class MeanFusionModel(_TorchReg):
        name = "MeanFusion"

        def __init__(self, lr: float = 1e-3, seed: int = config.RANDOM_SEED,
                     task: str = "reg", n_classes: int | None = None, grid_cols: int | None = None, **kw):
            torch.manual_seed(seed)
            super().__init__(MeanFusion(**_task_kw(task, n_classes, kw)), lr=lr, seed=seed,
                             task=task, grid_cols=grid_cols)

    class SingleLinkCNNModel(MeanFusionModel):
        """单链路 CNN 基线：架构与 MeanFusion 完全相同（CNNEncoder + Linear(d,2)）。
        MeanFusion 在 L=1 输入下"跨链路取平均"退化为恒等，等价于单链路 CNN——
        不新写模型，只在结果表里换个名字以便区分。用法：喂 X[:, l:l+1] 单条链路。
        """
        name = "SingleLinkCNN"


if __name__ == "__main__":
    if not _HAS_TORCH:
        print("本环境无 PyTorch。请在你本机运行；无 torch 时用 "
              "python -m backend.models.fusion_numpy 做多链路对照。")
        raise SystemExit(0)

    from backend.training.prepare import load_stage2_windows
    from backend.models.metrics import evaluate_xy
    from backend.data.sessions import session_grid

    data, _ = load_stage2_windows(T=16)
    Xtr, ytr, _ = data["train"]
    Xte, yte, _ = data["test"]
    for Model in (ConcatMLPModel, MeanFusionModel):
        m = Model(lr=1e-3).fit(Xtr, ytr, epochs=60)
        met = evaluate_xy(m.predict(Xte), yte, grid=session_grid("session_phase2_001"))
        print(f"{Model.name:11s} 测试集平均误差 = {met['mean_error_m']:.3f} m")
    print("基线自检通过。")
