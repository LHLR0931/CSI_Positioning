"""训练数据准备 —— 把 load → split → preprocess 串成模型可直接消费的张量。

阶段一：单链路，输出 [N, C=2, S]（幅度/相位两通道）。
供 KNN、CNN 及各训练脚本复用，保证三模型用同一份划分与预处理（公平比较）。
"""

from __future__ import annotations

import numpy as np

from backend import config
from backend.data.load import load_static, load_static_raw64
from backend.data.split import make_split
from backend.data.preprocess import Preprocessor, select_subcarriers


def _preprocessor_for_dataset(data: dict) -> Preprocessor:
    """构造与数据集物理子载波编号绑定的预处理器。"""
    return Preprocessor(subcarrier_indices=data.get("sc_idx"))


def load_stage1(link: int = 0, mode: str = "stratified", session: str = "session_001"):
    """返回 ({'train','val','test'}: (X[N,2,S], y[N]), preprocessor)。"""
    d = load_static(session)
    sp = make_split(d["label"], d["batch"], mode=mode)
    pp = _preprocessor_for_dataset(d).fit(d["amp"][sp["train"]], d["phase"][sp["train"]])

    def feat(idx):
        a, p = pp.transform(d["amp"][idx], d["phase"][idx])
        X = np.stack([a[:, link], p[:, link]], axis=1)  # [N, 2, S]
        return X.astype(np.float32), d["label"][idx].astype(np.int64)

    return {k: feat(sp[k]) for k in ("train", "val", "test")}, pp


def _apply_pp_windows(pp: Preprocessor, amp_w: np.ndarray, phase_w: np.ndarray) -> np.ndarray:
    """把窗口内每帧过预处理，输出 [Nw, 2, T, S]（通道=幅度/相位）。"""
    Nw, T, S = amp_w.shape
    a, p = pp.transform(amp_w.reshape(Nw * T, 1, S), phase_w.reshape(Nw * T, 1, S))
    a = a.reshape(Nw, T, S)
    p = p.reshape(Nw, T, S)
    return np.stack([a, p], axis=1).astype(np.float32)


def load_stage1_windows(T: int = 16, link: int = 0, mode: str = "stratified",
                        session: str = "session_001"):
    """给 2D CNN 用：把连续帧组成时间窗，输出 [Nw, C=2, T, S]。

    窗口在同一 (label, batch) 内**非重叠**切分，因此按窗口划分不会造成跨集帧泄漏。
    归一化统计量只在训练窗口的帧上拟合。
    """
    d = load_static(session)
    amp, phase = d["amp"][:, link], d["phase"][:, link]   # [N, S]
    labels, batches = d["label"], d["batch"]

    Xa, Xp, wl, wb = [], [], [], []
    for lbl in np.unique(labels):
        for b in np.unique(batches):
            idx = np.where((labels == lbl) & (batches == b))[0]
            for w in range(len(idx) // T):
                sel = idx[w * T:(w + 1) * T]
                Xa.append(amp[sel]); Xp.append(phase[sel]); wl.append(lbl); wb.append(b)
    Xa, Xp = np.asarray(Xa), np.asarray(Xp)               # [Nw, T, S]
    wl, wb = np.asarray(wl), np.asarray(wb)

    sp = make_split(wl, wb, mode=mode)
    tr = sp["train"]
    S = Xa.shape[-1]
    pp = _preprocessor_for_dataset(d).fit(
        Xa[tr].reshape(-1, 1, S), Xp[tr].reshape(-1, 1, S)
    )

    out = {k: (_apply_pp_windows(pp, Xa[sp[k]], Xp[sp[k]]), wl[sp[k]].astype(np.int64))
           for k in ("train", "val", "test")}
    return out, pp


# ===========================================================================
# 阶段二：多链路（4 条）数据准备
# ===========================================================================
def _label_xy(labels: np.ndarray) -> np.ndarray:
    x, y = config.label_to_xy(np.asarray(labels))
    return np.stack([x, y], axis=1).astype(np.float32)


def load_stage2_frames(mode: str = "stratified", session: str = "session_phase2_001",
                       test_batches=None, val_batches=None,
                       subcarrier_mode: str = "dataset"):
    """逐帧的 4 链路数据（给 NumPy 融合基线用）。

    返回 ({'train','val','test'}: (X[N,L=4,C=2,S], y[N]类别, xy[N,2]连续坐标), preprocessor)。
    """
    if subcarrier_mode not in {"dataset", "train_fold"}:
        raise ValueError("subcarrier_mode 必须是 'dataset' 或 'train_fold'")
    d = load_static_raw64(session) if subcarrier_mode == "train_fold" else load_static(session)
    sp = make_split(d["label"], d["batch"], mode=mode,
                    test_batches=test_batches, val_batches=val_batches)
    if subcarrier_mode == "train_fold":
        sc_idx = select_subcarriers(d["amp"][sp["train"]])
        amp, phase = d["amp"][:, :, sc_idx], d["phase"][:, :, sc_idx]
        pp = Preprocessor(subcarrier_indices=sc_idx)
    else:
        amp, phase = d["amp"], d["phase"]
        pp = _preprocessor_for_dataset(d)
    pp.fit(amp[sp["train"]], phase[sp["train"]])

    def feat(idx):
        a, p = pp.transform(amp[idx], phase[idx])   # [N,4,S]
        X = np.stack([a, p], axis=2).astype(np.float32)       # [N,4,2,S]
        return X, d["label"][idx].astype(np.int64), _label_xy(d["label"][idx])

    return {k: feat(sp[k]) for k in ("train", "val", "test")}, pp


def load_stage2_windows(T: int = 16, mode: str = "stratified", session: str = "session_phase2_001",
                        test_batches=None, val_batches=None,
                        subcarrier_mode: str = "dataset",
                        exclude_batches=None, return_batches: bool = False,
                        ratios=(0.70, 0.15, 0.15)):
    """时间窗的 4 链路数据（给 PyTorch CNN+GAT / 融合基线用）。

    返回 ({'train','val','test'}: (X[Nw,L=4,C=2,T,S], xy[Nw,2]连续坐标, label[Nw]类别), preprocessor)。
    回归目标是连续坐标 (x,y)，输出后可直接送卡尔曼跟踪层；label 供 snap-to-grid 后算
    准确率/混淆矩阵用。

    exclude_batches：这些轮次在划分之前整体剔除，不进 train/val/test，也不参与子载波
    选择和归一化统计（2026-10-08 为 D1 选模型阶段新增：r8–r10 必须完全不可见）。
    return_batches=True 时每个集合多返回第 4 项：每个窗口所属轮次（按轮次分别报成绩用）。
    ratios：只在 mode="stratified"（按窗口随机划分）时生效，默认值与 make_split 相同；
    2026-10-28 为 random_split_check.py 新增（按前人 6:2:2 随机划分做对照）。
    """
    if subcarrier_mode not in {"dataset", "train_fold"}:
        raise ValueError("subcarrier_mode 必须是 'dataset' 或 'train_fold'")
    d = load_static_raw64(session) if subcarrier_mode == "train_fold" else load_static(session)
    labels, batches = d["label"], d["batch"]
    Xa, Xp, wl, wb = [], [], [], []
    for lbl in np.unique(labels):
        for b in np.unique(batches):
            idx = np.where((labels == lbl) & (batches == b))[0]
            for w in range(len(idx) // T):
                sel = idx[w * T:(w + 1) * T]
                Xa.append(d["amp"][sel]); Xp.append(d["phase"][sel])  # [T,4,S]
                wl.append(lbl); wb.append(b)
    Xa, Xp = np.asarray(Xa), np.asarray(Xp)          # [Nw,T,4,S]
    wl, wb = np.asarray(wl), np.asarray(wb)
    if exclude_batches:
        keep = ~np.isin(wb, list(exclude_batches))
        Xa, Xp, wl, wb = Xa[keep], Xp[keep], wl[keep], wb[keep]
    L, S = Xa.shape[2], Xa.shape[3]

    sp = make_split(wl, wb, mode=mode, test_batches=test_batches, val_batches=val_batches,
                    ratios=ratios)
    tr = sp["train"]
    if subcarrier_mode == "train_fold":
        # 子载波结构筛选使用训练批次的全部原始帧（包括不足一个时间窗的尾帧），
        # 既不接触验证/测试批次，也避免窗口丢尾改变死子载波的众数比例。
        train_batches = np.unique(wb[tr])
        frame_train = np.isin(d["batch"], train_batches)   # 只含训练轮，被剔除的轮次自然不在内
        sc_idx = select_subcarriers(d["amp"][frame_train])
        Xa, Xp = Xa[..., sc_idx], Xp[..., sc_idx]
        S = len(sc_idx)
        pp = Preprocessor(subcarrier_indices=sc_idx)
    else:
        pp = _preprocessor_for_dataset(d)
    pp.fit(
        Xa[tr].reshape(-1, L, S), Xp[tr].reshape(-1, L, S)
    )

    def build(idx):
        n = len(idx)
        if n == 0:   # 例如 D1 最终阶段没有验证集：返回形状正确的空数组，不送进预处理
            T_ = Xa.shape[1]
            out = (np.empty((0, L, 2, T_, S), np.float32), np.empty((0, 2), np.float32),
                   np.empty(0, np.int64))
            return out + (np.empty(0, wb.dtype),) if return_batches else out
        a, p = pp.transform(Xa[idx].reshape(-1, L, S), Xp[idx].reshape(-1, L, S))
        a = a.reshape(n, -1, L, S)
        p = p.reshape(n, -1, L, S)
        X = np.stack([a, p], axis=3)                 # [Nw,T,L,C,S]
        X = X.transpose(0, 2, 3, 1, 4)               # [Nw,L,C,T,S]
        out = (X.astype(np.float32), _label_xy(wl[idx]), wl[idx].astype(np.int64))
        return out + (wb[idx],) if return_batches else out

    return {k: build(sp[k]) for k in ("train", "val", "test")}, pp


# ===========================================================================
# 阶段一（新口径）：直接从阶段二的四链路数据按链路切片
# ===========================================================================
# 为什么改：阶段一原先自己采、自己划分，和阶段二不是同一批样本，"单链路 vs 融合"
# 的差异里混着"两次采集环境不同"。改成从同一份四链路数据切片后，两者共享
# 采集轮次、划分、子载波筛选与预处理统计量，差异只剩链路数这一个变量。
# 代价：四链路时间对齐会丢掉约 7% 的帧（某条链路缺帧的时刻整帧作废）。
# 这个代价已在 2026-09-12 明确接受，换取严格单变量对比。
STAGE2_DEFAULT_SESSION = "session_phase2_001"


def load_stage1_frames_from_stage2(link: int = 0, mode: str = "by_batch",
                                   session: str = STAGE2_DEFAULT_SESSION,
                                   test_batches=None, val_batches=None,
                                   subcarrier_mode: str = "train_fold"):
    """给 KNN 用：阶段二四链路逐帧数据切出第 ``link`` 条，输出 [N, C=2, S]。

    返回 ({'train','val','test'}: (X[N,2,S], y[N] 类别), preprocessor)。
    """
    data, pp = load_stage2_frames(mode=mode, session=session, test_batches=test_batches,
                                  val_batches=val_batches, subcarrier_mode=subcarrier_mode)
    n_links = data["train"][0].shape[1]
    if not 0 <= link < n_links:
        raise ValueError(f"link 越界: {link}，数据集只有 {n_links} 条链路")
    # X[N,L,C,S] -> X[N,C,S]
    return {k: (v[0][:, link].copy(), v[1]) for k, v in data.items()}, pp


def load_stage1_windows_from_stage2(T: int = config.STAGE1_TIME_WINDOW, link: int = 0,
                                    mode: str = "by_batch",
                                    session: str = STAGE2_DEFAULT_SESSION,
                                    test_batches=None, val_batches=None,
                                    subcarrier_mode: str = "train_fold"):
    """给 CNN 用：阶段二四链路时间窗数据切出第 ``link`` 条，输出 [Nw, C=2, T, S]。

    返回 ({'train','val','test'}: (X[Nw,2,T,S], y[Nw] 类别), preprocessor)。
    """
    data, pp = load_stage2_windows(T=T, mode=mode, session=session,
                                   test_batches=test_batches, val_batches=val_batches,
                                   subcarrier_mode=subcarrier_mode)
    n_links = data["train"][0].shape[1]
    if not 0 <= link < n_links:
        raise ValueError(f"link 越界: {link}，数据集只有 {n_links} 条链路")
    # X[Nw,L,C,T,S] -> X[Nw,C,T,S]；标签取分类 label（第 3 个返回值），不取连续 xy
    return {k: (v[0][:, link].copy(), v[2]) for k, v in data.items()}, pp
