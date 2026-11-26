"""CSI 预处理 —— 幅度去噪 + 相位校准 + 归一化（归一化统计量只用训练集拟合）。

处理链：
  幅度  amp   → Hampel 去离群点 → Savitzky-Golay 平滑 → 标准化
  相位  phase → 线性去趋势(去 CFO/STO 偏移) → 标准化

关键纪律（防泄漏）：归一化的 mean/std **只能由训练集拟合**（fit），
再用同一套统计量 transform 验证集与测试集。相位去趋势也为未来 ToF 留口。

自检运行：python -m backend.data.preprocess
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

_EPS = 1e-6


# ---------------------------------------------------------------------------
# 子载波编号（2026-10-08 实测确认，BCM43455 + Nexmon，20 MHz）
# ---------------------------------------------------------------------------
# 原始 64 个子载波是 FFT 顺序：编号 1–27 = 物理 +1…+27，36–63 = 物理 −28…−1，0 = DC，
# 28–35 = 频带边缘（D1 实测平均幅度 10315 / 34190 / 134 / 0 / 7428 / 7169 / 22 / 10844，垃圾值）。
# 2026-10-08 之前：死子载波检测漏掉了 32 / 34 / 35（D1 选出 58 个），而且后续去趋势、平滑、
# CNN 卷积都按原始编号顺序处理，在频带正中（编号 27 与 36 之间）多出一个"假相邻"。
RAW_SUB = 64
VALID_SC_HT20 = np.r_[1:28, 36:64]          # 55 个有效子载波（原始编号）


def physical_k(idx) -> np.ndarray:
    """原始编号 -> 物理子载波号（−32…+31）。"""
    idx = np.asarray(idx)
    return np.where(idx < RAW_SUB // 2, idx, idx - RAW_SUB)


def select_subcarriers(amp_raw: np.ndarray, dead_frequency_threshold: float = 0.5) -> np.ndarray:
    """Select structural live subcarriers using training data only.

    ``amp_raw`` may have any leading sample/time/link axes and a final subcarrier axis ``S``.
    The caller is responsible for passing only training batches/folds. Returning physical indices
    keeps the operation reusable across fixed splits and grouped cross-validation.
    """
    amp_raw = np.asarray(amp_raw)
    if amp_raw.ndim < 2 or amp_raw.shape[-1] < 2:
        raise ValueError(f"amp_raw must have one or more sample axes plus subcarrier axis, got {amp_raw.shape}")
    flat = amp_raw.reshape(-1, amp_raw.shape[-1])
    if len(flat) == 0 or not np.isfinite(flat).all():
        raise ValueError("amp_raw must contain finite training samples")
    med = np.median(flat, axis=0)
    ref = np.median(med)
    keep_ratio = (med > 0.1 * ref) & (med < 10.0 * ref)
    top1_frequency = np.array([
        np.unique(flat[:, i], return_counts=True)[1].max() / len(flat)
        for i in range(flat.shape[1])
    ])
    keep = keep_ratio & (top1_frequency <= dead_frequency_threshold)
    if flat.shape[1] == RAW_SUB:               # 64 个原始子载波：只允许有效集合，排除 DC 与频带边缘
        keep &= np.isin(np.arange(RAW_SUB), VALID_SC_HT20)
    selected = np.flatnonzero(keep).astype(np.int64)
    if len(selected) == 0:
        raise ValueError("subcarrier selection removed every subcarrier")
    if flat.shape[1] == RAW_SUB:               # 按物理频率排序（−28…−1, +1…+27）：后续平滑、去趋势、
        selected = selected[np.argsort(physical_k(selected), kind="stable")]   # CNN 卷积沿真实相邻方向
    return selected


# ---------------------------------------------------------------------------
# 单步算子（对最后一维=子载波操作）
# ---------------------------------------------------------------------------
def _rolling_median_mad(x: np.ndarray, w: int):
    """沿最后一维的滚动中位数与 MAD（reflect 填充）。x:[N,S] -> med,mad:[N,S]。"""
    pad = w // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(xp, w, axis=1)  # [N,S,w]
    med = np.median(win, axis=2)
    mad = np.median(np.abs(win - med[..., None]), axis=2)
    return med, mad


def hampel(x: np.ndarray, w: int = 5, n_sigma: float = 3.0) -> np.ndarray:
    """Hampel 去离群：偏离滚动中位数超过 n_sigma×1.4826×MAD 的点用中位数替换。x:[N,S]。"""
    med, mad = _rolling_median_mad(x, w)
    thr = n_sigma * 1.4826 * mad
    out = x.copy()
    mask = np.abs(x - med) > thr
    out[mask] = med[mask]
    return out


def smooth(x: np.ndarray, w: int = 7, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay 平滑（scipy 缺失时回退移动平均）。x:[N,S]。"""
    if x.shape[-1] < w:
        return x
    if _HAS_SCIPY:
        return savgol_filter(x, w, poly, axis=-1).astype(np.float32)
    k = np.ones(w, np.float32) / w
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), -1, x).astype(np.float32)


def sanitize_phase(phase: np.ndarray, subcarrier_indices: np.ndarray | None = None) -> np.ndarray:
    """相位线性去趋势。

    ``subcarrier_indices`` 必须是原始 OFDM 子载波编号。筛掉 guard/DC 后编号通常不连续，
    若错误地用压缩后的 ``0..S-1`` 拟合，会把跨空洞的物理频率距离算错。
    """
    S = phase.shape[-1]
    order = None
    if subcarrier_indices is None:
        k = np.arange(S, dtype=np.float64)
    else:
        idx = np.asarray(subcarrier_indices)
        if idx.ndim != 1 or len(idx) != S:
            raise ValueError(
                f"subcarrier_indices 必须是一维且长度等于相位维度 {S}，当前 {idx.shape}"
            )
        # 2026-10-08：原始编号是 FFT 顺序，必须换成物理子载波号，并按物理频率排序后再展开相位
        k = physical_k(idx).astype(np.float64)
        order = np.argsort(k, kind="stable")
        if np.array_equal(order, np.arange(S)):
            order = None                       # 已按物理频率排好（select_subcarriers 的输出）
        else:
            phase, k = phase[..., order], k[order]
    pu = np.unwrap(phase, axis=-1)
    kbar = k.mean()
    denom = ((k - kbar) ** 2).sum()
    if denom <= _EPS:
        return (pu - pu.mean(-1, keepdims=True)).astype(np.float32)
    slope = ((pu - pu.mean(-1, keepdims=True)) * (k - kbar)).sum(-1, keepdims=True) / denom
    intercept = pu.mean(-1, keepdims=True) - slope * kbar
    out = (pu - (slope * k + intercept)).astype(np.float32)
    if order is not None:                      # 还原成调用方给的子载波顺序
        out = out[..., np.argsort(order)]
    return out


# ---------------------------------------------------------------------------
# 预处理器（fit 只用训练集）
# ---------------------------------------------------------------------------
class Preprocessor:
    """把去噪/校准/归一化封装成 fit/transform，强制「统计量只来自训练集」。"""

    def __init__(self, hampel_w: int = 5, hampel_sigma: float = 3.0,
                 savgol_w: int = 7, savgol_poly: int = 2,
                 subcarrier_indices: np.ndarray | None = None):
        self.hampel_w = hampel_w
        self.hampel_sigma = hampel_sigma
        self.savgol_w = savgol_w
        self.savgol_poly = savgol_poly
        self.subcarrier_indices = (None if subcarrier_indices is None else
                                   np.asarray(subcarrier_indices, dtype=np.int64).copy())
        self.amp_mean = self.amp_std = None
        self.ph_mean = self.ph_std = None
        self.fitted = False

    # -- 内部：逐链路清洗（形状 [N,L,S]）--
    def _clean_amp(self, amp: np.ndarray) -> np.ndarray:
        N, L, S = amp.shape
        flat = amp.reshape(N * L, S)
        flat = hampel(flat, self.hampel_w, self.hampel_sigma)
        flat = smooth(flat, self.savgol_w, self.savgol_poly)
        return flat.reshape(N, L, S)

    def _clean_phase(self, phase: np.ndarray) -> np.ndarray:
        N, L, S = phase.shape
        flat = sanitize_phase(
            phase.reshape(N * L, S), getattr(self, "subcarrier_indices", None)
        )
        return flat.reshape(N, L, S)

    def fit(self, amp: np.ndarray, phase: np.ndarray) -> "Preprocessor":
        a = self._clean_amp(amp)
        p = self._clean_phase(phase)
        self.amp_mean, self.amp_std = a.mean(0), a.std(0) + _EPS   # [L,S]
        self.ph_mean, self.ph_std = p.mean(0), p.std(0) + _EPS
        self.fitted = True
        return self

    def transform(self, amp: np.ndarray, phase: np.ndarray):
        if not self.fitted:
            raise RuntimeError("必须先在训练集上 fit()，再 transform()")
        a = (self._clean_amp(amp) - self.amp_mean) / self.amp_std
        p = (self._clean_phase(phase) - self.ph_mean) / self.ph_std
        return a.astype(np.float32), p.astype(np.float32)

    def fit_transform(self, amp: np.ndarray, phase: np.ndarray):
        return self.fit(amp, phase).transform(amp, phase)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema_version": 2,
            "hampel_w": self.hampel_w, "hampel_sigma": self.hampel_sigma,
            "savgol_w": self.savgol_w, "savgol_poly": self.savgol_poly,
            "subcarrier_indices": (None if self.subcarrier_indices is None else
                                   self.subcarrier_indices.tolist()),
            "amp_mean": self.amp_mean.tolist(), "amp_std": self.amp_std.tolist(),
            "ph_mean": self.ph_mean.tolist(), "ph_std": self.ph_std.tolist(),
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        pp = cls(
            d["hampel_w"], d["hampel_sigma"], d["savgol_w"], d["savgol_poly"],
            subcarrier_indices=d.get("subcarrier_indices"),
        )
        pp.amp_mean = np.array(d["amp_mean"], np.float32)
        pp.amp_std = np.array(d["amp_std"], np.float32)
        pp.ph_mean = np.array(d["ph_mean"], np.float32)
        pp.ph_std = np.array(d["ph_std"], np.float32)
        pp.fitted = True
        return pp


if __name__ == "__main__":
    from backend.data.load import load_static
    from backend.data.split import make_split

    d = load_static()
    sp = make_split(d["label"], d["batch"], seed=42, mode="stratified")
    pp = Preprocessor(subcarrier_indices=d.get("sc_idx"))
    pp.fit(d["amp"][sp["train"]], d["phase"][sp["train"]])
    a_tr, p_tr = pp.transform(d["amp"][sp["train"]], d["phase"][sp["train"]])
    a_te, p_te = pp.transform(d["amp"][sp["test"]], d["phase"][sp["test"]])
    print(f"scipy 可用: {_HAS_SCIPY}")
    print(f"训练集标准化后  amp 均值≈{a_tr.mean():+.3f} 标准差≈{a_tr.std():.3f}  "
          f"(应≈0 / ≈1)")
    print(f"测试集(用训练统计量) amp 均值≈{a_te.mean():+.3f} 标准差≈{a_te.std():.3f}")
    assert abs(a_tr.mean()) < 0.1 and 0.8 < a_tr.std() < 1.2, "训练集归一化异常"
    print("预处理自检通过：去噪+相位校准+归一化，统计量仅来自训练集。")
