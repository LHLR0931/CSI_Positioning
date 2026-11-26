"""真实 CSI 载入 —— nexmon pcap (BCM43455c0, 5GHz ch36/20MHz) → 统一 static.npz。

用法（项目根目录执行）:
    python -m backend.data.load_real            # 默认 session_001
    python -m backend.data.load_real session_002

原始 pcap 源目录（2026-08-31 起，与阶段二 data/phase2_r* 的布局对齐、和 datasets/raw
彻底分开——datasets/raw/<session>/ 现在只放处理产物 static.npz，不再混放原始 pcap）:
    data/phase1_<session>/label_<L>_r<R>.pcap

读取 data/phase1_<session>/label_<L>_r<R>.pcap：
  - L = 采集时的蛇形(serpentine)标签；R = 轮次(round)，作为防泄漏用的 batch
  - 每包解码 64 子载波复数 CSI（幅度 = |H|，与 I/Q 顺序无关）
  - 丢弃空子载波（边缘 guard 近 0、DC/参考子载波极大离群），保留有效子载波 S
  - 蛇形标签 → 物理 (x, y) → 框架用的栅格 label = y*GRID_COLS + x
输出 datasets/raw/<session>/static.npz:
  amp[N,1,S], phase[N,1,S], label[N], x[N], y[N], batch[N], links[1], sc_idx[S]

注意：丢弃空子载波用的是「结构性」判据（哪些子载波是 guard/DC），由 802.11/硬件决定，
不看标签，因此不构成数据泄漏；统计量标准化仍由 preprocess 只在训练集上拟合。
"""
from __future__ import annotations

import glob
import io
import re
import struct
import sys

import numpy as np

from backend import config

# nexmon CSI 包内偏移：Ethernet(14) + IP(20) + UDP(8) + nexmon 头(18) = 60
CSI_OFF = 60
RAW_SUB = 64
CSI_LEN = RAW_SUB * 4          # 64 子载波 × (int16 虚部 + int16 实部)

# 阶段一原始 pcap 源目录：data/phase1_<session>/，与 config.RAW_DIR（只放 static.npz 等
# 处理产物）分开，和阶段二 load_real_stage2.py 的 SRC_DIR 是同一种布局思路。
SRC_DIR = config.PROJECT_ROOT / "data"

_PAT = re.compile(r"label_(\d+)_r(\d+)\.pcap$")


def _read_pcap_stream(f) -> np.ndarray:
    """Decode a classic pcap byte stream into complex CSI shaped [N, 64]."""
    frames = []
    header = f.read(24)
    if len(header) < 24:
        return np.empty((0, RAW_SUB), np.complex64)
    magic = header[:4]
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        endian = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        endian = ">"
    else:
        return np.empty((0, RAW_SUB), np.complex64)

    while True:
        rh = f.read(16)                                # 每包记录头
        if len(rh) < 16:
            break
        _, _, incl, _ = struct.unpack(f"{endian}IIII", rh)
        if incl > 1_000_000:                           # reject a corrupt packet length
            return np.empty((0, RAW_SUB), np.complex64)
        data = f.read(incl)
        if len(data) < incl:
            break
        if len(data) < CSI_OFF + CSI_LEN:
            continue
        v = np.frombuffer(data[CSI_OFF:CSI_OFF + CSI_LEN], dtype="<i2")
        v = v.astype(np.float32).reshape(RAW_SUB, 2)
        frames.append(v[:, 1] + 1j * v[:, 0])          # real=second int16, imag=first
    return np.asarray(frames, np.complex64)


def read_pcap_bytes(payload: bytes) -> np.ndarray:
    """Decode an uploaded pcap without writing a temporary file."""
    return _read_pcap_stream(io.BytesIO(payload))


def _read_pcap(path: str) -> np.ndarray:
    """Decode one pcap file into complex CSI shaped [N, 64]."""
    with open(path, "rb") as f:
        return _read_pcap_stream(f)


def _serp_to_xy(label: int, cols: int) -> tuple[int, int]:
    """蛇形标签 → 物理 (x, y)：偶数排正序、奇数排逆序（与采集脚本一致）。"""
    y = label // cols
    off = label - y * cols
    x = off if (y % 2 == 0) else (cols - 1 - off)
    return x, y


def build(session: str = "session_001"):
    src_dir = SRC_DIR / f"phase1_{session}"
    out_dir = config.RAW_DIR / session
    files = sorted(glob.glob(str(src_dir / "label_*_r*.pcap")))
    if not files:
        raise FileNotFoundError(f"{src_dir} 下没有 label_*_r*.pcap，请先把采集的 pcap 拷进去。")

    csis, labs, xs, ys, batches = [], [], [], [], []
    for fp in files:
        m = _PAT.search(fp.replace("\\", "/"))
        if not m:
            continue
        l_serp, rnd = int(m.group(1)), int(m.group(2))
        x, y = _serp_to_xy(l_serp, config.GRID_COLS)
        lab = config.xy_to_label(x, y)                 # 栅格 label
        c = _read_pcap(fp)
        if c.shape[0] == 0:
            print("  跳过(空):", fp)
            continue
        n = c.shape[0]
        csis.append(c)
        labs.append(np.full(n, lab, np.int64))
        xs.append(np.full(n, x, np.int64))
        ys.append(np.full(n, y, np.int64))
        batches.append(np.full(n, f"r{rnd}", dtype="<U8"))
        name = fp.replace("\\", "/").split("/")[-1]
        print(f"  {name:22s} n={n:4d}  serp={l_serp} -> (x={x},y={y}) label={lab}  batch=r{rnd}")

    csi = np.concatenate(csis, axis=0)                 # [N, 64] complex
    label = np.concatenate(labs)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    batch = np.concatenate(batches)

    # --- 只用训练轮次选择子载波，最后一轮是当前 by_batch 历史测试轮 ---
    unique_batches = sorted(set(batch.tolist()), key=lambda v: int(v[1:]))
    selection_batches = unique_batches[:-1]
    if not selection_batches:
        raise RuntimeError("至少需要两个采集轮次才能隔离子载波选择与测试集")
    selection_csi = csi[np.isin(batch, selection_batches)]
    med = np.median(np.abs(selection_csi), axis=0)      # [64] 训练轮次每子载波幅度中位数
    ref = np.median(med)
    keep = (med > 0.1 * ref) & (med < 10.0 * ref)
    sc_idx = np.where(keep)[0]
    S = int(keep.sum())
    print(f"\n训练轮次有效子载波 {S}/64（选择轮次={selection_batches}）: {sc_idx.tolist()}")

    amp = np.abs(csi[:, keep]).astype(np.float32)[:, None, :]      # [N,1,S]
    phase = np.angle(csi[:, keep]).astype(np.float32)[:, None, :]  # [N,1,S] 原始相位, preprocess 再校准
    links = np.array(["piA_apX"])

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "static.npz"
    np.savez(out, amp=amp, phase=phase, label=label, x=x, y=y,
             batch=batch, links=links, sc_idx=sc_idx,
             subcarrier_selection_batches=np.array(selection_batches))

    print(f"\n已保存 {out}")
    print(f"  amp{amp.shape}  phase{phase.shape}  N={amp.shape[0]}")
    print(f"  classes={sorted(np.unique(label).tolist())}  batches={sorted(set(batch.tolist()))}")
    cls_missing = sorted(set(range(config.NUM_CLASSES)) - set(np.unique(label).tolist()))
    if cls_missing:
        print(f"  注意：缺少类别 {cls_missing}（对应 pcap 没采到）")
    if S != config.STAGE1_NUM_SUBCARRIERS:
        print(f"  警告：当前 S={S} 与 STAGE1_NUM_SUBCARRIERS={config.STAGE1_NUM_SUBCARRIERS} 不一致")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "session_001")
