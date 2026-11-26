"""数据加载 —— 读取 datasets/raw 下的 .npz 到内存数组。

真实数据到货后，只需在这里新增「原始文件 → 统一数组」的解析入口，
下游（预处理/划分/模型）全部复用，无需改动。
"""

from pathlib import Path

import numpy as np

from backend import config


def load_static(session: str = "session_001") -> dict:
    """读取静态格点数据集。返回含 amp/phase[N,L,S]、label/x/y/batch[N]、links[L] 的 dict。"""
    npz = config.RAW_DIR / session / "static.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"未找到 {npz}；请先采集真实 CSI 并运行 python -m backend.data.load_real {session}"
        )
    d = np.load(npz)
    return {k: d[k] for k in d.files}


def load_static_raw64(session: str = "session_phase2_001") -> dict:
    """读取阶段二保留全部 64 个物理子载波的交叉验证源数据。

    子载波筛选必须在每个训练折内部完成，因此正式阶段二评估不能直接从已经筛选过的
    ``static.npz`` 起步。
    """
    npz = config.RAW_DIR / session / "static_raw64.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"未找到 {npz}；请先运行 python -m backend.data.load_real_stage2 {session} 重建数据"
        )
    d = np.load(npz)
    out = {k: d[k] for k in d.files}
    if out["amp"].ndim != 3 or out["amp"].shape[-1] != 64:
        raise ValueError(f"阶段二 raw64 输入应为 [N,L,64]，当前 {out['amp'].shape}")
    return out


def load_trace(name: str = "trace_001") -> dict:
    """读取行走轨迹。返回含 amp/phase[T,L,S]、gt_x/gt_y/t[T]、links[L] 的 dict。"""
    npz = config.RAW_DIR / name / "trace.npz"
    if not npz.exists():
        raise FileNotFoundError(f"未找到 {npz}")
    d = np.load(npz)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    s = load_static()
    print(f"静态: amp{tuple(s['amp'].shape)} label {s['label'].min()}..{s['label'].max()} "
          f"links={[str(x) for x in s['links']]}")
