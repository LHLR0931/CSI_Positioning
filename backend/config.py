"""项目中央配置。

阶段一和阶段二的数据形状不同，必须使用各自的显式常量。保留的
``NUM_SUBCARRIERS``/``TIME_WINDOW`` 只是阶段一 API 的兼容别名，新的阶段二代码不得
使用它们推断输入形状；阶段二真实形状由数据集 manifest 与 ``sc_idx`` 决定。
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
SOURCE_DATA_DIR = PROJECT_ROOT / "data"       # 原始 PCAP，只读、禁止手工修改
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASET_ARTIFACTS_DIR = DATASETS_DIR / "raw"  # 历史目录名；内容实际为 static.npz 等派生产物
RAW_DIR = DATASET_ARTIFACTS_DIR               # 向后兼容旧调用方，勿再把它描述成原始 PCAP
PROCESSED_DIR = DATASETS_DIR / "processed"   # 预留，当前未使用
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"   # 训练产物(模型文件)

# ---------------------------------------------------------------------------
# 定位网格（5 × 3 = 15 个格点）
# ---------------------------------------------------------------------------
GRID_COLS = 5           # x 方向格点数, x = 0..4
GRID_ROWS = 3           # y 方向格点数, y = 0..2
NUM_CLASSES = GRID_COLS * GRID_ROWS  # 15
GRID_SPACING_M = 0.8    # 相邻格点物理间距(米)；导师 2026-09-12 定为 0.8 m


def label_to_xy(label: int) -> tuple[int, int]:
    """类别标签 → 网格坐标 (x, y)。"""
    return label % GRID_COLS, label // GRID_COLS


def xy_to_label(x: int, y: int) -> int:
    """网格坐标 (x, y) → 类别标签。"""
    return y * GRID_COLS + x


# ---------------------------------------------------------------------------
# CSI 形状（按阶段明确区分）
# ---------------------------------------------------------------------------
NUM_ANTENNAS = 1        # Pi 4B 板载单天线
CSI_CHANNELS = 2        # 每个子载波两个通道：幅度 + 相位
# 阶段一改为从阶段二四链路数据按链路切片取，故与阶段二共用同一套死子载波判据：
# 旧值 58 里有 3 个是 guard 区量化地板值，阶段二更严格的判据已将其剔除。
STAGE1_NUM_SUBCARRIERS = 55
STAGE1_TIME_WINDOW = 8
# 仅表示当前固定成绩单所用数据集的观测值；训练/加载必须以 sc_idx/manifest 为准，
# 不得把 55 当成模型结构常量（交叉验证各折可能筛出不同数量）。
STAGE2_NUM_SUBCARRIERS = 55
STAGE2_TIME_WINDOW = 16

# 阶段一在线 API 仍使用这两个旧名称。
NUM_SUBCARRIERS = STAGE1_NUM_SUBCARRIERS
TIME_WINDOW = STAGE1_TIME_WINDOW

# ---------------------------------------------------------------------------
# 链路（阶段一单链路；阶段二 2 Pi × 2 AP = 4 链路）
# ---------------------------------------------------------------------------
PI_IDS = ["piA", "piB"]
AP_IDS = ["apX", "apY"]
PI_IPS = {
    "piA": os.environ.get("CSI_PIA_HOST", "").strip(),
    "piB": os.environ.get("CSI_PIB_HOST", "").strip(),
}
AP_BSSIDS = {
    "apX": os.environ.get("CSI_APX_BSSID", "").strip(),
    "apY": os.environ.get("CSI_APY_BSSID", "").strip(),
}


def _require_values(values: dict[str, str], environment: dict[str, str], purpose: str) -> dict[str, str]:
    missing = [environment[name] for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing {purpose} configuration. Set: {', '.join(missing)}"
        )
    return values.copy()


def require_pi_ips() -> dict[str, str]:
    """Return the configured receiver hosts or raise a clear setup error."""
    return _require_values(
        PI_IPS,
        {"piA": "CSI_PIA_HOST", "piB": "CSI_PIB_HOST"},
        "receiver host",
    )


def require_ap_bssids() -> dict[str, str]:
    """Return the configured transmitter BSSIDs or raise a clear setup error."""
    return _require_values(
        AP_BSSIDS,
        {"apX": "CSI_APX_BSSID", "apY": "CSI_APY_BSSID"},
        "AP BSSID",
    )


def all_link_ids() -> list[str]:
    """阶段二全部链路 id，形如 'piA_apX'。"""
    return [f"{pi}_{ap}" for pi in PI_IDS for ap in AP_IDS]


NUM_LINKS_STAGE1 = 1
NUM_LINKS_STAGE2 = len(PI_IDS) * len(AP_IDS)  # 4

# ---------------------------------------------------------------------------
# 数据划分（防泄漏）
# ---------------------------------------------------------------------------
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15
RANDOM_SEED = 42

# 当前阶段二成绩单使用的历史固定划分。r7/r8 已被反复评估，不能再称为未知最终测试集；
# 后续正式结论应改用按轮次 grouped cross-validation 或新的未接触 session。
STAGE2_VAL_BATCHES = ("r6",)
STAGE2_TEST_BATCHES = ("r7", "r8")

# ---------------------------------------------------------------------------
# 射频采集参数（需与 GL.iNet GL-SFT1200 后台设置一致）
# ---------------------------------------------------------------------------
BAND = "5GHz"
CHANNEL = 36            # ch36 或 ch149
BANDWIDTH_MHZ = 20      # 已在 Pi 上用 ch36/20 验证抓包成功（csi_test.pcap）

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
STAGE1_MODELS = ["KNN", "CNN"]
STAGE2_MODELS = ["SingleLinkCNN", "ConcatMLP", "MeanCoordinateFusion", "GAT", "HeteroGAT"]
ALL_MODELS = STAGE1_MODELS + STAGE2_MODELS


def summary() -> str:
    """返回关键配置摘要，供人工核对。"""
    return (
        "CSI Positioning 配置摘要\n"
        f"  网格           : {GRID_COLS} × {GRID_ROWS} = {NUM_CLASSES} 类, 间距 {GRID_SPACING_M} m\n"
        f"  阶段一输入     : 子载波 {STAGE1_NUM_SUBCARRIERS}, 时窗 {STAGE1_TIME_WINDOW}\n"
        f"  阶段二输入     : 子载波由每折 sc_idx 决定（当前参考 {STAGE2_NUM_SUBCARRIERS}）, "
        f"时窗 {STAGE2_TIME_WINDOW}\n"
        f"  链路           : 阶段一 {NUM_LINKS_STAGE1} 条; 阶段二 {NUM_LINKS_STAGE2} 条 -> {all_link_ids()}\n"
        f"  阶段二旧成绩单 : val {list(STAGE2_VAL_BATCHES)}, test {list(STAGE2_TEST_BATCHES)}\n"
        f"  射频           : {BAND} ch{CHANNEL} {BANDWIDTH_MHZ}MHz\n"
        f"  模型           : 阶段一 {STAGE1_MODELS}; 阶段二 {STAGE2_MODELS}\n"
        f"  原始 PCAP      : {SOURCE_DATA_DIR}\n"
        f"  数据集派生产物 : {DATASET_ARTIFACTS_DIR}\n"
    )


if __name__ == "__main__":
    # 自检：打印配置 + 验证坐标映射自洽
    print(summary())
    for lbl in range(NUM_CLASSES):
        x, y = label_to_xy(lbl)
        assert xy_to_label(x, y) == lbl, f"坐标映射不自洽: label {lbl}"
    print(f"坐标映射自检通过：{NUM_CLASSES} 个标签往返一致。")
