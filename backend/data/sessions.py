"""采集 session 的唯一登记处：原始 pcap 在哪、网格多大、各轮次做什么用、哪些点被跳过。

2026-10-07 起有两种数据布局，所有读原始 pcap 的代码都必须经过这里，不得自己 glob：

    旧布局（pilot，10 点 / 1.0 m）：
        data/phase2_r<R>/label<L>_pi<P>.pcap          session 名 = session_phase2_001
        data/phase1_<session>/label_<L>_r<R>.pcap    session 名 = session_001 / session_002

    新布局（phase2_capture.ps1 2026-09-20 起，15 点 / 0.8 m）：
        data/<session>/phase2_r<R>/label<L>_pi<P>.pcap  session 名 = d1 / d3 / d5 / smoke
        data/<session>/session_meta.json                 网格、设备坐标、每轮 Pi 温度/开机时间/跳过的点

新布局的网格只认 session_meta.json，不读全局 config —— 全局 config 改过一次间距就把旧数据
的米制误差全部算错过（2026-09-12 的 ×0.8 事故）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from backend import config

ROUND_PAT = re.compile(r"phase2_r(\d+)$")


@dataclass(frozen=True)
class Grid:
    cols: int
    rows: int
    spacing_m: float

    @property
    def num_classes(self) -> int:
        return self.cols * self.rows

    def label_to_xy(self, label: int) -> tuple[int, int]:
        return label % self.cols, label // self.cols


# 旧数据的网格是历史事实，写死在这里，与全局 config 无关
LEGACY_GRID_5x2_1m = Grid(cols=5, rows=2, spacing_m=1.0)
LEGACY_SESSIONS: dict[str, Grid] = {
    "session_001": LEGACY_GRID_5x2_1m,
    "session_002": LEGACY_GRID_5x2_1m,
    "session_phase2_001": LEGACY_GRID_5x2_1m,
}
LEGACY_STAGE2_SESSION = "session_phase2_001"

# 新 session 的轮次用途。没登记的 session 全部轮次都是测试。
# 跨日 session（d3 / d5）的用途由预登记 report/健康分预登记.md（2026-10-11 冻结）第 2 节规定：
#   calib   = 当天标定轮（并入当天模型的训练集，也是健康分的参照）
#   test    = 主验证 t1/t2/t3（进成功判据）
#   perturb = 受控扰动轮（单独报告，不进成功判据）
#   ref     = 不进主分析的参考轮
#   train/val 为空：跨日 session 不参与子载波选择（解析时用 --sc-from d1）。
# d3 的 session_meta.json 里 r2 有 4 条记录：前 3 条是上午作废的尝试（目录已改名 anomaly_r2*，
# finished_utc=null），最后一条是正式标定轮；四条都没有 skipped_labels，skipped_points 不受影响。
SESSION_ROUND_ROLES: dict[str, dict[str, tuple[str, ...]]] = {
    "d1": {
        "train": ("r1", "r2", "r3", "r4", "r5", "r6"),
        "val": ("r7",),
        "calib": ("r8",),   # 健康分的干净基线
        "test": ("r9", "r10"),
    },
    "d3": {
        "train": (), "val": (),
        "ref": ("r1",),                 # 两台 Pi 上一次开机的标定轮（之后两台都重启 + 驱动重载过）
        "calib": ("r2",),               # 10:41–10:56
        "test": ("r3", "r4", "r5"),     # t1 / t2 / t3，中点约在标定结束后 8 / 24 / 40 分钟
        "perturb": ("r6",),             # 水瓶：piB 朝网格侧 20 cm、瓶中心高 0.6 m
    },
    # d5：采完后按实际轮次编号登记（预登记第 10 节）。
}


def config_grid() -> Grid:
    """全局 config 的网格。**只**用于模拟数据自检；真实数据一律用 session_grid(session)。"""
    return Grid(cols=config.GRID_COLS, rows=config.GRID_ROWS, spacing_m=config.GRID_SPACING_M)


def session_dir(session: str) -> Path:
    return config.SOURCE_DATA_DIR / session


def is_new_layout(session: str) -> bool:
    d = session_dir(session)
    return d.is_dir() and any(ROUND_PAT.search(p.name) for p in d.glob("phase2_r*") if p.is_dir())


def is_stage2(session: str) -> bool:
    return session.startswith("session_phase2") or is_new_layout(session)


def stage2_round_dirs(session: str) -> list[Path]:
    """该 session 的 phase2_r<R> 目录，按轮次号排序。找不到时报错，绝不退回别的目录。"""
    if session == LEGACY_STAGE2_SESSION:
        root = config.SOURCE_DATA_DIR
    elif is_new_layout(session):
        root = session_dir(session)
    else:
        raise FileNotFoundError(
            f"找不到 session '{session}' 的原始数据：{session_dir(session)} 下没有 phase2_r* 目录。"
            f"旧 pilot 数据请用 session 名 '{LEGACY_STAGE2_SESSION}'。"
        )
    dirs = [p for p in root.glob("phase2_r*") if p.is_dir() and ROUND_PAT.search(p.name)]
    return sorted(dirs, key=lambda p: int(ROUND_PAT.search(p.name).group(1)))


def source_pcap_files(session: str) -> list[Path]:
    """该 session 的全部原始 pcap（manifest 与模型包哈希共用这一份清单）。"""
    if is_stage2(session):
        try:
            dirs = stage2_round_dirs(session)
        except FileNotFoundError:
            return []
        return sorted(f for d in dirs for f in d.glob("label*_pi*.pcap"))
    return sorted((config.SOURCE_DATA_DIR / f"phase1_{session}").glob("label_*_r*.pcap"))


def read_session_meta(session: str) -> dict | None:
    """读 data/<session>/session_meta.json。PowerShell 5.1 写的是带 BOM 的 UTF-8。"""
    path = session_dir(session) / "session_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def session_grid(session: str) -> Grid:
    """该 session 实际采集时的网格。旧 session 查登记表；新 session 只认 session_meta.json。"""
    if session in LEGACY_SESSIONS:
        return LEGACY_SESSIONS[session]
    meta = read_session_meta(session)
    if meta is None or "grid" not in meta:
        raise ValueError(
            f"session '{session}' 没有 session_meta.json 或其中没有 grid 字段，"
            f"无法确定网格间距（不允许回退到全局 config）。"
        )
    g = meta["grid"]
    return Grid(cols=int(g["Width"]), rows=int(g["Height"]), spacing_m=float(g["Spacing"]))


def round_roles(session: str) -> dict[str, tuple[str, ...]] | None:
    """轮次用途；旧 session 返回 None（沿用 config.STAGE2_VAL/TEST_BATCHES 的历史划分）。"""
    if session in LEGACY_SESSIONS:
        return None
    if session in SESSION_ROUND_ROLES:
        return SESSION_ROUND_ROLES[session]
    return {"train": (), "val": (), "calib": (), "test": ("*",)}


def _as_list(v) -> list:
    """PowerShell 5.1 的 ConvertTo-Json 偶尔把单元素数组写成标量/对象，这里统一成 list。"""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def skipped_points(session: str) -> set[tuple[int, int]]:
    """采集时被明确 skip 的 (round, label)。

    以覆盖该点的**最后一条** round 记录为准：某点先被 skip、后来用 -Only -Force 重采成功，
    重采那条记录覆盖它，该点重新有效。
    """
    meta = read_session_meta(session)
    if not meta:
        return set()
    n_classes = session_grid(session).num_classes
    status: dict[tuple[int, int], bool] = {}
    for entry in _as_list(meta.get("rounds")):
        rnd = int(entry["round"])
        only = entry.get("only_label")
        skipped = {int(v) for v in _as_list(entry.get("skipped_labels"))}
        covered = range(n_classes) if only is None else [int(only)]
        for lab in covered:
            status[(rnd, lab)] = lab in skipped
    return {k for k, v in status.items() if v}
