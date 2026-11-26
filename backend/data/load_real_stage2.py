"""真实 CSI 载入（阶段二）—— 2 Pi × 2 AP 四链路 nexmon pcap → 统一 static.npz。

采集方式与阶段一不同：每台 Pi 一份 pcap，混采了两个 AP 的 CSI（未按 MAC 过滤），
靠 802.11 帧内 BSSID（不是外层 Nexmon 伪 MAC）区分 AP1/AP2。四条链路 =
{piA_apX, piA_apY, piB_apX, piB_apY}，piA/piB 对应 pi1/pi2 两份 pcap 各自拆出的
两条子流。

链路使用逻辑名称 piA / piB 与 apX / apY。实际 AP BSSID 由环境变量
CSI_APX_BSSID 和 CSI_APY_BSSID 提供，不写入源码。

用法（项目根目录执行）:
    python -m backend.data.load_real_stage2 session_phase2_001   # 旧 pilot：data/phase2_r*/（10 点 / 1.0 m）
    python -m backend.data.load_real_stage2 d1                   # 新布局：data/d1/phase2_r*/（15 点 / 0.8 m）
    python -m backend.data.load_real_stage2 d3 --sc-from d1      # 跨日 session：子载波沿用 D1 训练轮的选择
    python -m backend.data.load_real_stage2 smoke --sc-from self # 只做可读性检查

数据位置、网格、轮次用途、被跳过的点全部由 backend/data/sessions.py 决定（2026-10-07 起）。
新 session 的网格只读 data/<session>/session_meta.json，不读全局 config。

读取 <session 目录>/phase2_r<R>/label<L>_pi<P>.pcap（P=1,2）：
  - L 已经是网格 label（= y*GRID_COLS + x），阶段二文件名不是蛇形序号，不需要转换。
  - 每份 pcap 逐包解码，按 data[48:52] 的 4 字节判别值分流到 apX/apY；
    不属于 AP1/AP2 的帧（同信道上其他设备的 CSI）直接丢弃。
  - 行对齐：同一 (round, label) 下 4 条链路按包序号截断到最短长度对齐。原始 PCAP
    时间戳同时保存到输出，当前静态实验明确标记为 ``ordinal_static``；未来动态轨迹不得
    继续沿用这一对齐模式。
    依据：device-free 静态感知场景下，每个格点站立约 20 秒完成一次 episode 采集，
    信道相干时间远大于这一采集窗口（人体准静止 → 信道准静态），因此同一 episode 内
    任意包描述的都是同一准静态信道，按包序号对齐即可；时间戳级精确配对只有在信道
    within-window 快速变化（人体走动）的场景下才必要，这里不适用。
  - 子载波选择：只用训练轮次汇总四链路计算，验证/测试轮次不参与选择，避免评估信息
    泄漏。四链路共用同一个 sc_idx，保证链路间子载波对齐。额外加一条死子载波判据：
    众数占比(该子载波最常见取值的帧数占比)>0.5
    则剔除——guard/DC 的幅度判据抓不住"量化地板值恒定、偶发大幅尖峰"型死子载波
    （2026-08-31 诊断发现 sc=30 属于此类：99.7%-100% 的帧恒为 128.0，但因偶发跳
    到 1152.0 导致 std/CV 被拉高、幅度判据误判为存活）。

输出 datasets/raw/<session>/static.npz:
  amp[N,4,S], phase[N,4,S], timestamp[N,4], label[N], x[N], y[N], batch[N],
  links[4], bssid[4], sc_idx[S], alignment_mode, subcarrier_selection_batches
  batch = "r<round>"，与阶段一同样用于 by_batch 防泄漏切分。
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np

from backend import config
from backend.data.load_real import CSI_LEN, CSI_OFF, RAW_SUB
from backend.data import sessions
from backend.data.preprocess import select_subcarriers

PI_TO_ID = {"pi1": "piA", "pi2": "piB"}
LINK_ORDER = config.all_link_ids()      # ["piA_apX", "piA_apY", "piB_apX", "piB_apY"]

_FILE_PAT = re.compile(r"label(\d+)_pi(\d+)\.pcap$")
_ROUND_PAT = sessions.ROUND_PAT


def _ap_discriminators() -> dict[bytes, str]:
    """Map the four BSSID bytes stored in the CSI payload to logical AP IDs."""
    result: dict[bytes, str] = {}
    for ap_id, value in config.require_ap_bssids().items():
        compact = re.sub(r"[^0-9A-Fa-f]", "", value)
        if len(compact) != 12:
            raise ValueError(
                f"Invalid BSSID for {ap_id}; expected 12 hexadecimal digits."
            )
        discriminator = bytes.fromhex(compact[-8:])
        if discriminator in result:
            raise ValueError("Configured AP BSSIDs have the same payload discriminator.")
        result[discriminator] = ap_id
    return result


def _empty_stream() -> dict[str, np.ndarray]:
    return {"csi": np.empty((0, RAW_SUB), np.complex64),
            "timestamp": np.empty(0, np.float64)}


def _read_pcap_split(
    f, ap_discriminators: dict[bytes, str] | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """按 payload 内 BSSID 拆分 CSI，并保留每个有效包的 PCAP 时间戳。

    复用阶段一 load_real._read_pcap_stream 同一套 magic/record-header/CSI_OFF 判据，
    额外读 data[48:52] 的 4 字节判别值做 AP 归属；不属于 apX/apY 的帧（其他设备）丢弃。
    """
    if ap_discriminators is None:
        ap_discriminators = _ap_discriminators()
    out: dict[str, list[np.ndarray]] = {"apX": [], "apY": []}
    timestamps: dict[str, list[float]] = {"apX": [], "apY": []}
    header = f.read(24)
    if len(header) < 24:
        return {k: _empty_stream() for k in out}
    magic = header[:4]
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        endian = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        endian = ">"
    else:
        return {k: _empty_stream() for k in out}
    timestamp_scale = 1e-9 if magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d") else 1e-6

    while True:
        rh = f.read(16)
        if len(rh) < 16:
            break
        ts_sec, ts_frac, incl, _ = struct.unpack(f"{endian}IIII", rh)
        if incl > 1_000_000:
            break
        data = f.read(incl)
        if len(data) < incl:
            break
        if len(data) < CSI_OFF + CSI_LEN:
            continue
        ap_id = ap_discriminators.get(data[48:52])
        if ap_id is None:
            continue
        v = np.frombuffer(data[CSI_OFF:CSI_OFF + CSI_LEN], dtype="<i2")
        v = v.astype(np.float32).reshape(RAW_SUB, 2)
        out[ap_id].append(v[:, 1] + 1j * v[:, 0])
        timestamps[ap_id].append(ts_sec + ts_frac * timestamp_scale)

    return {
        k: {
            "csi": (np.asarray(v, np.complex64) if v else np.empty((0, RAW_SUB), np.complex64)),
            "timestamp": np.asarray(timestamps[k], np.float64),
        }
        for k, v in out.items()
    }


def _selection_rounds(session: str, present_rounds: set[int]) -> list[int]:
    """哪些轮次可以参与子载波选择：只能是训练轮，验证/标定/测试轮一律不参与。"""
    roles = sessions.round_roles(session)
    if roles is None:   # 旧 pilot：沿用当时的历史划分
        held_out = set(config.STAGE2_VAL_BATCHES) | set(config.STAGE2_TEST_BATCHES)
        return sorted(r for r in present_rounds if f"r{r}" not in held_out)
    train = set(roles["train"])
    return sorted(r for r in present_rounds if f"r{r}" in train)


def build(session: str = "session_phase2_001", sc_from: str | None = None):
    ap_discriminators = _ap_discriminators()
    round_dirs = sessions.stage2_round_dirs(session)
    if not round_dirs:
        raise FileNotFoundError(f"session '{session}' 下没有 phase2_r* 目录，请先确认采集数据路径。")
    grid = sessions.session_grid(session)
    skipped = sessions.skipped_points(session)
    print(f"session={session}  源目录={round_dirs[0].parent}  "
          f"网格={grid.cols}x{grid.rows} @ {grid.spacing_m} m  轮次={[p.name for p in round_dirs]}")
    if skipped:
        print(f"  采集时被 skip 的点（不进数据集）: {sorted(skipped)}")

    # --- 第一遍：逐 pcap 解码，按 (round,label) 汇总四条链路各自的复数 CSI 流 ---
    raw: dict[tuple[int, int], dict[str, dict[str, np.ndarray]]] = {}
    for rd in round_dirs:
        rnd = int(_ROUND_PAT.search(rd.name).group(1))
        for fp in sorted(rd.glob("label*_pi*.pcap")):
            m = _FILE_PAT.search(fp.name)
            if not m:
                continue
            lab, pi_num = int(m.group(1)), int(m.group(2))
            if lab >= grid.num_classes:
                raise ValueError(f"{fp} 的 label={lab} 超出网格 {grid.cols}x{grid.rows}")
            if (rnd, lab) in skipped:
                continue
            pi_id = PI_TO_ID.get(f"pi{pi_num}")
            if pi_id is None:
                print(f"  跳过(未知 pi 编号): {fp}")
                continue
            with fp.open("rb") as f:
                split = _read_pcap_split(f, ap_discriminators)
            key = (rnd, lab)
            raw.setdefault(key, {})
            for ap_id, stream in split.items():
                raw[key][f"{pi_id}_{ap_id}"] = stream

    # --- 子载波选择：只汇总训练轮次，验证/标定/测试轮次不参与任何筛选判据 ---
    present_rounds = {rnd for rnd, _ in raw}
    if sc_from is not None and sc_from != "self":
        ref = config.RAW_DIR / sc_from / "static.npz"
        if not ref.exists():
            raise FileNotFoundError(f"--sc-from {sc_from}：找不到 {ref}，请先构建该 session")
        with np.load(ref) as d:
            sc_idx = np.asarray(d["sc_idx"], dtype=np.int64)
            selection_batches = [f"{sc_from}:{b}" for b in d["subcarrier_selection_batches"].astype(str)]
        print(f"子载波沿用 {sc_from} 的选择：{len(sc_idx)}/{RAW_SUB}")
    else:
        if sc_from == "self":
            selection_rounds = sorted(present_rounds)
            print("  ⚠ --sc-from self：所有轮次都参与子载波选择，只能用于可读性检查，不能用于评估")
        else:
            selection_rounds = _selection_rounds(session, present_rounds)
            if not selection_rounds:
                raise RuntimeError(
                    f"session '{session}' 当前没有可用于子载波选择的训练轮次（已有轮次 {sorted(present_rounds)}）。"
                    f"跨日 session 用 --sc-from d1；只做可读性检查用 --sc-from self。"
                )
        pooled = np.concatenate(
            [stream["csi"] for (rnd, _), links in raw.items() if rnd in selection_rounds
             for stream in links.values() if stream["csi"].shape[0] > 0], axis=0
        )
        sc_idx = select_subcarriers(np.abs(pooled))
        selection_batches = [f"r{r}" for r in selection_rounds]
        print(f"有效子载波 {len(sc_idx)}/{RAW_SUB}（选择轮次={selection_rounds}）: {sc_idx.tolist()}")

    # --- 第二遍：按 (round,label) 四链路截断到最短长度、对齐、拼接 ---
    amps_raw, phases_raw = [], []
    timestamps_out, labs, xs, ys, batches = [], [], [], [], []
    trunc_report = []
    for (rnd, lab) in sorted(raw):
        links = raw[(rnd, lab)]
        missing = [lid for lid in LINK_ORDER if lid not in links]
        if missing:
            print(f"  跳过 round={rnd} label={lab}: 缺链路 {missing}")
            continue
        counts = {lid: links[lid]["csi"].shape[0] for lid in LINK_ORDER}
        n = min(counts.values())
        trunc_report.append((rnd, lab, counts, n))
        if n == 0:
            print(f"  跳过 round={rnd} label={lab}: 某链路 0 包 {counts}")
            continue
        stacked = np.stack([links[lid]["csi"][:n] for lid in LINK_ORDER], axis=1)
        stacked_t = np.stack([links[lid]["timestamp"][:n] for lid in LINK_ORDER], axis=1)
        x, y = grid.label_to_xy(lab)
        amps_raw.append(np.abs(stacked).astype(np.float32))                 # [n,4,64]
        phases_raw.append(np.angle(stacked).astype(np.float32))            # [n,4,64]
        timestamps_out.append(stacked_t)
        labs.append(np.full(n, lab, np.int64))
        xs.append(np.full(n, x, np.int64))
        ys.append(np.full(n, y, np.int64))
        batches.append(np.full(n, f"r{rnd}", dtype="<U8"))

    if not amps_raw:
        raise RuntimeError("没有任何 (round,label) 组四链路齐全，检查采集数据。")

    amp_raw = np.concatenate(amps_raw, axis=0)
    phase_raw = np.concatenate(phases_raw, axis=0)
    amp = amp_raw[:, :, sc_idx]
    phase = phase_raw[:, :, sc_idx]
    timestamp = np.concatenate(timestamps_out, axis=0)
    label = np.concatenate(labs)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    batch = np.concatenate(batches)
    links_arr = np.array(LINK_ORDER)
    bssid_arr = np.array([lid.split("_")[1] for lid in LINK_ORDER])

    # --- 截断损失报告 ---
    print("\n(round,label) 四链路原始包数 vs 截断后长度：")
    lost_total, raw_total = 0, 0
    for rnd, lab, counts, n in trunc_report:
        raw_sum = sum(counts.values())
        lost = raw_sum - n * 4
        lost_total += lost
        raw_total += raw_sum
        print(f"  r{rnd} label{lab}: " +
              " ".join(f"{lid}={counts[lid]}" for lid in LINK_ORDER) +
              f"  -> 截断到 n={n}  (丢弃 {lost} 包)")
    print(f"\n合计：原始 {raw_total} 包，截断后保留 {amp.shape[0] * 4} 包，丢弃 {lost_total} 包"
          f"（{100*lost_total/raw_total:.1f}%）")

    sess_dir = config.RAW_DIR / session
    sess_dir.mkdir(parents=True, exist_ok=True)
    # 网格随数据一起存：评价程序必须从这里读间距，不能读全局 config
    grid_fields = dict(
        grid_cols=np.array(grid.cols), grid_rows=np.array(grid.rows),
        grid_spacing_m=np.array(grid.spacing_m),
    )
    out = sess_dir / "static.npz"
    np.savez(
        out, amp=amp, phase=phase, timestamp=timestamp, label=label, x=x, y=y,
        batch=batch, links=links_arr, bssid=bssid_arr, sc_idx=sc_idx,
        alignment_mode=np.array("ordinal_static"),
        subcarrier_selection_batches=np.array(selection_batches),
        **grid_fields,
    )

    raw64_out = sess_dir / "static_raw64.npz"
    np.savez(
        raw64_out, amp=amp_raw, phase=phase_raw, timestamp=timestamp,
        label=label, x=x, y=y, batch=batch, links=links_arr, bssid=bssid_arr,
        sc_idx=np.arange(RAW_SUB, dtype=np.int64), alignment_mode=np.array("ordinal_static"),
        **grid_fields,
    )

    report_path = sess_dir / "truncation_report.json"
    report_path.write_text(json.dumps(
        [{"round": r, "label": l, "counts": c, "n_used": n} for r, l, c, n in trunc_report],
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n已保存 {out}")
    print(f"  amp{amp.shape}  phase{phase.shape}  N={amp.shape[0]}  links={links_arr.tolist()}")
    print(f"  classes={sorted(np.unique(label).tolist())}  batches={sorted(set(batch.tolist()))}")
    print(f"  截断报告已存 {report_path}")
    print(f"  交叉验证源数据已存 {raw64_out}（保留全部 {RAW_SUB} 个原始子载波）")
    cls_missing = sorted(set(range(grid.num_classes)) - set(np.unique(label).tolist()))
    if cls_missing:
        print(f"  注意：缺少类别 {cls_missing}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="阶段二四链路 pcap → static.npz / static_raw64.npz")
    ap.add_argument("session", nargs="?", default="session_phase2_001",
                    help="session_phase2_001（旧 pilot）或 d1 / d3 / d5 / smoke")
    ap.add_argument("--sc-from", default=None,
                    help="沿用另一个 session 的子载波选择（跨日 session 用 d1）；self = 全部轮次，仅可读性检查")
    args = ap.parse_args()
    build(args.session, sc_from=args.sc_from)
