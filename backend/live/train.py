"""当天标定轮：解析 + 训练 F-MLP（实时后端用）。

标定轮用现有的 phase2_capture.ps1 采，例如：
    .\\phase2_capture.ps1 -Session d14 -Round 1 -RoomNote "..."
采完后（后端 server 会自动调用本脚本；也可以手动运行）：
    python -X utf8 -m backend.live.train --session d14 --round 1

步骤：
    1. 解析：backend.data.load_real_stage2.build(session, sc_from="d1")，与 D+3 相同
       （会解析该 session 下已有的全部轮次，但训练只用 --round 指定的标定轮）
    2. 训练：backend.live.fmlp.train（协议见该文件开头）
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

from backend.data import sessions
from backend.data.sessions import session_dir


def _utc(s: str) -> datetime:
    s = s.strip().rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{''.join(ch for ch in frac if ch.isdigit())[:6]}"
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def calib_progress(session: str, rnd: int, n_points: int = 15) -> dict:
    """读 phase2_capture.ps1 写下的记录，判断标定轮进行到哪里。

    done      已采好的格点：两份 pcap 都在，且该格点最后一次记录不是 fail
    finished  session_meta.json 里这一轮的整轮记录（不是 -Only 单点重采）已有 finished_utc
    complete  finished，且 done + 被 skip 的格点 = 全部 15 个
    """
    rd = session_dir(session) / f"phase2_r{rnd}"
    last: dict[int, dict] = {}
    f = rd / "points_meta.csv"
    if f.exists():
        with open(f, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    last[int(row["label"])] = row
                except (KeyError, ValueError):
                    pass
    done = sorted(l for l, row in last.items()
                  if row.get("status") != "fail"
                  and (rd / f"label{l}_pi1.pcap").exists() and (rd / f"label{l}_pi2.pcap").exists())
    meta = sessions.read_session_meta(session) or {}
    full = [r for r in meta.get("rounds", []) if int(r.get("round", -1)) == rnd and r.get("only_label") in (None, -1)]
    finished = bool(full) and bool(full[-1].get("finished_utc"))
    skipped = set(full[-1].get("skipped_labels") or []) if full else set()
    ends = [_utc(row["end_utc"]) for l, row in last.items() if l in done and row.get("end_utc")]
    return dict(done=done, n_done=len(done), skipped=sorted(skipped), finished=finished,
                complete=finished and len(set(done) | skipped) >= n_points,
                calibrated_at=(max(ends).astimezone().isoformat(timespec="seconds") if ends else None))


def main() -> None:
    ap = argparse.ArgumentParser(description="当天标定轮：解析 + 训练 F-MLP")
    ap.add_argument("--session", required=True, help="例如 d14")
    ap.add_argument("--round", type=int, default=1, help="标定轮编号（默认 1）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-parse", action="store_true", help="已经解析过时跳过第 1 步")
    args = ap.parse_args()

    prog = calib_progress(args.session, args.round)
    print(f"==== {args.session} 第 {args.round} 轮（标定）：已采 {prog['n_done']} 个格点，"
          f"整轮{'已结束' if prog['finished'] else '未结束'}，被跳过 {prog['skipped']}")
    if not prog["complete"]:
        print("  ⚠ 标定轮还不完整，照样训练，但缺的格点模型永远预测不出来。")
    if not args.skip_parse:
        from backend.data.load_real_stage2 import build
        print("\n---- 1/2 解析 pcap（子载波沿用 D1）")
        build(args.session, sc_from="d1")
    print("\n---- 2/2 训练 F-MLP")
    from backend.live.fmlp import train
    od = train(args.session, f"r{args.round}", seed=args.seed)
    # 记下标定完成时间，server 在 status.calibrated_at 里报给前端
    import json
    meta_f = od / "train_meta.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8"))
    meta["calibrated_at"] = prog["calibrated_at"]
    meta_f.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TRAIN_OK")


if __name__ == "__main__":
    main()
