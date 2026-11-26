"""导出 Developer Replay 用的逐窗口预测：D1 留出测试轮 r9 / r10 × 四个模型（2026-10-25）。

这是数据导出，不是新实验
------------------------
不训练、不调参、不改模型：只加载 `train_d1.py --phase final` 已经保存的权重
（backend/artifacts/d1/final/cls_ampphase_pp2_seed<N>/），在 r9 / r10 上推理一次，
把每个窗口的预测写成 JSON + CSV，供 Unity 客户端的 Developer Replay 回放。

为什么只能用 D1 r9 / r10
------------------------
手机端的四个按钮是 KNN / GCN / ConcatMLP / ConcatMLP-mask。四个模型都训练过、且有
留出测试轮的只有 D1 最终测试阶段：训练 r1–r8（r8 = piB 13:06 重启后的标定轮），
r9 / r10 不参与训练、早停或标定（训练轮数固定取自选模型阶段，那时 r8–r10 完全不可见）。
D+3 只训练了 ConcatMLP-mask 与 F-MLP，没有 KNN / GCN / ConcatMLP。

窗口规则（与训练评估完全相同，见 prepare.load_stage2_windows）
-------------------------------------------------------------
同一轮、同一格点的连续 16 帧为一个窗口，不重叠，尾部不足 16 帧丢弃；
四链路对齐后约 10 帧/秒 → 每窗口约 1.6 秒；每格点约 41 秒 → 约 25 个窗口。

自检（只打印"一致 / 不一致"）
----------------------------
用导出的预测重算每轮平均误差与准确率，必须与 results.json 里已保存的数一致，
证明导出的就是论文用的那次推理。这些数此前已报告过（论文中标为诊断证据），不是新的测试曝光。

用法（项目根目录，csi 环境）：
    python -X utf8 -m backend.training.export_replay                 # 种子 42
    python -X utf8 -m backend.training.export_replay --seed 42 1 2   # 三个种子各出一份
    python -X utf8 -m backend.training.export_replay --check-data    # 只检查窗口与时间戳，不加载模型
输出：backend/artifacts/replay/replay_d1_r9r10_seed<N>.json / .csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from backend import config
from backend.data import sessions
from backend.data.load import load_static_raw64
from backend.training import train_d1 as TD

SESSION = "d1"
PHASE = "final"
TEST_ROUNDS = ("r9", "r10")
CALIB_ROUND = "r8"
TRAIN_ROUNDS = ("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8")
MODELS = ("KNN", "GCN", "ConcatMLP", "ConcatMLP-mask")
SYD = timezone(timedelta(hours=11))   # 2026-10-08 悉尼已是夏令时 AEDT（10-04 起），固定 +11:00
OUT_DIR = config.ARTIFACTS_DIR / "replay"
MODEL_DESC = {
    "KNN": "4-link KNN (k=5, distance-weighted) on per-window time-averaged amplitude+phase (440-d, train-set standardised). "
           "Probabilities = distance-weighted vote of the 5 neighbours, so they are coarse.",
    "GCN": "Sample-graph GCN (structure ported from prior RSSI+GNN work), same features and k as 4-link KNN. "
           "Probabilities = softmax of logits.",
    "ConcatMLP": "Shared 2D-CNN encoder per link, 4 link features concatenated, MLP head, trained end-to-end. "
                 "Probabilities = softmax of logits.",
    "ConcatMLP-mask": "ConcatMLP + random link masking during training (p=0.25). Probabilities = softmax of logits.",
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iso(t: float) -> str:
    return datetime.fromtimestamp(float(t), SYD).isoformat(timespec="milliseconds")


def window_index(T: int):
    """按 load_stage2_windows 完全相同的顺序枚举窗口，返回测试轮窗口的元数据（顺序与 data['test'] 一致）。"""
    d = load_static_raw64(SESSION)
    labels, batches, ts = d["label"], d["batch"], d["timestamp"]     # ts: [帧, 4 链路] Unix 秒
    rows = []
    for lbl in np.unique(labels):
        for b in np.unique(batches):
            idx = np.where((labels == lbl) & (batches == b))[0]
            for w in range(len(idx) // T):
                sel = idx[w * T:(w + 1) * T]
                rows.append((int(lbl), str(b), w, float(np.nanmean(ts[sel[0]])), float(np.nanmean(ts[sel[-1]]))))
    test = [r for r in rows if r[1] in TEST_ROUNDS]       # make_split: np.sort(isin) → 保持枚举顺序
    calib_end = float(np.nanmean(ts[batches == CALIB_ROUND], axis=1).max())
    return test, calib_end


def load_data():
    data, _ = TD.load(PHASE, "ampphase")
    Xte, yte, lte, bte = data["test"]
    return Xte, yte, lte, bte


def check_data() -> None:
    T = config.STAGE2_TIME_WINDOW
    Xte, _, lte, bte = load_data()
    meta, calib_end = window_index(T)
    assert len(meta) == len(lte), f"窗口数不一致：枚举 {len(meta)}，测试集 {len(lte)}"
    assert all(m[0] == int(l) and m[1] == str(b) for m, l, b in zip(meta, lte, bte)), "窗口顺序与测试集不一致"
    print(f"X 形状 {Xte.shape}；窗口与测试集逐条对齐：通过")
    for r in TEST_ROUNDS:
        m = [x for x in meta if x[1] == r]
        per = np.bincount([x[0] for x in m], minlength=15)
        print(f"{r}: {len(m)} 个窗口，每点窗口数 {per.min()}–{per.max()}，"
              f"时间 {iso(min(x[3] for x in m))[11:19]} – {iso(max(x[4] for x in m))[11:19]}，"
              f"开始于标定结束后 {(min(x[3] for x in m) - calib_end) / 60:.1f} 分钟")


def predict(name: str, Xte, seed: int, grid, mdir: Path):
    """返回 (pred_label[N], proba[N,15], 权重文件路径)。"""
    from backend.models.sample_gcn import window_features
    if name == "KNN":
        from backend.models.knn import KNNModel
        f = mdir / "KNN.joblib"
        m = KNNModel.load(f)
        F = window_features(Xte)
        lab = np.asarray(m.predict(F)).astype(int)        # 与 train_d1 评估时同一调用
        return lab, m.predict_proba(F), f
    if name == "GCN":
        from backend.models.sample_gcn import SampleGCNModel
        f = mdir / "GCN.pt"
        m = SampleGCNModel(n_classes=grid.num_classes, grid_cols=grid.cols, seed=seed, k=TD.KNN_K).load(mdir / "GCN")
        z = m.logits(Xte)
        z = np.exp(z - z.max(1, keepdims=True))
        p = z / z.sum(1, keepdims=True)
        return p.argmax(1), p, f
    f = mdir / f"{name}.pt"
    m, _ = TD.make_model(name, seed, grid, "cls", Xte.shape[2])
    m.load(mdir / name)
    p = m.predict_proba(Xte)
    return p.argmax(1), p, f


def export(seed: int) -> None:
    T = config.STAGE2_TIME_WINDOW
    grid = sessions.session_grid(SESSION)
    tag = TD.tag_of("cls", "ampphase", seed)
    mdir = config.ARTIFACTS_DIR / "d1" / PHASE / tag
    saved = json.loads((mdir / "results.json").read_text(encoding="utf-8"))["models"]
    Xte, yte, lte, bte = load_data()
    meta, calib_end = window_index(T)
    assert len(meta) == len(lte) and all(m[0] == int(l) and m[1] == str(b) for m, l, b in zip(meta, lte, bte)), \
        "窗口顺序与测试集不一致"

    # 每轮按开始时间排序，给出回放用的窗口编号
    order = {r: sorted([i for i, m in enumerate(meta) if m[1] == r], key=lambda i: meta[i][3]) for r in TEST_ROUNDS}
    win_no = {i: k for r in TEST_ROUNDS for k, i in enumerate(order[r])}

    records, model_meta = [], {}
    for name in MODELS:
        lab, proba, wfile = predict(name, Xte, seed, grid, mdir)
        pred_xy = np.stack([lab % grid.cols, lab // grid.cols], 1).astype(np.float32)
        # 自检：与 results.json 一致
        status = []
        for r in TEST_ROUNDS:
            s = bte == r
            got = TD.metrics_for(pred_xy[s], yte[s], lte[s], grid)
            ref = saved[name]["by_round"][r]
            de = abs(got["mean_error_m"] - ref["mean_error_m"])
            da = abs(got["accuracy"] - ref["accuracy"])
            status.append(f"{r} {'一致' if de < 1e-3 and da < 1e-3 else f'不一致(误差差 {de:.4f} m, 准确率差 {da:.4f})'}")
        print(f"{name:16s} 与 results.json 对照：{'；'.join(status)}")
        model_meta[name] = {"weights_file": str(wfile.relative_to(config.ARTIFACTS_DIR.parent.parent)).replace("\\", "/"),
                            "weights_md5": md5(wfile),
                            "epochs_fixed": saved[name].get("epochs_fixed"),
                            "description": MODEL_DESC[name]}
        for r in TEST_ROUNDS:
            for i in order[r]:
                lbl, rnd, w, t0, t1 = meta[i]
                pl = int(lab[i])
                err = float(np.hypot(pred_xy[i, 0] - yte[i, 0], pred_xy[i, 1] - yte[i, 1]) * grid.spacing_m)
                records.append({
                    "session": SESSION, "capture_date": "2026-10-08", "round": rnd,
                    "window_index": win_no[i], "window_id": f"{SESSION}-{rnd}-w{win_no[i]:03d}",
                    "point_window_index": w,
                    "model": name, "model_version": f"d1-final-{tag}",
                    "true_label": lbl, "pred_label": pl, "correct": pl == lbl,
                    "error_m": round(err, 4),
                    "true_x_m": round(float(yte[i, 0]) * grid.spacing_m, 3),
                    "true_y_m": round(float(yte[i, 1]) * grid.spacing_m, 3),
                    "pred_x_m": round(float(pred_xy[i, 0]) * grid.spacing_m, 3),
                    "pred_y_m": round(float(pred_xy[i, 1]) * grid.spacing_m, 3),
                    "confidence": round(float(proba[i].max()), 4),
                    "probs": [round(float(v), 4) for v in proba[i]],
                    "t_start": iso(t0), "t_end": iso(t1),
                    "t_start_unix": round(t0, 3), "t_end_unix": round(t1, 3),
                    "minutes_since_calibration": round((t0 - calib_end) / 60, 2),
                })

    out = {
        "meta": {
            "exported_at": datetime.now(SYD).isoformat(timespec="seconds"),
            "source_script": "backend/training/export_replay.py",
            "session": SESSION, "capture_date": "2026-10-08", "site": "home, 5x3 grid, 0.8 m spacing",
            "grid": {"cols": grid.cols, "rows": grid.rows, "spacing_m": grid.spacing_m,
                     "label_rule": "label = y*5 + x, coordinates (0.8x, 0.8y) m"},
            "links": list(config.all_link_ids()),
            "train_rounds": list(TRAIN_ROUNDS), "calibration_round": CALIB_ROUND,
            "test_rounds": list(TEST_ROUNDS),
            "test_rounds_used_for": "none of training, early stopping or calibration (held-out test)",
            "calibration_end": iso(calib_end),
            "window_rule": "16 consecutive aligned frames (~1.6 s at ~10 fps), non-overlapping, per round and point",
            "seed": seed, "preprocessing": "pp2 (55 valid subcarriers sorted by frequency), amplitude+phase",
            "models": model_meta,
        },
        "records": records,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / f"replay_d1_r9r10_seed{seed}.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    cp = OUT_DIR / f"replay_d1_r9r10_seed{seed}.csv"
    cols = [k for k in records[0] if k != "probs"] + [f"prob_{k}" for k in range(grid.num_classes)]
    with open(cp, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for rec in records:
            wr.writerow([rec[k] for k in cols if not k.startswith("prob_")] + rec["probs"])
    print(f"\n已写出 {len(records)} 条记录（{len(records) // len(MODELS)} 个窗口 × {len(MODELS)} 个模型）：\n  {jp}\n  {cp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="导出 D1 r9/r10 四个模型的逐窗口预测（Developer Replay 用）")
    ap.add_argument("--seed", type=int, nargs="+", default=[42])
    ap.add_argument("--check-data", action="store_true", help="只检查窗口与时间戳，不加载模型")
    args = ap.parse_args()
    if args.check_data:
        check_data()
    else:
        for s in args.seed:
            export(s)
