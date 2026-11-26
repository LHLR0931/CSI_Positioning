"""D+3 检验：冻结单链路 CNN 编码器 + 融合（F-MLP）是否比 ConcatMLP-mask 更抗漂移（2026-10-14，跑之前写下判据）

背景
----
D1 上（backend/training/frozen_fusion.py）发现：用阶段二独立训练的 4 个单链路 CNN 作编码器（冻结）、
再接 MLP 头的 F-MLP，r10 误差 0.397 m，好于端到端 ConcatMLP（0.826）和 ConcatMLP-mask（0.560）。
但 D1 只有 r9 / r10 两轮，而且之前看过。D+3 从没用这个方法测过，所以在这里做一次干净检验。
本文件的判据在运行前冻结（2026-10-14 17:20）。

完全沿用 crossday 的 D+3 设置，不在 D+3 上调任何东西
--------------------------------------------------
    训练集      D1 r1–r8 + D+3 标定轮 r2，预处理器 = crossday/d3/preprocessor.json（与 mask 模型同一份）
    编码器      crossday/d3/cnn_<链路>_seed<N>.pt（crossday train 时已训练好的单链路 CNN，冻结）
    融合方式    F-KNN（k=5）/ F-GCN / F-MLP，与 frozen_fusion.py 完全相同的类和参数
    训练轮数    F-GCN、F-MLP 取 D1 select 阶段 frozen_fusion.json 的 best_epoch + 1
                （与 crossday 从 D1 select 取 ConcatMLP-mask 轮数的做法相同）
    对照        ConcatMLP-mask 等权（crossday/d3/mask_seed<N>.pt）。脚本会重算它，与 crossday 的
                results.json 核对（3 种子平均误差相差 > 0.005 m 就停）
    指标        crossday.round_metrics：每轮先按格点平均，再 15 个格点等权
    种子        42 / 1 / 2，取平均

判据（跑之前写下，用户 2026-10-14 17:20 确认）
--------------------------------------------
Δ_r = F-MLP 平均误差 − ConcatMLP-mask 等权平均误差（各自 3 种子平均），r = r3 / r4 / r5（测试轮）
    明确支持        3 轮平均 Δ ≤ −0.05 m，且 3 轮的 Δ 都 < 0
    方向性支持      3 轮平均 Δ < 0，但不满足上一条
    不支持          3 轮平均 Δ ≥ 0
次要（只报告，不设判据）：F-KNN、F-GCN 与 mask 的差；Δ_A = F-MLP − min(F-KNN, F-GCN)（复现 D1 的问题 A）；
水瓶轮 r6 为探索性，不进判据。

用法（项目根目录；需要先跑过 frozen_fusion --phase select，以及 crossday train --day d3）：
    python -X utf8 -m backend.training.frozen_fusion_d3
输出：backend/artifacts/crossday/d3/frozen_fusion_results.json
"""
from __future__ import annotations

import json
import time

import numpy as np

from backend import config
from backend.data.preprocess import Preprocessor
from backend.training import crossday as C
from backend.training.frozen_fusion import as_window, encode, make_mlp

DAY = "d3"
SEEDS = (42, 1, 2)
LINKS = config.all_link_ids()
CLEAR_M = 0.05                 # 判据：3 轮平均 Δ ≤ −0.05 m
CHECK_TOL = 0.005              # mask 重算与 crossday results.json 的容差，m
METHODS = ("mask 等权", "F-KNN", "F-GCN", "F-MLP")
OUT = C.OUT_ROOT / DAY / "frozen_fusion_results.json"


def select_epochs(seed: int) -> dict:
    f = config.ARTIFACTS_DIR / "d1" / "select" / f"cls_ampphase_pp2_seed{seed}" / "frozen_fusion.json"
    if not f.exists():
        raise SystemExit(f"找不到 {f}：先跑 python -X utf8 -m backend.training.frozen_fusion --phase select")
    res = json.loads(f.read_text(encoding="utf-8"))["models"]
    return {k: v["best_epoch"] + 1 for k, v in res.items() if "best_epoch" in v}


def lab_metrics(pred: np.ndarray, lab: np.ndarray, grid) -> dict:
    err = np.linalg.norm(C.label_xy(pred, grid.cols) - C.label_xy(lab, grid.cols), axis=1) * grid.spacing_m
    return C.round_metrics(err, pred == lab, lab)


def main() -> None:
    proto, grid = C.day_protocol(DAY), C._grid(DAY)
    od = C.OUT_ROOT / DAY
    meta = json.loads((od / "train_meta.json").read_text(encoding="utf-8"))
    pp = Preprocessor.load(od / "preprocessor.json")
    base = C.session_windows(C.BASE_SESSION, proto["base_rounds"])
    cal = C.session_windows(DAY, [proto["calib"]])
    Xtr = np.concatenate([C.to_model_input(pp, base), C.to_model_input(pp, cal)])
    ytr = np.concatenate([base["label"], cal["label"]])
    if len(ytr) != meta["n_train"]:
        raise SystemExit(f"训练窗口数 {len(ytr)} 与 crossday 训练时的 {meta['n_train']} 不同——训练集对不上，停止。")
    rounds = proto["test"] + proto["perturb"]
    part = C.session_windows(DAY, rounds)
    X, lab, bat = C.to_model_input(pp, part), part["label"], part["batch"]
    print(f"==== D+3 冻结编码器检验：训练 D1 {proto['base_rounds']} + {DAY} {proto['calib']}（{len(ytr)} 窗口，"
          f"与 crossday 一致 ✓），测试 {proto['test']}，探索 {proto['perturb']}")

    per = {r: {m: [] for m in METHODS} for r in rounds}       # per[r][方法] = 每个种子的指标
    for seed in SEEDS:
        t0 = time.time()
        ep = select_epochs(seed)
        preds = {}
        preds["mask 等权"] = C.logits(C.make_model("mask", seed, grid).load(od / f"mask_seed{seed}"), X).argmax(1)
        encs = [C.make_model("cnn", seed, grid).load(od / f"cnn_{lid}_seed{seed}") for lid in LINKS]
        Ftr, Fte = encode(encs, Xtr), encode(encs, X)

        from backend.models.knn import KNNModel
        preds["F-KNN"] = KNNModel(k=5, standardize=True).fit(Ftr, ytr).predict(Fte)

        from backend.models.sample_gcn import SampleGCNModel
        g = SampleGCNModel(n_classes=grid.num_classes, grid_cols=grid.cols, seed=seed, k=5)
        g.fit(as_window(Ftr), ytr, epochs=ep["F-GCN"])
        preds["F-GCN"] = g.logits(as_window(Fte)).argmax(1)

        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        mlp = make_mlp(seed, Ftr.shape[1], grid).fit((Ftr - mu) / sd, ytr, epochs=ep["F-MLP"])
        preds["F-MLP"] = mlp._forward_np((Fte - mu) / sd).argmax(1)

        for r in rounds:
            mr = bat == r
            for m in METHODS:
                per[r][m].append(lab_metrics(np.asarray(preds[m])[mr], lab[mr], grid))
        print(f"  种子 {seed} 完成：F-GCN {ep['F-GCN']} 轮，F-MLP {ep['F-MLP']} 轮  ({time.time() - t0:.0f}s)")

    # mask 重算与 crossday results.json 核对
    ref = json.loads((od / "results.json").read_text(encoding="utf-8"))["rounds"]
    for r in rounds:
        got = float(np.mean([x["mae"] for x in per[r]["mask 等权"]]))
        want = ref[r]["methods"]["等权"]["mae"]
        if abs(got - want) > CHECK_TOL:
            raise SystemExit(f"!! {r} mask 等权重算 {got:.3f} m，crossday results.json 为 {want:.3f} m —— 先停下来查原因")
    print("  ConcatMLP-mask 等权重算与 crossday results.json 一致 ✓")

    avg = lambda r, m, k="mae": float(np.mean([x[k] for x in per[r][m]]))
    t = C.round_times(DAY, rounds, proto["calib"])
    res = {"day": DAY, "seeds": list(SEEDS), "minutes_after_calib": t, "rounds": {}}
    print(f"\n===== D+3  3 个种子平均；单位 m；Δ = 方法 − mask 等权（负 = 比 mask 好）")
    for r in rounds:
        role = "test" if r in proto["test"] else "perturb（探索性）"
        print(f"\n[{r}] {role}" + (f"  中点 ≈ 标定后 {t[r]} 分钟" if r in t else ""))
        print(f"  {'方法':10s} {'MAE':>7s} {'(逐种子)':>24s} {'P90':>6s} {'准确率':>6s} {'>1.6m':>6s} {'Δ vs mask':>10s}")
        rr = {"role": role, "methods": {}}
        for m in METHODS:
            seeds_mae = [x["mae"] for x in per[r][m]]
            d = avg(r, m) - avg(r, "mask 等权")
            rr["methods"][m] = {"mae": avg(r, m), "mae_per_seed": seeds_mae, "p90": avg(r, m, "p90"),
                                "acc": avg(r, m, "acc"), "far": avg(r, m, "far"), "delta_vs_mask": d}
            print(f"  {m:10s} {avg(r, m):7.3f} {' / '.join(f'{v:.3f}' for v in seeds_mae):>24s} "
                  f"{avg(r, m, 'p90'):6.2f} {avg(r, m, 'acc'):6.0%} {avg(r, m, 'far'):6.0%} "
                  f"{'' if m == 'mask 等权' else f'{d:+10.3f}'}")
        rr["delta_A"] = avg(r, "F-MLP") - min(avg(r, "F-KNN"), avg(r, "F-GCN"))
        print(f"  Δ_A = F-MLP − min(F-KNN, F-GCN) = {rr['delta_A']:+.3f} m（只报告）")
        res["rounds"][r] = rr

    deltas = [res["rounds"][r]["methods"]["F-MLP"]["delta_vs_mask"] for r in proto["test"]]
    mean_d = float(np.mean(deltas))
    c1, c2 = mean_d <= -CLEAR_M, all(d < 0 for d in deltas)
    verdict = "明确支持" if (c1 and c2) else ("方向性支持" if mean_d < 0 else "不支持")
    print("\n===== 判据（跑之前写下，用户 2026-10-14 17:20 确认）")
    print(f"  Δ（F-MLP − mask 等权）：" + "  ".join(f"{r} {d:+.3f}" for r, d in zip(proto["test"], deltas))
          + f"   3 轮平均 {mean_d:+.3f} m")
    print(f"    {'✓' if c1 else '✗'} 3 轮平均 Δ ≤ −{CLEAR_M} m")
    print(f"    {'✓' if c2 else '✗'} 3 轮 Δ 都 < 0")
    print(f"  结论：\"冻结单链路编码器 + MLP 比 ConcatMLP-mask 更抗漂移\" —— {verdict}")
    res.update(test_deltas=dict(zip(proto["test"], deltas)), mean_delta=mean_d,
               conditions={"mean<=-0.05": c1, "all<0": c2}, verdict=verdict)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
