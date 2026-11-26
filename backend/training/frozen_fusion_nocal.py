"""跨天 + 不标定：解耦训练的单链路编码器（F-MLP）能否替代当天标定？（2026-10-14，跑之前写下判据）

问题
----
D+3 上已经证明：①当天标定是必要的（ConcatMLP-mask 不标定 1.74–1.84 m，比地板还差）；②标定之后，
冻结的单链路编码器 + MLP 头（F-MLP）比 ConcatMLP-mask 更抗渐进漂移（平均好 0.155 m）。
但 F-MLP 从没在"不标定"的情况下测过。这里回答：只用 D1 训练、直接测 3 天后的 D+3（跨天 + 跨重启 +
不标定），F-MLP 会怎样？

设置（全部沿用已有文件，不在 D+3 上调任何东西）
----------------------------------------------
    训练集      只用 D1 r1–r8（3013 窗口），不含 D+3 任何数据
    预处理      crossday/d3/preprocessor_nocal.json（只用 D1 拟合；与 D1 final 的预处理器逐项相同，
                已核对：最大差 2.4e-7）
    编码器      D1 final 的单链路 CNN（cls_ampphase_pp2_seed<N>/CNN-<链路>.pt，只用 D1 r1–r8 训练），冻结
    融合        F-MLP / F-KNN / F-GCN，与 frozen_fusion.py 同类同参；F-MLP、F-GCN 的轮数取 D1 select
                阶段 frozen_fusion.json 的 best_epoch + 1（与 frozen_fusion_d3.py 相同）
    对照        ConcatMLP-mask 不标定（crossday/d3/mask_nocal_seed<N>.pt，即预登记里的 S4）
    测试        D+3 r3 / r4 / r5（与主检验同样的三轮）；r2、r6 只报告
    指标        crossday.round_metrics（每轮按 15 个格点等权）；地板 = 永远猜正中间
自检：①D1 单链路 CNN 经本脚本的输入流程在 D1 r9/r10 上的成绩须与 results.json 一致（确认输入流程相同）；
     ②mask 不标定在 D+3 上的重算须与 crossday results.json 的 S4 一致。

跑之前写下的判据（D+3 已解盲过 mask 与 F-MLP 的"标定后"结果、以及 mask 不标定的结果；
F-MLP 不标定从未算过。只看 r3 / r4 / r5，3 种子平均）
------------------------------------------------------------------------------
    Δ_r = F-MLP 不标定 − mask 不标定
    A 优于 mask 不标定：3 轮平均 Δ ≤ −0.10 m，且 3 轮的 Δ 都 < 0
    B 摆脱地板：3 轮 F-MLP 不标定的误差都 < 地板 − 0.10 m（约 1.105 m）
    结论：
      A 且 B   → "解耦编码器在跨天不标定时仍明显有效（但仍需与标定后结果比较，说明标定还有多少价值）"
      A 非 B   → "比 mask 好，但仍在地板附近：解耦编码器不能替代当天标定"
      非 A     → "不能替代当天标定；解耦编码器的作用范围是标定后的渐进漂移"
（阈值取 0.10 m 而不是 0.05 m：不标定时误差在 1.7–1.8 m，种子间波动更大。）

用法（项目根目录）：
    python -X utf8 -m backend.training.frozen_fusion_nocal
输出：backend/artifacts/crossday/d3/frozen_fusion_nocal_results.json
"""
from __future__ import annotations

import json
import time

import numpy as np

from backend import config
from backend.data.preprocess import Preprocessor
from backend.training import crossday as C
from backend.training import train_d1 as T
from backend.training.frozen_fusion import as_window, encode, make_mlp
from backend.training.frozen_fusion_d3 import lab_metrics, select_epochs

DAY = "d3"
SEEDS = (42, 1, 2)
LINKS = config.all_link_ids()
THR = 0.10
CHECK_TOL = 0.005
METHODS = ("地板（猜中间）", "mask 不标定", "F-KNN 不标定", "F-GCN 不标定", "F-MLP 不标定")
OUT = C.OUT_ROOT / DAY / "frozen_fusion_nocal_results.json"


def main() -> None:
    proto, grid = C.day_protocol(DAY), C._grid(DAY)
    od = C.OUT_ROOT / DAY
    pp = Preprocessor.load(od / "preprocessor_nocal.json")
    base = C.session_windows(C.BASE_SESSION, proto["base_rounds"])       # D1 r1–r8
    Xtr, ytr = C.to_model_input(pp, base), base["label"]
    rounds = proto["test"] + [proto["calib"]] + proto["perturb"]          # r3 r4 r5 + r2 r6（只报告）
    part = C.session_windows(DAY, rounds)
    X, lab, bat = C.to_model_input(pp, part), part["label"], part["batch"]
    d1test = C.session_windows(C.BASE_SESSION, ["r9", "r10"])             # 自检 ①
    Xd1, labd1, batd1 = C.to_model_input(pp, d1test), d1test["label"], d1test["batch"]
    print(f"==== 跨天不标定：训练只用 D1 {proto['base_rounds']}（{len(ytr)} 窗口），测 D+3 {rounds}")

    center = np.full(len(lab), (grid.rows // 2) * grid.cols + grid.cols // 2)
    per = {r: {m: [] for m in METHODS} for r in rounds}
    for seed in SEEDS:
        t0 = time.time()
        d = config.ARTIFACTS_DIR / "d1" / "final" / T.tag_of("cls", "ampphase", seed)
        ref = json.loads((d / "results.json").read_text(encoding="utf-8"))["models"]
        encs = []
        for l, lid in enumerate(LINKS):
            m = C.make_model("cnn", seed, grid).load(d / f"CNN-{lid}")
            z = C.logits(m, Xd1[:, l:l + 1])                                # 自检 ①
            for r in ("r9", "r10"):
                mr = batd1 == r
                e, c = C.errors_from_logits(z[mr], labd1[mr], grid)
                got = float(e.mean())                                         # train_d1 按窗口平均
                want = ref[f"CNN-{lid}"]["by_round"][r]["mean_error_m"]
                if abs(got - want) > CHECK_TOL:
                    raise SystemExit(f"!! 种子 {seed} CNN-{lid} D1 {r}：本流程 {got:.3f}，results.json {want:.3f}"
                                     f" —— 输入流程与 train_d1 不一致，停止。")
            encs.append(m)
        ep = select_epochs(seed)
        Ftr, Fte = encode(encs, Xtr), encode(encs, X)
        preds = {"地板（猜中间）": center,
                 "mask 不标定": C.logits(C.make_model("mask", seed, grid).load(od / f"mask_nocal_seed{seed}"), X).argmax(1)}
        from backend.models.knn import KNNModel
        preds["F-KNN 不标定"] = KNNModel(k=5, standardize=True).fit(Ftr, ytr).predict(Fte)
        from backend.models.sample_gcn import SampleGCNModel
        g = SampleGCNModel(n_classes=grid.num_classes, grid_cols=grid.cols, seed=seed, k=5)
        g.fit(as_window(Ftr), ytr, epochs=ep["F-GCN"])
        preds["F-GCN 不标定"] = g.logits(as_window(Fte)).argmax(1)
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        mlp = make_mlp(seed, Ftr.shape[1], grid).fit((Ftr - mu) / sd, ytr, epochs=ep["F-MLP"])
        preds["F-MLP 不标定"] = mlp._forward_np((Fte - mu) / sd).argmax(1)
        for r in rounds:
            mr = bat == r
            for m in METHODS:
                per[r][m].append(lab_metrics(np.asarray(preds[m])[mr], lab[mr], grid))
        print(f"  种子 {seed} 完成（D1 单链路 CNN 经本流程复算与 results.json 一致 ✓）  ({time.time() - t0:.0f}s)")

    ref3 = json.loads((od / "results.json").read_text(encoding="utf-8"))["rounds"]   # 自检 ②
    for r in proto["test"] + proto["perturb"]:
        got = float(np.mean([x["mae"] for x in per[r]["mask 不标定"]]))
        want = ref3[r]["methods"]["S4 不标定·等权"]["mae"]
        if abs(got - want) > CHECK_TOL:
            raise SystemExit(f"!! {r} mask 不标定重算 {got:.3f}，crossday S4 为 {want:.3f}，停止。")
    print("  mask 不标定重算与 crossday results.json（S4）一致 ✓")

    cal = json.loads((od / "frozen_fusion_results.json").read_text(encoding="utf-8"))["rounds"] \
        if (od / "frozen_fusion_results.json").exists() else {}
    avg = lambda r, m, k="mae": float(np.mean([x[k] for x in per[r][m]]))
    t = C.round_times(DAY, rounds, proto["calib"])
    res = {"day": DAY, "seeds": list(SEEDS), "train": "D1 r1-r8 only", "rounds": {}}
    print("\n===== 跨天不标定（D1 训练 → D+3 测试），3 个种子平均；单位 m；Δ = 方法 − mask 不标定")
    for r in rounds:
        role = "test" if r in proto["test"] else ("calib 轮（这里只当测试，只报告）" if r == proto["calib"] else "水瓶（只报告）")
        print(f"\n[{r}] {role}" + (f"  中点 ≈ 标定后 {t[r]} 分钟" if r in t else ""))
        print(f"  {'方法':14s} {'MAE':>7s} {'(逐种子)':>24s} {'P90':>6s} {'准确率':>6s} {'>1.6m':>6s} {'Δ vs mask':>10s}")
        rr = {"role": role, "methods": {}}
        for m in METHODS:
            s = [x["mae"] for x in per[r][m]]
            dd = avg(r, m) - avg(r, "mask 不标定")
            rr["methods"][m] = {"mae": avg(r, m), "mae_per_seed": s, "p90": avg(r, m, "p90"),
                                "acc": avg(r, m, "acc"), "far": avg(r, m, "far"), "delta_vs_mask": dd}
            show = m not in ("地板（猜中间）", "mask 不标定")
            print(f"  {m:14s} {avg(r, m):7.3f} {' / '.join(f'{v:.3f}' for v in s):>24s} {avg(r, m, 'p90'):6.2f} "
                  f"{avg(r, m, 'acc'):6.0%} {avg(r, m, 'far'):6.0%} {f'{dd:+10.3f}' if show else ''}")
        if r in cal:
            c = cal[r]["methods"]
            print(f"  （参考：标定后 F-MLP {c['F-MLP']['mae']:.3f}，标定后 mask {c['mask 等权']['mae']:.3f}）")
        res["rounds"][r] = rr

    floor = {r: avg(r, "地板（猜中间）") for r in proto["test"]}
    deltas = [res["rounds"][r]["methods"]["F-MLP 不标定"]["delta_vs_mask"] for r in proto["test"]]
    fm = [avg(r, "F-MLP 不标定") for r in proto["test"]]
    A = float(np.mean(deltas)) <= -THR and all(x < 0 for x in deltas)
    B = all(f < floor[r] - THR for f, r in zip(fm, proto["test"]))
    verdict = ("解耦编码器在跨天不标定时仍明显有效（但仍需与标定后结果比较）" if A and B else
               "比 mask 好，但仍在地板附近：解耦编码器不能替代当天标定" if A else
               "不能替代当天标定；解耦编码器的作用范围是标定后的渐进漂移")
    print("\n===== 判据（跑之前写下，见文件开头；r3 / r4 / r5）")
    print(f"  Δ（F-MLP 不标定 − mask 不标定）：" + "  ".join(f"{r} {x:+.3f}" for r, x in zip(proto["test"], deltas))
          + f"   平均 {np.mean(deltas):+.3f}")
    print(f"    {'✓' if A else '✗'} A：3 轮平均 Δ ≤ −{THR} 且 3 轮都 < 0")
    print(f"    {'✓' if B else '✗'} B：3 轮 F-MLP 不标定都 < 地板 − {THR}（"
          + "  ".join(f"{r} {f:.3f} vs {floor[r] - THR:.3f}" for r, f in zip(proto["test"], fm)) + "）")
    print(f"  结论：{verdict}")
    res.update(deltas=dict(zip(proto["test"], deltas)), floor=floor, conditions={"A": A, "B": B}, verdict=verdict)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
