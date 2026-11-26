"""E5 在 D+3 上的预登记复现：漂移后，联合训练表征的近邻结构是否比单独训练的差（2026-10-14，跑之前写下判据）

背景
----
D1 上的 E5（backend/training/e5_feature_vs_head.py，实验记录 B10.6）发现：不训练任何参数的 KNN（k=5、标准化）
在 256 维拼接特征上，
    r9 （标定后约 10 分钟）  联合训练表征 joint 0.290 m，单独训练表征 single 0.270 m —— 相近
    r10（标定后约 47 分钟）  joint 1.019 m，single 0.390 m —— 联合训练表征的近邻结构崩溃
D1 的 r9 / r10 之前看过，所以那是诊断分析。本脚本在 D+3 上按跑前写下的判据检验这一发现能否复现。

证据等级（必须如实写进论文）
--------------------------
D+3 的 r3 / r4 / r5 已用于 4.4 检验（frozen_fusion_d3.py）：F-KNN（= 这里的 KNN(single)）、F-MLP、
ConcatMLP-mask 的端到端结果都已看过。从没算过的只有"联合训练表征上的 KNN"。
所以本检验是"同一批测试轮上的预登记复现（未看结果）"，不是独立数据上的验证。

设置（完全沿用 crossday 的 D+3 设置，不在 D+3 上调任何东西）
---------------------------------------------------------
    训练集      D1 r1–r8 + D+3 标定轮 r2（3391 窗口），预处理器 = crossday/d3/preprocessor.json
    编码器
      joint     ConcatMLP（共享编码器、联合训练、无遮挡）。D+3 上原来没有，本脚本补训：与 crossday 的
                mask 模型同一份训练集、同一套超参数（crossday.make_model 的参数），只把模型换成
                ConcatMLPModel；轮数 = D1 select（pp2）results.json 中 ConcatMLP 的 best_epoch + 1
                （种子 42 / 1 / 2 = 9 / 8 / 6）。存为 crossday/d3/joint_seed<N>.pt
      mask      crossday/d3/mask_seed<N>.pt（现成，只报告）
      single    crossday/d3/cnn_<链路>_seed<N>.pt（现成，即 F-MLP / F-KNN 用的编码器）
    特征        每条链路过编码器得 64 维，四条拼成 256 维（frozen_fusion.encode）
    KNN         KNNModel(k=5, standardize=True)，与 E5、F-KNN 相同；不训练任何参数
    指标        crossday.round_metrics（每轮按 15 个格点等权）；3 个种子平均
    测试轮      r3 / r4 / r5（标定后约 8 / 24 / 40 分钟）；r6 水瓶轮只报告

判据（跑之前写下，用户 2026-10-14 23:06 确认）
--------------------------------------------
Δ_r = KNN(joint) − KNN(single)（各自 3 种子平均，单位 m；正 = 联合训练表征更差）
    复现成功    3 轮平均 Δ ≥ +0.10 m，且 r5（≈40 分钟）的 Δ ≥ +0.10 m
    方向一致    3 轮平均 Δ > 0，但不满足上一条
    未复现      3 轮平均 Δ ≤ 0
不要求 3 轮都 > 0：D1 上标定后约 10 分钟时差距只有 +0.02 m，r3（≈8 分钟）本来就预期接近 0。
只报告（不设判据）：
    · Δ 是否随时间增大（r3 → r4 → r5）
    · mask 编码器同口径：KNN(mask) − KNN(single)
    · KNN(joint) 离地板（永远猜正中间）有多近
    · joint 补训模型的端到端成绩（D+3 上从没算过）
    · r6 水瓶轮

自检（任何一道不过就停）
----------------------
    ① 训练窗口数必须等于 crossday 训练时的 3391
    ② 重算的 KNN(single) 必须与 crossday/d3/frozen_fusion_results.json 里的 F-KNN 一致
       （每轮 3 种子平均误差相差 ≤ 0.005 m）——确认特征与 KNN 的流程与 4.4 完全相同

用法（项目根目录，csi 环境；需要先跑过 crossday train --day d3 和 frozen_fusion_d3）：
    python -X utf8 -m backend.training.e5_d3
输出：backend/artifacts/crossday/d3/e5_d3_results.json，以及 joint_seed<N>.pt
"""
from __future__ import annotations

import json
import time

import numpy as np

from backend import config
from backend.data.preprocess import Preprocessor
from backend.models.knn import KNNModel
from backend.training import crossday as C
from backend.training.frozen_fusion import encode
from backend.training.frozen_fusion_d3 import lab_metrics

DAY = "d3"
SEEDS = (42, 1, 2)
LINKS = config.all_link_ids()
THR = 0.10                     # 判据阈值，m
CHECK_TOL = 0.005              # 自检②容差，m
JOINT_EPOCHS = {42: 9, 1: 8, 2: 6}    # D1 select pp2 中 ConcatMLP 的 best_epoch + 1（脚本会再核对一次）
ENCS = ("joint", "mask", "single")
OUT = C.OUT_ROOT / DAY / "e5_d3_results.json"


def joint_epochs(seed: int) -> int:
    f = config.ARTIFACTS_DIR / "d1" / "select" / f"cls_ampphase_pp2_seed{seed}" / "results.json"
    ep = json.loads(f.read_text(encoding="utf-8"))["models"]["ConcatMLP"]["best_epoch"] + 1
    if ep != JOINT_EPOCHS[seed]:
        raise SystemExit(f"{f} 里 ConcatMLP 的轮数 {ep} 与脚本写下的 {JOINT_EPOCHS[seed]} 不符——select 结果被改过？停止。")
    return ep


def make_joint(seed: int, grid):
    from backend.models.fusion_baseline import ConcatMLPModel
    kw = dict(lr=1e-3, seed=seed, freq_bins=C.FREQ_BINS, in_ch=len(C.CHANNELS), task="cls",
              n_classes=grid.num_classes, grid_cols=grid.cols)      # 与 crossday.make_model 相同
    return ConcatMLPModel(**kw)


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
        raise SystemExit(f"自检①失败：训练窗口数 {len(ytr)} 与 crossday 训练时的 {meta['n_train']} 不同，停止。")
    rounds = proto["test"] + proto["perturb"]
    part = C.session_windows(DAY, rounds)
    X, lab, bat = C.to_model_input(pp, part), part["label"], part["batch"]
    print(f"==== E5 D+3 复现：训练 D1 {proto['base_rounds']} + {DAY} {proto['calib']}（{len(ytr)} 窗口，"
          f"与 crossday 一致 ✓ 自检①），测试 {proto['test']}，只报告 {proto['perturb']}")

    centre = np.full(len(lab), (grid.rows // 2) * grid.cols + grid.cols // 2)
    floor = {r: lab_metrics(centre[bat == r], lab[bat == r], grid)["mae"] for r in rounds}

    keys = [f"KNN({e})" for e in ENCS] + ["joint 端到端"]
    per = {r: {k: [] for k in keys} for r in rounds}
    for seed in SEEDS:
        t0 = time.time()
        ep = joint_epochs(seed)
        joint = make_joint(seed, grid).fit(Xtr, ytr, epochs=ep)
        joint.save(od / f"joint_seed{seed}")
        mask = C.make_model("mask", seed, grid).load(od / f"mask_seed{seed}")
        cnns = [C.make_model("cnn", seed, grid).load(od / f"cnn_{lid}_seed{seed}") for lid in LINKS]
        enc_models = {"joint": [joint] * len(LINKS), "mask": [mask] * len(LINKS), "single": cnns}

        preds = {"joint 端到端": C.logits(joint, X).argmax(1)}
        for e in ENCS:
            Ftr, Fte = encode(enc_models[e], Xtr), encode(enc_models[e], X)
            preds[f"KNN({e})"] = KNNModel(k=5, standardize=True).fit(Ftr, ytr).predict(Fte)
        for r in rounds:
            mr = bat == r
            for k in keys:
                per[r][k].append(lab_metrics(np.asarray(preds[k])[mr], lab[mr], grid))
        print(f"  种子 {seed} 完成：joint 补训 {ep} 轮  ({time.time() - t0:.0f}s)")

    # 自检②：KNN(single) 与 frozen_fusion_d3 的 F-KNN 一致
    ref = json.loads((od / "frozen_fusion_results.json").read_text(encoding="utf-8"))["rounds"]
    for r in rounds:
        got = float(np.mean([x["mae"] for x in per[r]["KNN(single)"]]))
        want = ref[r]["methods"]["F-KNN"]["mae"]
        if abs(got - want) > CHECK_TOL:
            raise SystemExit(f"自检②失败：{r} KNN(single) 重算 {got:.3f} m，frozen_fusion_results.json 的 F-KNN "
                             f"为 {want:.3f} m —— 特征或 KNN 流程与 4.4 不同，先停下来查原因")
    print("  KNN(single) 重算与 frozen_fusion_results.json 的 F-KNN 一致 ✓ 自检②")

    avg = lambda r, k, m="mae": float(np.mean([x[m] for x in per[r][k]]))
    t = C.round_times(DAY, rounds, proto["calib"])
    res = {"day": DAY, "seeds": list(SEEDS), "joint_epochs": JOINT_EPOCHS,
           "minutes_after_calib": t, "rounds": {}}
    print("\n===== D+3  3 个种子平均；单位 m；Δ = KNN(编码器) − KNN(single)（正 = 比单独训练表征差）")
    for r in rounds:
        role = "test" if r in proto["test"] else "perturb（只报告）"
        print(f"\n[{r}] {role}" + (f"  中点 ≈ 标定后 {t[r]} 分钟" if r in t else "") + f"   地板 {floor[r]:.3f}")
        print(f"  {'方法':12s} {'MAE':>7s} {'(逐种子)':>24s} {'P90':>6s} {'准确率':>6s} {'>1.6m':>6s} {'Δ vs single':>12s}")
        rr = {"role": role, "floor": floor[r], "methods": {}}
        for k in keys:
            s = [x["mae"] for x in per[r][k]]
            d = avg(r, k) - avg(r, "KNN(single)") if k.startswith("KNN(") else None
            rr["methods"][k] = {"mae": avg(r, k), "mae_per_seed": s, "p90": avg(r, k, "p90"),
                                "acc": avg(r, k, "acc"), "far": avg(r, k, "far"), "delta_vs_single": d}
            print(f"  {k:12s} {avg(r, k):7.3f} {' / '.join(f'{v:.3f}' for v in s):>24s} "
                  f"{avg(r, k, 'p90'):6.2f} {avg(r, k, 'acc'):6.0%} {avg(r, k, 'far'):6.0%} "
                  f"{'' if d is None or k == 'KNN(single)' else f'{d:+12.3f}'}")
        res["rounds"][r] = rr

    D = {r: res["rounds"][r]["methods"]["KNN(joint)"]["delta_vs_single"] for r in proto["test"]}
    Dm = {r: res["rounds"][r]["methods"]["KNN(mask)"]["delta_vs_single"] for r in proto["test"]}
    mean_d, last = float(np.mean(list(D.values()))), proto["test"][-1]
    c1, c2 = mean_d >= THR, D[last] >= THR
    verdict = "复现成功" if (c1 and c2) else ("方向一致" if mean_d > 0 else "未复现")
    print("\n===== 判据（跑之前写下，用户 2026-10-14 23:06 确认）")
    print("  Δ（KNN(joint) − KNN(single)）：" + "  ".join(f"{r} {d:+.3f}" for r, d in D.items())
          + f"   3 轮平均 {mean_d:+.3f} m")
    print(f"    {'✓' if c1 else '✗'} 3 轮平均 Δ ≥ +{THR} m")
    print(f"    {'✓' if c2 else '✗'} {last} 的 Δ ≥ +{THR} m")
    print(f"  结论：\"漂移后联合训练表征的近邻结构比单独训练的差\" —— {verdict}")
    print("  只报告：")
    print("    Δ 随时间：" + " → ".join(f"{d:+.3f}" for d in D.values()))
    print("    mask 同口径 Δ（KNN(mask) − KNN(single)）：" + "  ".join(f"{r} {d:+.3f}" for r, d in Dm.items())
          + f"   平均 {float(np.mean(list(Dm.values()))):+.3f}")
    print("    KNN(joint) 与地板之差：" + "  ".join(
        f"{r} {res['rounds'][r]['methods']['KNN(joint)']['mae'] - floor[r]:+.3f}" for r in proto["test"]))
    res.update(test_deltas_joint=D, test_deltas_mask=Dm, mean_delta_joint=mean_d,
               conditions={f"mean>=+{THR}": c1, f"{last}>=+{THR}": c2}, verdict=verdict)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
