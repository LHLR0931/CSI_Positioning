"""随机划分 vs 按轮次划分：同一份 D1 数据上，随机划分会让结果好多少？（2026-10-28，诊断）

为什么做
--------
论文第 1 条贡献是"按轮次 / 跨天评估，揭示随机划分掩盖的时间漂移"。在此之前，"随机划分偏乐观"
只有间接证据：本文展示了漂移存在，再引用前人用了 6:2:2 随机划分。本脚本在**本文自己的数据**上
按前人的方式划分一次，给出直接对照数字。

证据等级：**诊断**。D1 的 r9 / r10 之前已看过；本实验不选模型、不调参，只演示两种划分方式报出的
误差差多少，不涉及"测试集只看一次"。

做法
----
数据：D1 全部 10 轮（r1–r10），pp2 预处理，16 帧不重叠窗口，幅度 + 相位，15 类分类。
随机划分：把所有窗口按格点分层，**6:2:2 随机**分成训练 / 验证 / 测试（与 Wang 2026 相同比例），
          划分种子固定为 42（所有模型种子共用同一个划分）。预处理统计量只用训练窗口拟合。
模型：
    Center          永远猜正中间（地板）
    KNN             四链路 KNN（k=5，440 维，训练集标准化），与 train_d1 完全相同
    ConcatMLP-mask  与 train_d1 完全相同；在验证集上早停（看定位误差，最多 150 epoch，patience 20）
                    模型种子 42 / 1 / 2
对照（不重新训练，直接读已有结果）：D1 final（训练 r1–r8，测 r9 / r10）的 KNN 与 ConcatMLP-mask，
    `backend/artifacts/d1/final/cls_ampphase_pp2_seed<N>/results.json`。
统计单位：轮次。随机划分的测试窗口来自全部 10 轮，按轮次分别算误差，主要看 r10（与按轮次划分的
    r10 同一批时间）；10 轮平均与 r9 只报告。

两种划分的已知差别（论文里要写明）
-----------------------------------
    1. 训练集：随机划分约 60% × 全部窗口（约 2300 个，含 r9 / r10 的窗口）；按轮次是 r1–r8 全部（3013 个）。
       随机划分的训练窗口**更少**，所以它若更好，不能归因于"训练数据更多"。
    2. 轮数：随机划分有验证集，用早停；按轮次 final 用 select 阶段定下的固定轮数。
    3. 子载波筛选是结构性的（pp2 固定只留 55 个有效子载波），两种划分得到同一组子载波。

判据（运行前冻结于 2026-10-28；按"结论粗细原则"，要求相差一倍以上）
-------------------------------------------------------------------------------------
    ① ConcatMLP-mask：随机划分下 r10 的平均误差（3 种子平均）≤ 按轮次划分 r10 平均误差的一半；
    ② KNN：随机划分下 r10 的平均误差 ≤ 按轮次划分 r10 平均误差的一半。
    ①② 都满足 → "随机划分明显偏乐观"成立；只满足一条 → 部分成立；都不满足 → 不成立。
    无论结果如何都照实报告。

用法（项目根目录）：
    python -X utf8 -m backend.training.random_split_check --seed 42 1 2
    python -X utf8 -m backend.training.random_split_check --summary      # 只汇总、判定
输出：backend/artifacts/d1/random_split/random_split_check.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from backend import config
from backend.data import sessions
from backend.training.prepare import load_stage2_windows
from backend.training.train_d1 import (SESSION, KNN_K, PREPROC, by_round, make_model, tag_of)

RATIOS = (0.6, 0.2, 0.2)       # 与 Wang 2026 相同
SPLIT_SEED = config.RANDOM_SEED  # 42，所有模型种子共用同一个划分
MODELS = ("Center", "KNN", "ConcatMLP-mask")
MAIN_ROUND = "r10"
OUT = config.ARTIFACTS_DIR / "d1" / "random_split"
RES = OUT / "random_split_check.json"


def load_random():
    if SPLIT_SEED != config.RANDOM_SEED:
        raise SystemExit("划分种子必须是 config.RANDOM_SEED")
    data, pp = load_stage2_windows(
        T=config.STAGE2_TIME_WINDOW, mode="stratified", session=SESSION,
        subcarrier_mode="train_fold", return_batches=True, ratios=RATIOS)
    return data, pp


def mean_over_rounds(by_r: dict) -> float:
    return float(np.mean([m["mean_error_m"] for m in by_r.values()]))


def run(seeds: list[int]) -> dict:
    grid = sessions.session_grid(SESSION)
    data, pp = load_random()
    Xtr, ytr, ltr, btr = data["train"]
    Xva, _, lva, _ = data["val"]
    Xte, _, _, bte = data["test"]
    n = {k: len(data[k][1]) for k in ("train", "val", "test")}
    print(f"随机划分 6:2:2（种子 {SPLIT_SEED}）窗口数 train/val/test = {n['train']}/{n['val']}/{n['test']}")
    print(f"测试窗口来自轮次：{sorted(set(bte.tolist()), key=lambda s: int(s[1:]))}\n")

    res = {"session": SESSION, "preproc": PREPROC, "ratios": RATIOS, "split_seed": SPLIT_SEED,
           "n_windows": n, "models": {}}
    center = np.array([(grid.cols - 1) / 2, (grid.rows - 1) / 2], np.float32)
    res["models"]["Center"] = {"by_round": by_round(np.tile(center, (len(Xte), 1)), data["test"], grid)}

    from backend.models.knn import KNNModel
    from backend.models.sample_gcn import window_features
    knn = KNNModel(k=KNN_K, standardize=True).fit(window_features(Xtr), ltr)
    lab = knn.predict(window_features(Xte))
    pred = np.stack([lab % grid.cols, lab // grid.cols], axis=1).astype(np.float32)
    res["models"]["KNN"] = {"by_round": by_round(pred, data["test"], grid)}

    res["models"]["ConcatMLP-mask"] = {"per_seed": {}}
    for s in seeds:
        t0 = time.time()
        model, _ = make_model("ConcatMLP-mask", s, grid, "cls", Xtr.shape[2])
        model.fit(Xtr, ltr, epochs=150, patience=20, X_val=Xva, y_val=lva, early_stop_on="error")
        row = {"by_round": by_round(model.predict(Xte), data["test"], grid),
               "best_epoch": model.best_epoch, "seconds": round(time.time() - t0, 1)}
        res["models"]["ConcatMLP-mask"]["per_seed"][str(s)] = row
        print(f"ConcatMLP-mask 种子 {s} 训练完成（最佳 epoch {model.best_epoch}，{row['seconds']:.0f}s）")

    OUT.mkdir(parents=True, exist_ok=True)
    pp.save(OUT / "preprocessor.json")
    RES.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {RES}\n")
    return res


def by_round_reference(seeds: list[int]) -> dict:
    """D1 final（按轮次）的已有结果：KNN 与 ConcatMLP-mask 在 r9 / r10 的平均误差。"""
    out = {"KNN": {}, "ConcatMLP-mask": {}}
    for s in seeds:
        f = config.ARTIFACTS_DIR / "d1" / "final" / tag_of("cls", "ampphase", s) / "results.json"
        if not f.exists():
            raise SystemExit(f"找不到按轮次划分的对照结果 {f}")
        models = json.loads(f.read_text(encoding="utf-8"))["models"]
        for name in out:
            if name not in models:
                raise SystemExit(f"{f} 里没有 {name}")
            out[name][str(s)] = {r: m["mean_error_m"] for r, m in models[name]["by_round"].items()}
    return out


def summary(seeds: list[int]) -> None:
    if not RES.exists():
        raise SystemExit(f"还没有结果 {RES}，先训练")
    res = json.loads(RES.read_text(encoding="utf-8"))
    ref = by_round_reference(seeds)
    rounds = sorted(res["models"]["Center"]["by_round"], key=lambda s: int(s[1:]))

    def rnd(name):
        """随机划分下某模型的 {轮次: 平均误差}（神经网络取种子平均）及准确率。"""
        m = res["models"][name]
        if "per_seed" in m:
            rows = [m["per_seed"][str(s)]["by_round"] for s in seeds]
            return ({r: float(np.mean([x[r]["mean_error_m"] for x in rows])) for r in rounds},
                    {r: float(np.mean([x[r]["accuracy"] for x in rows])) for r in rounds})
        return ({r: m["by_round"][r]["mean_error_m"] for r in rounds},
                {r: m["by_round"][r]["accuracy"] for r in rounds})

    print(f"随机划分 6:2:2 窗口数 train/val/test = "
          f"{res['n_windows']['train']}/{res['n_windows']['val']}/{res['n_windows']['test']}")
    print(f"\n{'模型':16s} {'随机 r9':>9s} {'随机 r10':>9s} {'随机 10轮平均':>13s} {'按轮次 r9':>10s} {'按轮次 r10':>11s}")
    verdict = {}
    for name in MODELS:
        err, acc = rnd(name)
        mean10 = float(np.mean(list(err.values())))
        if name == "Center":
            print(f"{name:16s} {err['r9']:9.3f} {err['r10']:9.3f} {mean10:13.3f} {'—':>10s} {'—':>11s}")
            continue
        r9 = float(np.mean([ref[name][str(s)]["r9"] for s in seeds]))
        r10 = float(np.mean([ref[name][str(s)]["r10"] for s in seeds]))
        print(f"{name:16s} {err['r9']:9.3f} {err['r10']:9.3f} {mean10:13.3f} {r9:10.3f} {r10:11.3f}"
              f"   （随机 r10 准确率 {acc['r10']:.0%}）")
        verdict[name] = {"random_r10": err["r10"], "by_round_r10": r10, "ratio": err["r10"] / r10,
                         "pass": err["r10"] <= 0.5 * r10}

    print("\n判据（跑前写下）：随机划分 r10 ≤ 按轮次 r10 的一半")
    for name, v in verdict.items():
        print(f"  {name:16s} {v['random_r10']:.3f} / {v['by_round_r10']:.3f} = {v['ratio']:.2f}  "
              f"{'✓' if v['pass'] else '✗'}")
    k = sum(v["pass"] for v in verdict.values())
    print("判定：" + {2: "随机划分明显偏乐观——成立", 1: "部分成立", 0: "不成立"}[k])
    if "ConcatMLP-mask" in res["models"]:
        per = res["models"]["ConcatMLP-mask"]["per_seed"]
        print("ConcatMLP-mask 随机 r10 逐种子：" + " / ".join(
            f"{per[str(s)]['by_round']['r10']['mean_error_m']:.3f}" for s in seeds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="随机划分 vs 按轮次划分（D1，诊断）")
    ap.add_argument("--seed", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--summary", action="store_true", help="只汇总已有结果并判定，不训练")
    args = ap.parse_args()
    if not args.summary:
        run(args.seed)
    summary(args.seed)
