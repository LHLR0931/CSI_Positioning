"""E4：为什么"每条链路学充分"就更抗漂移？——检验"冗余（误差不同步）"解释（2026-10-14，跑之前写下判据）

问题
----
E2 显示：r10（标定后约 47 分钟）上，四种编码器单靠一条链路定位都接近地板，单独训练的编码器与联合训练的
编码器差距基本消失（4 条平均：single 1.215 m，joint 1.269 m）。但融合之后差距很大（F-MLP 0.397 m，
ConcatMLP 0.826 m）。所以冻结编码器的优势出在"组合"这一步，而不是每条链路各自更抗漂移。

待检验的解释（冗余）：单独训练时，四个编码器各自从自己的链路里提取全部位置信息，四条链路出错的时机
不同步；融合时一条错了，其余几条能纠正。联合训练的编码器出错更"同步"，组合起来纠不回来。

做法（不训练任何编码器，沿用 E2 的逐链路线性探测）
--------------------------------------------------
对每个种子、每种编码器（joint / sep / mask / single，定义同 e2_linear_probe.py）、每条链路，得到线性
探测器在测试窗口上的 15 类概率。然后算两个量：
    φ（误差相关）   每个窗口每条链路"猜错 = 1、猜对 = 0"；对 6 对链路分别算两个 0/1 序列的相关系数 φ，
                    取平均。φ 越小 = 出错越不同步。这是集成学习里衡量分类器多样性的标准量之一
                    （Kuncheva & Whitaker 2003, Machine Learning 51:181–207 的相关系数 ρ）。
    组合增益        把 4 条链路探测器的概率直接取平均（不训练任何融合参数）再取最大，得到融合误差；
                    组合增益 = 4 条链路各自误差的平均 − 融合误差。增益越大 = 组合起来纠错越多。
                    这个量扣除了"单条链路本身准不准"的影响。
误差与准确率都按轮次计算（train_d1.metrics_for）。

跑之前写下的判据（D1 r9 / r10 之前看过，是诊断分析；只看 r10）
------------------------------------------------------------
    ① 误差更不同步：single 的平均 φ 比 joint、sep 都低，且 3 个种子各自都如此
    ② 组合增益更大：single 的组合增益比 joint、sep 都大 ≥ 0.10 m（3 种子平均）
    ①② 都满足 → "冗余（误差不同步）"解释有数据支持
    只满足一条 → 部分支持，论文按实际情况描述
    都不满足   → 不支持，"为什么学充分就更抗漂移"保持为未解释
r9 与 mask 只报告。

用法（项目根目录；需要 E1 的 ConcatMLP-sep final 模型）：
    python -X utf8 -m backend.training.e4_redundancy
输出：backend/artifacts/d1/final/e4_redundancy.json
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np

from backend import config
from backend.data import sessions
from backend.training import train_d1 as T
from backend.training.e2_linear_probe import CHECK_TOL, ENCS, _feats, encoder_fns

SEEDS = (42, 1, 2)
LINKS = config.all_link_ids()
GAIN_MIN = 0.10
DESC = {"joint": "共享编码器 + 联合", "sep": "独立编码器 + 联合", "mask": "共享 + 联合 + 遮挡（参考）",
        "single": "单独训练（F-MLP 用的）"}
OUT = config.ARTIFACTS_DIR / "d1" / "final" / "e4_redundancy.json"


def probe_proba(Ftr, ytr, Fte, n_classes: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(Ftr), ytr)
    p = np.zeros((len(Fte), n_classes))
    p[:, clf.classes_] = clf.predict_proba(sc.transform(Fte))
    return p


def phi(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    grid = sessions.session_grid("d1")
    data, _ = T.load("final", "ampphase")
    Xtr, ltr = data["train"][0], data["train"][2]
    Xte, lte, bte = data["test"][0], data["test"][2], data["test"][3]
    rounds = sorted(set(bte.tolist()), key=lambda s: int(s[1:]))
    to_xy = lambda lab: np.stack([lab % grid.cols, lab // grid.cols], 1).astype(np.float32)
    # rec[enc][round] = 每个种子的 {phi, indiv, fused, gain}
    rec = {e: {r: [] for r in rounds} for e in ENCS}
    for seed in SEEDS:
        d = config.ARTIFACTS_DIR / "d1" / "final" / T.tag_of("cls", "ampphase", seed)
        ref = json.loads((d / "results.json").read_text(encoding="utf-8"))["models"]
        if "ConcatMLP-sep" not in ref:
            raise SystemExit(f"种子 {seed} 没有 ConcatMLP-sep 的 final 结果：先跑 E1。")
        models = {}
        for name in ("ConcatMLP", "ConcatMLP-mask", "ConcatMLP-sep") + tuple(f"CNN-{l}" for l in LINKS):
            m, _ = T.make_model(name, seed, grid, "cls", Xtr.shape[2])
            models[name] = m.load(d / name)
        for name in ("ConcatMLP", "ConcatMLP-mask", "ConcatMLP-sep"):
            for r, m in T.by_round(models[name].predict(Xte), data["test"], grid).items():
                if abs(m["mean_error_m"] - ref[name]["by_round"][r]["mean_error_m"]) > CHECK_TOL:
                    raise SystemExit(f"!! 种子 {seed} {name} {r} 与 results.json 不一致，停止。")
        P = {e: [] for e in ENCS}                         # P[enc][l] = [N,15] 概率
        for l in range(len(LINKS)):
            for enc, fn in encoder_fns(models, l).items():
                P[enc].append(probe_proba(_feats(fn, Xtr), ltr, _feats(fn, Xte), grid.num_classes))
        for enc in ENCS:
            wrong = [(p.argmax(1) != lte).astype(float) for p in P[enc]]
            indiv = [T.by_round(to_xy(p.argmax(1)), data["test"], grid) for p in P[enc]]
            fused = T.by_round(to_xy(np.mean(P[enc], axis=0).argmax(1)), data["test"], grid)
            for r in rounds:
                mr = bte == r
                ph = float(np.nanmean([phi(wrong[i][mr], wrong[j][mr]) for i, j in combinations(range(len(LINKS)), 2)]))
                ind = float(np.mean([x[r]["mean_error_m"] for x in indiv]))
                fu = fused[r]["mean_error_m"]
                rec[enc][r].append(dict(phi=ph, indiv=ind, fused=fu, gain=ind - fu,
                                        fused_acc=fused[r]["accuracy"]))
        print(f"种子 {seed} 完成（三个端到端模型与 results.json 一致 ✓）")

    avg = lambda e, r, k: float(np.mean([x[k] for x in rec[e][r]]))
    per = lambda e, r, k: [x[k] for x in rec[e][r]]
    out = {"seeds": list(SEEDS), "rounds": {}}
    for r in rounds:
        print(f"\n===== {r}  3 个种子平均（逐种子 42 / 1 / 2）；误差单位 m")
        print(f"  {'编码器':26s} {'φ 误差相关':>10s} {'(逐种子)':>20s} {'单链路平均':>10s} {'概率平均融合':>12s} "
              f"{'组合增益':>9s} {'(逐种子)':>24s}")
        out["rounds"][r] = {}
        for e in ENCS:
            print(f"  {DESC[e]:26s} {avg(e, r, 'phi'):10.3f} {' / '.join(f'{v:.3f}' for v in per(e, r, 'phi')):>20s} "
                  f"{avg(e, r, 'indiv'):10.3f} {avg(e, r, 'fused'):8.3f}({avg(e, r, 'fused_acc'):3.0%}) "
                  f"{avg(e, r, 'gain'):+9.3f} {' / '.join(f'{v:+.3f}' for v in per(e, r, 'gain')):>24s}")
            out["rounds"][r][e] = {k: avg(e, r, k) for k in ("phi", "indiv", "fused", "gain", "fused_acc")}
            out["rounds"][r][e].update(phi_per_seed=per(e, r, "phi"), gain_per_seed=per(e, r, "gain"))

    r = "r10"
    c1 = all(rec["single"][r][i]["phi"] < min(rec["joint"][r][i]["phi"], rec["sep"][r][i]["phi"])
             for i in range(len(SEEDS)))
    dg = {e: avg("single", r, "gain") - avg(e, r, "gain") for e in ("joint", "sep")}
    c2 = all(v >= GAIN_MIN for v in dg.values())
    verdict = ("\"冗余（误差不同步）\"解释有数据支持" if c1 and c2 else
               "部分支持（见各条）" if c1 or c2 else "不支持，保持为未解释")
    print("\n===== 判据（跑之前写下，见文件开头；r10）")
    print(f"  ① single 的 φ 在 3 个种子上都低于 joint 和 sep  {'✓' if c1 else '✗'}"
          f"（平均：single {avg('single', r, 'phi'):.3f}，joint {avg('joint', r, 'phi'):.3f}，"
          f"sep {avg('sep', r, 'phi'):.3f}）")
    print(f"  ② 组合增益差：single − joint {dg['joint']:+.3f} m，single − sep {dg['sep']:+.3f} m"
          f"（都需 ≥ {GAIN_MIN}）  {'✓' if c2 else '✗'}")
    print(f"  结论：{verdict}")
    out.update(conditions={"c1_phi_lower_all_seeds": c1, "c2_gain_diff": dg, "c2": c2}, verdict=verdict)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
