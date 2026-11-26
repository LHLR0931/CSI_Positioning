"""健康分有效性检验：同一窗口内，健康分高的链路，单链路定位误差是不是更小？

检验方法
--------
配对单位是（窗口, 链路）。每个窗口 4 条链路两两组合 6 对；对每一对问：
"q 高的那条，单链路误差 e 是否更小"。数一致的比例：
    一致 = (q_i > q_j 且 e_i < e_j) 或 (q_i < q_j 且 e_i > e_j)
    q 相等的对记 0.5；e 相等的对（例如两条都猜对，e 都是 0）没有信息，不计入。
0.5 = 健康分没用；越接近 1 越有用。这个一致比例在数学上就是 AUROC。
在同一窗口内部比，是为了控制"窗口本身的难度"（人站在难分的点上，四条链路都会错）。

为什么用单链路模型的误差：健康分要回答的是"这条链路此刻可不可靠"，单链路模型的误差
是这条链路单独提供信息时的可靠程度的直接度量。

不碰最终测试集的用法（2026-10-08）
----------------------------------
用选模型阶段的结构调指标：r5 = 标定轮（piB 11:04 重启后第一轮），在 r6 / r7 上检验，
单链路模型取 select 阶段训练好的（训练 r1–r5）。r9 / r10 不参与调参。

用法（项目根目录）：
    python -X utf8 -m backend.features.link_health d1 --calib r5
    python -X utf8 -m backend.features.eval_health d1 --phase select --tag cls_ampphase --seeds 42 1 2
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

from backend import config
from backend.data import sessions
from backend.features.link_health import INDICATORS
from backend.training import train_d1

LINKS = config.all_link_ids()


def concordance(q: np.ndarray, e: np.ndarray) -> tuple[float, int]:
    """q, e: [Nw, L]。返回（一致比例, 有效对数）。"""
    num, den = 0.0, 0
    for i, j in combinations(range(q.shape[1]), 2):
        de = e[:, i] - e[:, j]
        dq = q[:, i] - q[:, j]
        valid = de != 0
        den += int(valid.sum())
        num += float((np.sign(dq[valid]) == -np.sign(de[valid])).sum()
                     + 0.5 * (dq[valid] == 0).sum())
    return (num / den if den else float("nan")), den


def single_link_errors(phase: str, tag: str, rounds: list[str]):
    """用已训练好的单链路模型，算每个测试窗口、每条链路的定位误差（米）。"""
    task, channels = tag.split("_")[0], tag.split("_")[1]
    seed = int(tag.split("seed")[1])
    grid = sessions.session_grid("d1")
    data, _ = train_d1.load(phase, channels)
    X, yxy, lab, bat = data["test"]
    if phase == "select":                 # r6 是 select 阶段的验证集，也拿来检验健康分
        Xv, yv, lv, bv = data["val"]
        X, yxy, lab, bat = (np.concatenate([X, Xv]), np.concatenate([yxy, yv]),
                            np.concatenate([lab, lv]), np.concatenate([bat, bv]))
    keep = np.isin(bat, rounds)
    X, yxy, lab, bat = X[keep], yxy[keep], lab[keep], bat[keep]
    errs = np.zeros((len(lab), len(LINKS)))
    for l, lid in enumerate(LINKS):
        model, _ = train_d1.make_model(f"CNN-{lid}", seed, grid, task, X.shape[2])
        model.load(train_d1.out_dir(phase, tag) / f"CNN-{lid}")
        pred = model.predict(X[:, l:l + 1])
        errs[:, l] = np.linalg.norm(pred - yxy, axis=1) * grid.spacing_m
    return errs, lab, bat


def main(session: str, phase: str, tag_base: str, seeds: list[int]) -> None:
    h = np.load(config.DATASET_ARTIFACTS_DIR / session / "link_health.npz")
    calib = [str(c) for c in h["calib_rounds"]]
    rounds = {"select": ["r6", "r7"], "dev": ["r1", "r2", "r3"], "final": ["r9", "r10"]}[phase]
    want_calib = {"select": ["r5"], "dev": ["r5"], "final": ["r8"]}[phase]
    if calib != want_calib:
        raise SystemExit(f"link_health.npz 的标定轮是 {calib}，{phase} 阶段应为 {want_calib}。"
                         f"先运行 python -X utf8 -m backend.features.link_health {session} --calib {want_calib[0]}")
    print(f"标定轮={calib}  检验轮={rounds}  单链路模型={tag_base}_seed{seeds}\n")
    # 指纹偏离度的不封顶版本：距离 ÷ 该链路标定窗口距离的中位数（分位数超过标定轮最差窗口后会封顶在 1）
    cal_m = np.isin(h["batch"], calib)
    fp_ratio = h["raw_fp_dist"] / np.median(h["raw_fp_dist"][cal_m], axis=0, keepdims=True)
    names = ["q_mean", "q_max", *[f"仅{k}" for k in INDICATORS], "仅fp比值"]
    scores = {name: [] for name in names}
    rows = []
    for seed in seeds:
        e, lab, bat = single_link_errors(phase, f"{tag_base}_seed{seed}", rounds)
        for r in rounds:
            m_e = bat == r
            m_h = h["batch"] == r
            if not np.array_equal(h["label"][m_h], lab[m_e]):
                raise SystemExit(f"{r}：健康分窗口与模型窗口顺序对不上，检查切窗逻辑")
            cand = {"q_mean": h["q_mean"][m_h], "q_max": h["q_max"][m_h],
                    **{f"仅{k}": 1 - h[f"ind_{k}"][m_h] for k in INDICATORS},
                    "仅fp比值": -fp_ratio[m_h]}
            for name, q in cand.items():
                c, n = concordance(q, e[m_e])
                scores[name].append(c)
                rows.append((seed, r, name, c, n))
    print(f"{'健康分':10s}" + "".join(f"{f's{s}-{r}':>10s}" for s in seeds for r in rounds) + f"{'平均':>8s}")
    for name in scores:
        vals = [c for (s, r, nm, c, n) in rows if nm == name]
        print(f"{name:10s}" + "".join(f"{v:>10.3f}" for v in vals) + f"{np.mean(vals):>8.3f}")
    n_pairs = [n for (s, r, nm, c, n) in rows if nm == "q_mean"]
    print(f"\n每格的有效链路对数：{n_pairs}（两条链路误差相同的对不计入）")
    print("0.5 = 健康分没用；> 0.5 = 健康分高的链路误差确实更小。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="健康分有效性检验（窗口内一致比例 = AUROC）")
    ap.add_argument("session")
    ap.add_argument("--phase", default="select", choices=["select", "dev", "final"])
    ap.add_argument("--tag", default="cls_ampphase", help="单链路模型的设置，例如 cls_ampphase")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    a = ap.parse_args()
    main(a.session, a.phase, a.tag, a.seeds)
