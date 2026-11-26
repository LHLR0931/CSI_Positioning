"""E2：联合训练是否让部分链路"学得不充分"？——逐链路线性探测（2026-10-14，跑之前写下判据）

背景
----
E1 显示：冻结单链路编码器（F-MLP）比端到端联合训练好，主要靠"每条链路单独训练"，而不是"不共享权重"。
机器学习文献（Du et al. ICML 2023 的 modality laziness；Huang et al. ICML 2022 的 modality competition）
的解释是：联合训练时，网络只要拼起来能猜对就行，于是部分输入的编码器学得不充分。
Du et al. 表 1 的检验方法：把编码器冻结，只用某一个模态的特征训练一个线性分类器，
比较"联合训练得到的编码器"和"单模态训练得到的编码器"。本脚本对四条链路做同样的检验。

做法（不训练任何编码器，D1 final 已有的模型）
---------------------------------------------
对每个种子、每条链路 l，取以下编码器在链路 l 上的 64 维特征：
    joint   ConcatMLP 的共享编码器（共享编码器 + 联合训练）
    sep     ConcatMLP-sep 的第 l 个编码器（独立编码器 + 联合训练，E1 新训练的）
    mask    ConcatMLP-mask 的共享编码器（参考）
    single  单链路 CNN-<l> 的编码器（单独训练，也就是 F-MLP 用的那个）
只用这条链路的特征，在训练集（D1 r1–r8）上拟合线性分类器（标准化 + 逻辑回归 C=0.1，与项目里的线性
基线同设置），在 r9 / r10 上评估。指标与主表相同（train_d1.metrics_for）。
脚本先核对三个端到端模型四条链路都在时的成绩与 results.json 一致。

跑之前写下的判据（D1 r9 / r10 之前看过，是诊断分析）
-----------------------------------------------
gap_l = 联合训练编码器在链路 l 上的探测误差 − single 编码器在链路 l 上的探测误差（r9，3 种子平均，m）
    ① 整体学得不充分：joint 与 sep 的 4 条链路平均 gap 都 ≥ 0.10 m
    ② 只学了一部分：sep 的 4 个 gap 中，最大与最小相差 ≥ 0.15 m
    ①② 都满足 → "联合训练让部分链路学得不充分"成立（在本数据上复现 Du et al. 表 1 的现象）
    只满足 ①   → "联合训练让各链路整体学得不充分，但不集中在部分链路"
    ① 不满足   → 不成立（E1 的差距另有原因）
只报告、不设判据：
    - joint 中，10-13 去链路检验里被依赖的两条（piA_apY、piB_apX）的 gap 是否小于另外两条
    - mask 的平均 gap 是否介于 joint 与 0 之间（遮挡训练部分缓解）
    - r10 的同样表格
局限（论文要写）：联合训练的编码器本来就不是为"单条链路定位"训练的，探测误差偏高有一部分是目标不同造成的；
这正是 Du et al. 用的同一种度量，结论按"与该文献同口径"来表述。

用法（项目根目录；需要先跑完 E1 的 ConcatMLP-sep select / final）：
    python -X utf8 -m backend.training.e2_linear_probe
输出：backend/artifacts/d1/final/e2_linear_probe.json
"""
from __future__ import annotations

import json

import numpy as np

from backend import config
from backend.data import sessions
from backend.training import train_d1 as T

SEEDS = (42, 1, 2)
LINKS = config.all_link_ids()
RELIED = ("piA_apY", "piB_apX")          # 10-13 去链路检验：ConcatMLP 主要依赖的两条
ENCS = ("joint", "sep", "mask", "single")
DESC = {"joint": "ConcatMLP 共享编码器（联合）", "sep": "ConcatMLP-sep 独立编码器（联合）",
        "mask": "ConcatMLP-mask 共享编码器（参考）", "single": "单链路 CNN 编码器（单独训练）"}
GAP_ALL, GAP_SPREAD = 0.10, 0.15
CHECK_TOL = 0.005
OUT = config.ARTIFACTS_DIR / "d1" / "final" / "e2_linear_probe.json"


def _feats(fn, X, batch: int = 512) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            out.append(fn(X[s:s + batch]))
    return np.concatenate(out)


def encoder_fns(models: dict, l: int) -> dict:
    """每种编码器在链路 l 上的特征函数：输入 [N,L,C,T,S] 的 numpy，输出 [N,64]。"""
    import torch

    def run(enc):
        enc = enc.eval()
        dev = next(enc.parameters()).device
        return lambda xb: enc(torch.tensor(np.asarray(xb[:, l]), dtype=torch.float32, device=dev)).cpu().numpy()
    return {"joint": run(models["ConcatMLP"].net.encoder),
            "sep": run(models["ConcatMLP-sep"].net.encoders[l]),
            "mask": run(models["ConcatMLP-mask"].net.encoder),
            "single": run(models[f"CNN-{LINKS[l]}"].net.encoder)}


def probe(Ftr, ytr, Fte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr)
    return LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(Ftr), ytr).predict(sc.transform(Fte))


def main() -> None:
    grid = sessions.session_grid("d1")
    data, _ = T.load("final", "ampphase")
    Xtr, ltr = data["train"][0], data["train"][2]
    Xte = data["test"][0]
    n_ch = Xtr.shape[2]
    # res[enc][link][round] = 每个种子的 {mae, acc}
    res = {e: {lid: {} for lid in LINKS} for e in ENCS}
    for seed in SEEDS:
        d = config.ARTIFACTS_DIR / "d1" / "final" / T.tag_of("cls", "ampphase", seed)
        ref = json.loads((d / "results.json").read_text(encoding="utf-8"))["models"]
        if "ConcatMLP-sep" not in ref:
            raise SystemExit(f"种子 {seed} 没有 ConcatMLP-sep 的 final 结果：先跑 E1（见 e1_shared_encoder.py）。")
        models = {}
        for name in ("ConcatMLP", "ConcatMLP-mask", "ConcatMLP-sep") + tuple(f"CNN-{l}" for l in LINKS):
            m, _ = T.make_model(name, seed, grid, "cls", n_ch)
            models[name] = m.load(d / name)
        for name in ("ConcatMLP", "ConcatMLP-mask", "ConcatMLP-sep"):     # 与主表核对
            for r, m in T.by_round(models[name].predict(Xte), data["test"], grid).items():
                want = ref[name]["by_round"][r]["mean_error_m"]
                if abs(m["mean_error_m"] - want) > CHECK_TOL:
                    raise SystemExit(f"!! 种子 {seed} {name} {r}：复算 {m['mean_error_m']:.3f}，"
                                     f"results.json {want:.3f} —— 模型文件与主表不是同一批，停止。")
        for l, lid in enumerate(LINKS):
            for enc, fn in encoder_fns(models, l).items():
                lab = probe(_feats(fn, Xtr), ltr, _feats(fn, Xte))
                pred = np.stack([lab % grid.cols, lab // grid.cols], 1).astype(np.float32)
                for r, m in T.by_round(pred, data["test"], grid).items():
                    res[enc][lid].setdefault(r, []).append({"mae": m["mean_error_m"], "acc": m["accuracy"]})
        print(f"种子 {seed} 完成（三个端到端模型与 results.json 一致 ✓）")

    avg = lambda enc, lid, r, k="mae": float(np.mean([x[k] for x in res[enc][lid][r]]))
    out = {"seeds": list(SEEDS), "rounds": {}}
    for r in ("r9", "r10"):
        print(f"\n===== {r}  逐链路线性探测：平均误差 m（准确率），3 种子平均")
        print(f"  {'编码器':34s} " + " ".join(f"{l:>14s}" for l in LINKS) + f" {'4 条平均':>10s}")
        out["rounds"][r] = {}
        for enc in ENCS:
            cells = " ".join(f"{avg(enc, l, r):7.3f}({avg(enc, l, r, 'acc'):4.0%})" for l in LINKS)
            mean = float(np.mean([avg(enc, l, r) for l in LINKS]))
            print(f"  {DESC[enc]:34s} {cells} {mean:10.3f}")
            out["rounds"][r][enc] = {l: {"mae": avg(enc, l, r), "acc": avg(enc, l, r, "acc"),
                                         "mae_per_seed": [x["mae"] for x in res[enc][l][r]]} for l in LINKS}
        print(f"  {'gap = 该编码器 − single':34s}")
        for enc in ("joint", "sep", "mask"):
            g = {l: avg(enc, l, r) - avg("single", l, r) for l in LINKS}
            out["rounds"][r][f"gap_{enc}"] = g
            print(f"    {enc:30s} " + " ".join(f"{g[l]:+14.3f}" for l in LINKS) + f" {np.mean(list(g.values())):+10.3f}")

    g = out["rounds"]["r9"]
    mj, ms, mm = (float(np.mean(list(g[f"gap_{e}"].values()))) for e in ("joint", "sep", "mask"))
    spread = max(g["gap_sep"].values()) - min(g["gap_sep"].values())
    c1, c2 = mj >= GAP_ALL and ms >= GAP_ALL, spread >= GAP_SPREAD
    verdict = ("\"联合训练让部分链路学得不充分\"成立" if c1 and c2 else
               "\"联合训练让各链路整体学得不充分，但不集中在部分链路\"" if c1 else "不成立")
    print("\n===== 判据（跑之前写下，见文件开头；r9）")
    print(f"  ① 平均 gap：joint {mj:+.3f}、sep {ms:+.3f}（都需 ≥ {GAP_ALL}） {'✓' if c1 else '✗'}")
    print(f"  ② sep 的 gap 最大 − 最小 = {spread:.3f}（需 ≥ {GAP_SPREAD}） {'✓' if c2 else '✗'}")
    print(f"  结论：{verdict}")
    gj = g["gap_joint"]
    relied = float(np.mean([gj[l] for l in RELIED]))
    others = float(np.mean([gj[l] for l in LINKS if l not in RELIED]))
    print(f"  只报告：joint 中被依赖的 {RELIED} 平均 gap {relied:+.3f}，另外两条 {others:+.3f}；"
          f"mask 平均 gap {mm:+.3f}（joint {mj:+.3f}）")
    out.update(mean_gap={"joint": mj, "sep": ms, "mask": mm}, sep_spread=spread,
               relied_vs_others={"relied": relied, "others": others},
               conditions={"c1": c1, "c2": c2}, verdict=verdict)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
