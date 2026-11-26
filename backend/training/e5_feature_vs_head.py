"""E5：联合训练在漂移下更差，问题出在"特征"还是"融合头"？（2026-10-14，跑之前写下判据）

背景
----
r10（标定后约 47 分钟）上：端到端 ConcatMLP 0.826 m，F-MLP（单独训练的编码器冻结 + 重新训练的 MLP 头）
0.397 m，相差 0.43 m。E1 说明差距来自"单独训练"而不是"不共享权重"；E4 排除了"误差不同步"的解释。
E4 还看到：联合训练的编码器配上它自己端到端训练的头（0.826），比在同样特征上做逐链路探测再概率平均
（0.716）还差——提示问题可能有一部分在融合头本身。

两种解释：
    (i)  特征：联合训练编码器的特征拼起来以后，在漂移下组织得不好
    (ii) 头  ：端到端训练出来的融合头学到的跨链路组合方式，在漂移下很脆弱

做法：固定编码器、只换融合方式（不训练任何编码器）
------------------------------------------------
编码器（与 e2_linear_probe.py 相同）：joint（ConcatMLP 共享编码器）、sep（ConcatMLP-sep 独立编码器）、
single（单链路 CNN 编码器，即 F-MLP 用的）、mask（ConcatMLP-mask，参考）。四条链路的 64 维特征拼成 256 维。
融合方式：
    e2e    原来端到端训练出来的头（直接读 results.json；single 没有端到端头）
    refit  冻结编码器，在训练集标准化后的特征上重新训练一个 MLP 头（结构与 ConcatMLP 的头相同，
           frozen_fusion.make_mlp）。轮数按与 F-MLP 完全相同的协议确定：select 阶段用 select 的编码器、
           r1–r5 训练、r6 早停得到最佳轮数；final 阶段用 final 的编码器、r1–r8 训练、轮数 = 最佳轮数 + 1。
           single 的 refit 就是 F-MLP 的复现。
    KNN    k=5、标准化，不训练任何参数

分解（精确成立，因为 refit(single) 就是 F-MLP）：
    e2e(enc) − refit(single) = [e2e(enc) − refit(enc)] + [refit(enc) − refit(single)]
         总差距                     头的作用 head_effect        特征的作用 feat_effect

跑之前写下的判据（D1 r9 / r10 之前看过，是诊断分析；只看 r10，3 种子平均，阈值 0.10 m）
-------------------------------------------------------------------------------
对 joint 和 sep 两种联合训练的编码器分别看：
    只有 feat_effect ≥ 0.10        → 问题主要在特征
    只有 head_effect ≥ 0.10        → 问题主要在端到端训练出来的头
    两者都 ≥ 0.10                  → 两者都有
    两者都 < 0.10                  → 无法判定
joint 与 sep 结论一致才算定论；不一致按实际情况描述。
只报告：KNN 下的特征差距 KNN(enc) − KNN(single)（不依赖任何训练的旁证，方向应与 feat_effect 一致）；
r9；mask。

用法（项目根目录；需要 E1 的 ConcatMLP-sep select / final 模型）：
    python -X utf8 -m backend.training.e5_feature_vs_head
输出：backend/artifacts/d1/final/e5_feature_vs_head.json
"""
from __future__ import annotations

import json

import numpy as np

from backend import config
from backend.data import sessions
from backend.training import train_d1 as T
from backend.training.e2_linear_probe import CHECK_TOL, _feats, encoder_fns
from backend.training.frozen_fusion import make_mlp

SEEDS = (42, 1, 2)
LINKS = config.all_link_ids()
ENCS = ("joint", "sep", "single", "mask")
E2E = {"joint": "ConcatMLP", "sep": "ConcatMLP-sep", "mask": "ConcatMLP-mask"}
DESC = {"joint": "共享编码器 + 联合", "sep": "独立编码器 + 联合", "single": "单独训练（F-MLP 的）",
        "mask": "共享 + 联合 + 遮挡（参考）"}
THR = 0.10
OUT = config.ARTIFACTS_DIR / "d1" / "final" / "e5_feature_vs_head.json"


def load_models(phase: str, seed: int, grid, n_ch: int) -> dict:
    d = config.ARTIFACTS_DIR / "d1" / phase / T.tag_of("cls", "ampphase", seed)
    models = {}
    for name in ("ConcatMLP", "ConcatMLP-mask", "ConcatMLP-sep") + tuple(f"CNN-{l}" for l in LINKS):
        if not (d / f"{name}.pt").exists():
            raise SystemExit(f"找不到 {d / name}.pt：{'先跑 E1' if name == 'ConcatMLP-sep' else '模型缺失'}。")
        m, _ = T.make_model(name, seed, grid, "cls", n_ch)
        models[name] = m.load(d / name)
    return models


def concat_feats(models: dict, X) -> dict:
    """每种编码器：4 条链路的 64 维特征拼成 [N,256]。"""
    per = [encoder_fns(models, l) for l in range(len(LINKS))]
    return {e: np.concatenate([_feats(per[l][e], X) for l in range(len(LINKS))], axis=1) for e in ENCS}


def main() -> None:
    grid = sessions.session_grid("d1")
    to_xy = lambda lab: np.stack([lab % grid.cols, lab // grid.cols], 1).astype(np.float32)

    # ---- select：只为了确定 refit 头的训练轮数（r7 不看） ----
    sdata, _ = T.load("select", "ampphase")
    best_ep = {}
    for seed in SEEDS:
        smodels = load_models("select", seed, grid, sdata["train"][0].shape[2])
        F, Fv = concat_feats(smodels, sdata["train"][0]), concat_feats(smodels, sdata["val"][0])
        for e in ENCS:
            mu, sd = F[e].mean(0), F[e].std(0) + 1e-6
            m = make_mlp(seed, F[e].shape[1], grid).fit(
                (F[e] - mu) / sd, sdata["train"][2], epochs=150, patience=20,
                X_val=(Fv[e] - mu) / sd, y_val=sdata["val"][2], early_stop_on="error")
            best_ep[(seed, e)] = m.best_epoch + 1
        print(f"select 种子 {seed}：refit 头的训练轮数 " + "  ".join(f"{e} {best_ep[(seed, e)]}" for e in ENCS))
    del sdata

    # ---- final ----
    data, _ = T.load("final", "ampphase")
    Xtr, ltr, Xte = data["train"][0], data["train"][2], data["test"][0]
    rec = {e: {h: {} for h in ("e2e", "refit", "KNN")} for e in ENCS}   # rec[enc][head][round] = [每种子误差]
    acc = {e: {h: {} for h in ("e2e", "refit", "KNN")} for e in ENCS}
    ffm = {}
    for seed in SEEDS:
        d = config.ARTIFACTS_DIR / "d1" / "final" / T.tag_of("cls", "ampphase", seed)
        ref = json.loads((d / "results.json").read_text(encoding="utf-8"))["models"]
        ff = d / "frozen_fusion.json"
        if ff.exists():
            ffm[seed] = json.loads(ff.read_text(encoding="utf-8"))["models"]["F-MLP"]["by_round"]
        models = load_models("final", seed, grid, Xtr.shape[2])
        for e, name in E2E.items():                                    # e2e：与主表核对后直接用
            for r, m in T.by_round(models[name].predict(Xte), data["test"], grid).items():
                want = ref[name]["by_round"][r]["mean_error_m"]
                if abs(m["mean_error_m"] - want) > CHECK_TOL:
                    raise SystemExit(f"!! 种子 {seed} {name} {r} 复算 {m['mean_error_m']:.3f}，results.json {want:.3f}，停止。")
                rec[e]["e2e"].setdefault(r, []).append(m["mean_error_m"])
                acc[e]["e2e"].setdefault(r, []).append(m["accuracy"])
        Ftr, Fte = concat_feats(models, Xtr), concat_feats(models, Xte)
        from backend.models.knn import KNNModel
        for e in ENCS:
            mu, sd = Ftr[e].mean(0), Ftr[e].std(0) + 1e-6
            mlp = make_mlp(seed, Ftr[e].shape[1], grid).fit((Ftr[e] - mu) / sd, ltr, epochs=best_ep[(seed, e)])
            preds = {"refit": mlp.predict((Fte[e] - mu) / sd),
                     "KNN": to_xy(KNNModel(k=5, standardize=True).fit(Ftr[e], ltr).predict(Fte[e]))}
            for h, p in preds.items():
                for r, m in T.by_round(p, data["test"], grid).items():
                    rec[e][h].setdefault(r, []).append(m["mean_error_m"])
                    acc[e][h].setdefault(r, []).append(m["accuracy"])
        print(f"final 种子 {seed} 完成（三个端到端模型与 results.json 一致 ✓）")

    mean = lambda e, h, r: float(np.mean(rec[e][h][r])) if rec[e][h].get(r) else float("nan")
    rounds = sorted(rec["joint"]["refit"], key=lambda s: int(s[1:]))
    out = {"seeds": list(SEEDS), "refit_epochs": {f"{s}_{e}": v for (s, e), v in best_ep.items()}, "rounds": {}}
    for r in rounds:
        print(f"\n===== {r}  3 个种子平均（逐种子 42 / 1 / 2）；单位 m")
        print(f"  {'编码器':26s} {'e2e 原来的头':>22s} {'refit 重训 MLP 头':>26s} {'KNN（不训练）':>24s}")
        out["rounds"][r] = {}
        for e in ENCS:
            cell = lambda h: ("—" if not rec[e][h].get(r) else
                              f"{mean(e, h, r):.3f} ({' / '.join(f'{v:.3f}' for v in rec[e][h][r])})")
            print(f"  {DESC[e]:26s} {cell('e2e'):>22s} {cell('refit'):>26s} {cell('KNN'):>24s}")
            out["rounds"][r][e] = {h: {"mae": mean(e, h, r), "per_seed": rec[e][h].get(r, []),
                                       "acc": float(np.mean(acc[e][h][r])) if acc[e][h].get(r) else None}
                                   for h in ("e2e", "refit", "KNN")}
        if ffm:
            fm = float(np.mean([ffm[s][r]["mean_error_m"] for s in ffm]))
            print(f"  （核对：frozen_fusion 里的 F-MLP = {fm:.3f}，本脚本 refit(single) = {mean('single', 'refit', r):.3f}；"
                  f"GPU 训练不完全确定，相差 0.0x m 属正常）")
        print("  分解：总差距 = 头的作用 + 特征的作用")
        for e in ("joint", "sep", "mask"):
            tot = mean(e, "e2e", r) - mean("single", "refit", r)
            head = mean(e, "e2e", r) - mean(e, "refit", r)
            feat = mean(e, "refit", r) - mean("single", "refit", r)
            knn = mean(e, "KNN", r) - mean("single", "KNN", r)
            print(f"    {e:6s} 总差距 {tot:+.3f} = 头 {head:+.3f} + 特征 {feat:+.3f}   （旁证：KNN 下特征差距 {knn:+.3f}）")
            out["rounds"][r][f"decomp_{e}"] = {"total": tot, "head_effect": head, "feat_effect": feat,
                                                "knn_feat_gap": knn}

    print("\n===== 判据（跑之前写下，见文件开头；r10）")
    verdicts = {}
    for e in ("joint", "sep"):
        dc = out["rounds"]["r10"][f"decomp_{e}"]
        H, F = dc["head_effect"] >= THR, dc["feat_effect"] >= THR
        verdicts[e] = {(True, False): "问题主要在端到端训练出来的头", (False, True): "问题主要在特征",
                       (True, True): "两者都有", (False, False): "无法判定"}[(H, F)]
        print(f"  {e:6s} 头 {dc['head_effect']:+.3f}（{'≥' if H else '<'} {THR}）  特征 {dc['feat_effect']:+.3f}"
              f"（{'≥' if F else '<'} {THR}） → {verdicts[e]}")
    final = verdicts["joint"] if verdicts["joint"] == verdicts["sep"] else \
        f"joint 与 sep 不一致（joint：{verdicts['joint']}；sep：{verdicts['sep']}），按实际情况描述"
    print(f"  结论：{final}")
    out.update(verdicts=verdicts, verdict=final)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
