"""阶段二"问题 1"检验：不加遮挡的 ConcatMLP 是否过度依赖某一条链路？（2026-10-13，跑之前写下判据）

做法（不重新训练）
------------------
用 D1 final（pp2，cls，ampphase）已经训练好的 ConcatMLP 和 ConcatMLP-mask，3 个种子，
在 r9 / r10 上推理时依次"去掉"一条链路：把这条链路的编码特征（编码器之后、拼接之前）换掉，
其余三条不变。有两种换法：

    zero  置 0。与 ConcatMLP-mask 训练时的遮挡方式相同，对 mask 模型是"见过的"输入。
    mean  换成这条链路在训练集上的平均编码特征。对没见过遮挡的 ConcatMLP 更公平，
          用来排除"ConcatMLP 只是不认识全 0 输入"这种解释。

指标与 D1 主表完全相同（train_d1.metrics_for），3 个种子取平均。"四条都在"的结果必须与
results.json 里的主表数字一致，脚本会核对。

跑之前写下的判据（r9 / r10 已经看过，所以这是诊断分析，不是预登记检验）
--------------------------------------------------------------------
对每个模型、每一轮、每种换法：
    Δ_l    = 去掉链路 l 后的平均误差 − 四条都在的平均误差
    集中度 = max_l Δ_l ÷ Σ_l max(Δ_l, 0)        四条链路影响一样大时 = 0.25

"ConcatMLP 过度依赖某一条链路"成立，需要在 r9、r10 两轮、两种换法下都满足：
    ① ConcatMLP 的集中度 ≥ 0.5，即一条链路造成至少一半的损失；
    ② 对这条链路，ConcatMLP 的 Δ 比 ConcatMLP-mask 的 Δ 大 0.1 m 以上。
只满足一部分 → 论文按实际情况描述；都不满足 → 论文只写"随机遮挡让融合更好、更稳"。

用法（项目根目录）：
    python -X utf8 -m backend.training.link_drop_check
输出：backend/artifacts/d1/final/link_drop_check.json
"""
from __future__ import annotations

import json

import numpy as np

from backend import config
from backend.data import sessions
from backend.training import train_d1 as T

SEEDS = (42, 1, 2)
MODELS = ("ConcatMLP", "ConcatMLP-mask")
MODES = ("zero", "mean")
LINKS = config.all_link_ids()
CONC_MIN = 0.5          # 判据①
GAP_MIN = 0.10          # 判据②，单位 m
OUT = config.ARTIFACTS_DIR / "d1" / "final" / "link_drop_check.json"


def _encode(net, X, batch=512):
    """[N,L,C,T,S] -> 编码特征 [N,L,D]（与 ConcatMLP.forward 的前半段相同）。"""
    import torch
    dev = next(net.parameters()).device
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.tensor(X[s:s + batch], dtype=torch.float32, device=dev)
            B, L, C, Tt, S = xb.shape
            out.append(net.encoder(xb.reshape(B * L, C, Tt, S)).reshape(B, L, -1).cpu().numpy())
    return np.concatenate(out)


def _head(net, F, batch=2048):
    """编码特征 [N,L,D] -> logits（与 ConcatMLP.forward 的后半段相同）。"""
    import torch
    dev = next(net.parameters()).device
    out = []
    with torch.no_grad():
        for s in range(0, len(F), batch):
            fb = torch.tensor(F[s:s + batch], dtype=torch.float32, device=dev)
            out.append(net.head(fb.reshape(len(fb), -1)).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    grid = sessions.session_grid("d1")
    data, _ = T.load("final", "ampphase")
    Xtr = data["train"][0]
    Xte = data["test"][0]
    conds = ["all"] + [f"{m}:{l}" for m in MODES for l in LINKS]

    # res[model][cond][round] = 每个种子的 {mae, acc, far, p90}
    res: dict = {m: {c: {} for c in conds} for m in MODELS}
    for seed in SEEDS:
        d = config.ARTIFACTS_DIR / "d1" / "final" / T.tag_of("cls", "ampphase", seed)
        ref = json.loads((d / "results.json").read_text(encoding="utf-8"))["models"]
        for name in MODELS:
            model, _ = T.make_model(name, seed, grid, "cls", Xtr.shape[2])
            model.load(d / name)
            net = model.net.eval()
            mean_f = _encode(net, Xtr).mean(axis=0)            # [L,D]，只用训练集
            F_te = _encode(net, Xte)
            for c in conds:
                F = F_te.copy()
                if c != "all":
                    mode, lid = c.split(":")
                    l = LINKS.index(lid)
                    F[:, l] = 0.0 if mode == "zero" else mean_f[l]
                lab = _head(net, F).argmax(1)
                pred = np.stack([lab % grid.cols, lab // grid.cols], axis=1).astype(np.float32)
                for r, m in T.by_round(pred, data["test"], grid).items():
                    res[name][c].setdefault(r, []).append(dict(
                        mae=m["mean_error_m"], acc=m["accuracy"], far=m["far_rate"], p90=m["p90_error_m"]))
                    if c == "all":                               # 与主表核对
                        want = ref[name]["by_round"][r]["mean_error_m"]
                        if abs(m["mean_error_m"] - want) > 0.005:
                            print(f"!! 种子 {seed} {name} {r}：四条都在 {m['mean_error_m']:.3f} m，"
                                  f"与 results.json 的 {want:.3f} m 不一致（>0.005）——先停下来查原因")
        print(f"种子 {seed} 完成")

    avg = lambda xs, k: float(np.mean([x[k] for x in xs]))
    rounds = sorted(res[MODELS[0]]["all"], key=lambda s: int(s[1:]))
    summary = {"seeds": list(SEEDS), "rounds": {}}
    for r in rounds:
        print(f"\n===== {r}  3 个种子平均；单位 m；括号内为准确率")
        print(f"  {'模型':15s} {'换法':5s} {'四条都在':>14s} " + " ".join(f"{'去 ' + l:>16s}" for l in LINKS)
              + f" {'集中度':>6s}")
        summary["rounds"][r] = {}
        for name in MODELS:
            base = avg(res[name]["all"][r], "mae")
            base_acc = avg(res[name]["all"][r], "acc")
            summary["rounds"][r][name] = {"all": {"mae": base, "acc": base_acc}}
            for mode in MODES:
                mae = {l: avg(res[name][f"{mode}:{l}"][r], "mae") for l in LINKS}
                acc = {l: avg(res[name][f"{mode}:{l}"][r], "acc") for l in LINKS}
                delta = {l: mae[l] - base for l in LINKS}
                pos = sum(max(v, 0.0) for v in delta.values())
                worst = max(delta, key=delta.get)
                conc = delta[worst] / pos if pos > 0 else float("nan")
                summary["rounds"][r][name][mode] = dict(mae=mae, acc=acc, delta=delta, worst=worst,
                                                        concentration=conc)
                cells = " ".join(f"{mae[l]:7.3f}({acc[l]:4.0%})  " for l in LINKS)
                print(f"  {name:15s} {mode:5s} {base:7.3f}({base_acc:4.0%})  {cells} {conc:6.2f}")
                print(f"  {'':15s} {'Δ':5s} {'':14s} " + " ".join(f"{delta[l]:+16.3f}" for l in LINKS))

    print("\n===== 判据（跑之前写下，见文件开头）")
    ok_all = True
    for r in rounds:
        for mode in MODES:
            s = summary["rounds"][r]
            a, b = s["ConcatMLP"][mode], s["ConcatMLP-mask"][mode]
            w = a["worst"]
            c1 = a["concentration"] >= CONC_MIN
            gap = a["delta"][w] - b["delta"][w]
            c2 = gap > GAP_MIN
            ok_all &= c1 and c2
            print(f"  {r} {mode:4s}：ConcatMLP 影响最大的链路 = {w}，集中度 {a['concentration']:.2f} "
                  f"{'✓' if c1 else '✗'}（≥ {CONC_MIN}）；该链路 Δ：ConcatMLP {a['delta'][w]:+.3f} vs "
                  f"mask {b['delta'][w]:+.3f}，差 {gap:+.3f} {'✓' if c2 else '✗'}（> {GAP_MIN}）")
    verdict = "成立" if ok_all else "不完全成立（按实际情况描述，见上面各行）"
    print(f"  结论：\"ConcatMLP 过度依赖某一条链路\" {verdict}")
    summary["verdict"] = verdict
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
