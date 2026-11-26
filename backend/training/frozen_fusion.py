"""阶段三补充实验：KNN / 样本图 GCN / MLP 吃同一份 CNN 特征，只比"融合方式"（2026-10-14，跑之前写下判据）

为什么要做
----------
D1 主表里，四链路 KNN、样本图 GCN 的输入是"16 帧时间平均后的原始幅度+相位"（4×2×55 = 440 维），
不经过 CNN；ConcatMLP 则是每条链路先过 CNN 编码器再拼接。所以"MLP 比 KNN / GCN 好"可能来自
编码器，也可能来自融合方式，两者分不开。而论文的叙事是"阶段二选定 CNN 作编码器，阶段三比较
融合方式"——要让这句话成立，三种融合方式必须吃同一份 CNN 特征。

做法
----
编码器 = 阶段二已经训练好的 4 个单链路 CNN（CNN-<链路>，与 D1 主表同一批模型文件，冻结不再训练）。
每条链路的 X 过各自的编码器得到 64 维特征，四条拼接成 256 维。三种融合方式都只看这 256 维：

    F-KNN   KNN（k=5，距离加权，训练集标准化）。与主表 KNN 同一个类、同一组参数。
    F-GCN   样本图 GCN。与主表 GCN 同一个类、同一组参数（2 层 128→64、k=5、300 轮 / 早停 15）。
    F-MLP   一个 MLP 头：Linear(256,64) → ReLU → Linear(64,15)，结构与 ConcatMLP 的头完全相同；
            训练设置与 ConcatMLP 相同（Adam 1e-3、batch 32、最多 150 轮 / 早停 20、看验证集定位误差）。
            输入为训练集标准化后的特征（KNN / GCN 内部也做同样的标准化，三者输入一致）。

"F-" 表示 frozen（编码器冻结）。同一个种子下三种融合方式用的是同一组编码器，可以逐种子配对比较。
协议与 train_d1 完全相同：select（训练 r1–r5、r6 早停、看 r7）→ final（训练 r1–r8，轮数取 select 的
最佳轮数 + 1，在 r9 / r10 上只评估一次）。指标函数也是 train_d1.metrics_for。
脚本会先用加载的单链路 CNN 重算一遍它自己的成绩，与 results.json 核对，对不上就停。

已知局限（论文要写）：编码器是在训练轮上训练的，训练集特征会比测试集"更干净"，这对三种融合方式
是一样的；冻结编码器的 F-MLP 与端到端训练的 ConcatMLP 不是同一个模型，二者差别 = 联合训练的作用。

跑之前写下的判据（r9 / r10 在之前的实验里已经看过，所以这是诊断分析，不是预登记检验）
--------------------------------------------------------------------------------
Δ = F-MLP 平均误差 − min(F-KNN, F-GCN) 平均误差（同一种子配对，再对 3 个种子取平均）

  问题 A："在同一份 CNN 特征上，MLP 融合方式本身更好"成立，需同时满足：
      ① r10：3 种子平均 Δ ≤ −0.10 m，且 3 个种子的 Δ 都 < 0；
      ② r9 ：3 种子平均 Δ ≤ +0.05 m（标定后紧接着，不比 KNN/GCN 中较好的那个差太多）。
     不成立 → 论文写"给定 CNN 特征后，融合方式差别不大，主表里 MLP 的优势主要来自 CNN 编码器"。

  问题 B："CNN 特征本身让 KNN / GCN 更抗漂移"成立，需：
      r10 上 F-KNN 比主表 KNN、F-GCN 比主表 GCN 都至少好 0.10 m（3 种子平均）。

  问题 C（只报告，不设判据）：端到端 ConcatMLP vs F-MLP，看联合训练编码器是否有额外好处。

用法（项目根目录，先 select 再 final；3 个种子在本机 GPU 上合计几分钟）：
    python -X utf8 -m backend.training.frozen_fusion --phase select
    python -X utf8 -m backend.training.frozen_fusion --phase final
    python -X utf8 -m backend.training.frozen_fusion --summary
输出：backend/artifacts/d1/<phase>/cls_ampphase_pp2_seed<N>/frozen_fusion.json，
      backend/artifacts/d1/final/frozen_fusion_summary.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from backend import config
from backend.data import sessions
from backend.training import train_d1 as T

SEEDS = (42, 1, 2)
TASK, CHANNELS = "cls", "ampphase"
LINKS = config.all_link_ids()
MODELS = ("F-KNN", "F-GCN", "F-MLP")
A_GAP_R10, A_TOL_R9, B_GAIN = 0.10, 0.05, 0.10          # 判据阈值，单位 m（见文件开头）
CHECK_TOL = 0.005                                        # 单链路 CNN 复算与 results.json 的容差，m


def _dir(phase: str, seed: int):
    return config.ARTIFACTS_DIR / "d1" / phase / T.tag_of(TASK, CHANNELS, seed)


def load_encoders(phase: str, seed: int, grid, n_ch: int, data) -> list:
    """加载阶段二的 4 个单链路 CNN，并核对它们在测试轮上的成绩与 results.json 一致。"""
    d = _dir(phase, seed)
    ref = json.loads((d / "results.json").read_text(encoding="utf-8"))["models"]
    models = []
    for l, lid in enumerate(LINKS):
        name = f"CNN-{lid}"
        model, _ = T.make_model(name, seed, grid, TASK, n_ch)
        model.load(d / name)
        got = T.by_round(model.predict(data["test"][0][:, l:l + 1]), data["test"], grid)
        for r, m in got.items():
            want = ref[name]["by_round"][r]["mean_error_m"]
            if abs(m["mean_error_m"] - want) > CHECK_TOL:
                raise SystemExit(f"!! 种子 {seed} {name} {r}：复算 {m['mean_error_m']:.3f} m，"
                                 f"results.json 为 {want:.3f} m —— 编码器与主表不是同一批，先停下来查原因")
        models.append(model)
    return models


def encode(models, X, batch: int = 512) -> np.ndarray:
    """[N,L,C,T,S] -> [N, L*64]：每条链路过自己的编码器（冻结），再拼接。"""
    import torch
    feats = []
    for l, model in enumerate(models):
        enc = model.net.encoder.eval()
        dev = next(enc.parameters()).device
        out = []
        with torch.no_grad():
            for s in range(0, len(X), batch):
                xb = torch.tensor(np.asarray(X[s:s + batch, l]), dtype=torch.float32, device=dev)
                out.append(enc(xb).cpu().numpy())
        feats.append(np.concatenate(out))
    return np.concatenate(feats, axis=1).astype(np.float32)


def as_window(F: np.ndarray) -> np.ndarray:
    """[N,D] -> [N,1,1,1,D]。sample_gcn.window_features 对第 3 轴（时间）取平均再展平，
    这里时间轴长度为 1，平均后原样还原成 [N,D]，于是 GCN 类不用改一行就能吃 CNN 特征。"""
    return F[:, None, None, None, :]


def make_mlp(seed: int, in_dim: int, grid):
    import torch
    from torch import nn
    from backend.models.fusion_gat import _TorchReg
    torch.manual_seed(seed)          # 与 ConcatMLPModel 相同：构造网络前设种子
    head = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, grid.num_classes))
    return _TorchReg(head, lr=1e-3, seed=seed, task="cls", grid_cols=grid.cols)


def run(phase: str, seed: int, data, grid) -> dict:
    n_ch = data["train"][0].shape[2]
    t0 = time.time()
    encs = load_encoders(phase, seed, grid, n_ch, data)
    F = {k: encode(encs, data[k][0]) for k in ("train", "val", "test") if len(data[k][1])}
    ltr = data["train"][2]
    print(f"\n==== 阶段={phase}  种子={seed}  特征 {F['train'].shape}（4 条链路 × 64 维，编码器冻结）  "
          f"单链路 CNN 复算与 results.json 一致 ✓  ({time.time() - t0:.0f}s)")

    fixed = {}
    if phase == "final":
        sel = _dir("select", seed) / "frozen_fusion.json"
        if not sel.exists():
            raise SystemExit(f"先跑 --phase select（找不到 {sel}）：final 的训练轮数取自 select。")
        fixed = {k: v["best_epoch"] + 1 for k, v in
                 json.loads(sel.read_text(encoding="utf-8"))["models"].items() if "best_epoch" in v}

    mu, sd = F["train"].mean(0), F["train"].std(0) + 1e-6
    S = {k: (v - mu) / sd for k, v in F.items()}
    has_val = "val" in F
    results = {}
    for name in MODELS:
        t0 = time.time()
        extra = {}
        if name == "F-KNN":
            from backend.models.knn import KNNModel
            lab = KNNModel(k=T.KNN_K, standardize=True).fit(F["train"], ltr).predict(F["test"])
            pred = np.stack([lab % grid.cols, lab // grid.cols], 1).astype(np.float32)
        else:
            if name == "F-GCN":
                from backend.models.sample_gcn import SampleGCNModel
                model = SampleGCNModel(n_classes=grid.num_classes, grid_cols=grid.cols, seed=seed, k=T.KNN_K)
                Xtr, Xte = as_window(F["train"]), as_window(F["test"])
                Xva = as_window(F["val"]) if has_val else None
                ep, pat = None, None                 # GCN 用前人的 300 轮 / 早停 15（与主表相同）
            else:
                model = make_mlp(seed, S["train"].shape[1], grid)
                Xtr, Xte = S["train"], S["test"]
                Xva = S["val"] if has_val else None
                ep, pat = 150, 20                    # 与 train_d1 中 ConcatMLP 相同
            if phase != "final":
                model.fit(Xtr, ltr, epochs=ep, patience=pat, X_val=Xva, y_val=data["val"][2],
                          early_stop_on="error")
                extra = {"best_epoch": model.best_epoch,
                         "best_val_error_m": model.best_val_loss * grid.spacing_m}
            else:
                if name not in fixed:
                    raise SystemExit(f"select 结果里没有 {name}，先重跑 --phase select")
                model.fit(Xtr, ltr, epochs=fixed[name])
                extra = {"epochs_fixed": fixed[name]}
            pred = model.predict(Xte)
        row = {"by_round": T.by_round(pred, data["test"], grid), **extra,
               "seconds": round(time.time() - t0, 1)}
        results[name] = row
        line = "  ".join(f"{r}: 误差 {m['mean_error_m']:.3f} m | P90 {m['p90_error_m']:.2f} | "
                         f"准确率 {m['accuracy']:.0%} | >1.6m {m['far_rate']:.0%}"
                         for r, m in row["by_round"].items())
        note = f"  最佳轮数={row['best_epoch']}" if "best_epoch" in row else ""
        print(f"{name:6s} {line}   ({row['seconds']:.0f}s){note}")

    out = _dir(phase, seed) / "frozen_fusion.json"
    out.write_text(json.dumps({"session": "d1", "phase": phase, "seed": seed, "preproc": T.PREPROC,
                               "task": TASK, "channels": CHANNELS, "feature_dim": int(F["train"].shape[1]),
                               "encoder": "阶段二单链路 CNN-<链路>（同目录模型文件，冻结）",
                               "protocol": T.PHASES[phase], "models": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {out}")
    return results


def summary() -> None:
    """final 阶段 3 个种子汇总 + 判据。参考行（主表 KNN / GCN / ConcatMLP）读同目录 results.json。"""
    ff, ref = {}, {}
    for s in SEEDS:
        d = _dir("final", s)
        if not (d / "frozen_fusion.json").exists():
            raise SystemExit(f"种子 {s} 还没跑 final：{d / 'frozen_fusion.json'}")
        ff[s] = json.loads((d / "frozen_fusion.json").read_text(encoding="utf-8"))["models"]
        ref[s] = json.loads((d / "results.json").read_text(encoding="utf-8"))["models"]
    get = lambda src, name, r, k="mean_error_m": [src[s][name]["by_round"][r][k] for s in SEEDS]
    rounds = sorted(ff[SEEDS[0]]["F-KNN"]["by_round"], key=lambda x: int(x[1:]))
    rows = [("单链路 CNN 平均", None), ("KNN（主表，原始特征）", "KNN"), ("GCN（主表，原始特征）", "GCN"),
            ("F-KNN", "F-KNN"), ("F-GCN", "F-GCN"), ("F-MLP", "F-MLP"), ("ConcatMLP（端到端）", "ConcatMLP")]
    fmt = lambda xs, f: f"{f.format(np.mean(xs))} ({f.format(min(xs))}–{f.format(max(xs))})"
    out = {"seeds": list(SEEDS), "rounds": {}}
    for r in rounds:
        print(f"\n===== final  测试轮 {r}  3 个种子：平均（最小–最大）")
        print(f"  {'模型':22s} {'平均误差 m':22s} {'P90 m':20s} {'准确率 %':16s} {'>1.6m %':14s}")
        out["rounds"][r] = {}
        for label, name in rows:
            if name is None:
                err = [np.mean([ref[s][f"CNN-{l}"]["by_round"][r]["mean_error_m"] for l in LINKS]) for s in SEEDS]
                print(f"  {label:22s} {fmt(err, '{:.3f}'):22s}")
                out["rounds"][r][label] = {"mae": float(np.mean(err))}
                continue
            src = ff if name.startswith("F-") else ref
            err, p90 = get(src, name, r), get(src, name, r, "p90_error_m")
            acc = [a * 100 for a in get(src, name, r, "accuracy")]
            far = [a * 100 for a in get(src, name, r, "far_rate")]
            print(f"  {label:22s} {fmt(err, '{:.3f}'):22s} {fmt(p90, '{:.2f}'):20s} "
                  f"{fmt(acc, '{:.0f}'):16s} {fmt(far, '{:.0f}'):14s}")
            out["rounds"][r][name] = {"mae": float(np.mean(err)), "mae_per_seed": err,
                                      "p90": float(np.mean(p90)), "acc": float(np.mean(acc)) / 100,
                                      "far": float(np.mean(far)) / 100}

    print("\n===== 判据（跑之前写下，见文件开头）")
    delta = {r: [ff[s]["F-MLP"]["by_round"][r]["mean_error_m"]
                 - min(ff[s]["F-KNN"]["by_round"][r]["mean_error_m"], ff[s]["F-GCN"]["by_round"][r]["mean_error_m"])
                 for s in SEEDS] for r in rounds}
    for r in rounds:
        print(f"  {r}：Δ = F-MLP − min(F-KNN, F-GCN)，逐种子 " + " / ".join(f"{d:+.3f}" for d in delta[r])
              + f"，平均 {np.mean(delta[r]):+.3f} m")
    a1 = np.mean(delta["r10"]) <= -A_GAP_R10 and all(d < 0 for d in delta["r10"])
    a2 = np.mean(delta["r9"]) <= A_TOL_R9
    verdict_a = "成立" if (a1 and a2) else "不成立"
    print(f"  问题 A：① r10 平均 Δ ≤ −{A_GAP_R10} 且 3 种子都 < 0 {'✓' if a1 else '✗'}；"
          f"② r9 平均 Δ ≤ +{A_TOL_R9} {'✓' if a2 else '✗'} → \"同一份 CNN 特征上 MLP 融合更好\" {verdict_a}")
    m = lambda name: out["rounds"]["r10"][name]["mae"]
    gk, gg = m("KNN") - m("F-KNN"), m("GCN") - m("F-GCN")
    verdict_b = "成立" if (gk >= B_GAIN and gg >= B_GAIN) else "不成立"
    print(f"  问题 B：r10 上 CNN 特征带来的改善 KNN {gk:+.3f} m、GCN {gg:+.3f} m（都需 ≥ {B_GAIN}）"
          f" → \"CNN 特征让 KNN/GCN 更抗漂移\" {verdict_b}")
    for r in rounds:
        c = out["rounds"][r]["ConcatMLP"]["mae"] - out["rounds"][r]["F-MLP"]["mae"]
        print(f"  问题 C（只报告）：{r} 端到端 ConcatMLP − F-MLP = {c:+.3f} m"
              f"（负 = 联合训练编码器更好）")
    out.update(delta={r: list(map(float, v)) for r, v in delta.items()},
               verdict_A=verdict_a, verdict_B=verdict_b)
    p = config.ARTIFACTS_DIR / "d1" / "final" / "frozen_fusion_summary.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="KNN / GCN / MLP 共用冻结 CNN 特征的融合方式对比")
    ap.add_argument("--phase", choices=["select", "final"])
    ap.add_argument("--summary", action="store_true", help="只汇总 final 结果并给出判据")
    ap.add_argument("--seed", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    if args.summary:
        summary()
        raise SystemExit(0)
    if args.phase is None:
        raise SystemExit("训练时必须写明 --phase select 或 --phase final")
    grid = sessions.session_grid("d1")
    data, _ = T.load(args.phase, CHANNELS)
    for s in args.seed:
        run(args.phase, s, data, grid)
