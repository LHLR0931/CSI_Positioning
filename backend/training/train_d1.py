"""D1（2026-10-08，5×3 / 0.8 m）训练与评估 —— 2026-10-08 与用户商定的协议。

为什么分两个阶段
----------------
D1 当天 piB 开机 3 次（09:15 / 11:04 / 13:06），重启会让 piB 两条链路的指纹整体改变。
已定的方案 A：每次重启后先采一轮带标签的标定轮，并把它并入训练集，再测之后的轮次。
今天的数据里正好有两段结构相同的"重启 → 标定 → 20 分钟后 → 40 分钟后"：

    选模型阶段（select）：piB 11:04 重启，r5 = 标定轮，r6 / r7 = 之后 20 / 40 分钟
        训练 r1–r5，r6 只用于早停，在 r7 上比较各模型；r8–r10 整体剔除、完全不可见。
    最终测试阶段（final）：piB 13:06 重启，r8 = 标定轮，r9 / r10 = 之后 20 / 40 分钟
        训练 r1–r8，没有验证集可早停，训练轮数固定为选模型阶段该模型的最佳轮数；
        在 r9 / r10 上**只评估一次**，按轮次分别报成绩。

注意：选模型阶段的成绩只用来比较模型之间的好坏，不能当作最终精度报告
（r7 前面的 r6 就在训练集附近，比测试集"好考"）。D1 的测试只有 2 轮，单靠 D1
做不了显著性检验，正式结论等 D+3 / D+5。

模型（2026-10-12 按导师要求调整：阶段一 = 单链路 KNN / CNN；阶段二 = 四链路 KNN / 样本图 GCN / MLP）
----
    Center          永远猜网格正中间（地板，不训练）
    KNN-<链路> ×4   单链路 KNN（k=5，距离加权）。特征 = 窗口 16 帧均值（2×55=110 维），训练集标准化
    CNN-<链路> ×4   单链路 CNN
    KNN             四链路 KNN：同上，特征 4×2×55=440 维
    GCN             样本图 GCN：结构移植自前人 RSSI+GNN 论文（见 backend/models/sample_gcn.py），
                    与四链路 KNN 用完全相同的特征和 k，差别只在图上的信息传递
    ConcatMLP       四链路编码特征拼接
    ConcatMLP-mask  ConcatMLP + 训练时随机遮挡链路（p=0.25）—— 跨日预登记的主模型
    ConcatMLP-sep   （诊断，2026-10-14 E1，只能 --model 单独指定）ConcatMLP 但 4 条链路的编码器不共享权重；
                    判据与汇总见 backend/training/e1_shared_encoder.py
  Linear（逻辑回归）不再默认运行（用户 2026-10-12 决定论文不报）；类本身保留，跨日预登记 S7 要用。

可切换的设置（命令行），每组设置的结果分开保存，互不覆盖
--------------------------------------------------------
    --task reg|cls          神经网络输出：reg = 回归坐标 (x,y)；cls = 15 类分类。
                            2026-10-08 r7 初测：分类平均误差/准确率更好，但 P90 与 >1.6 m
                            比例更差，尚未定，需多种子比较后由用户决定。
    --channels amp|ampphase 输入通道：只用幅度 / 幅度+相位。所有模型（含 Linear）共用。
    --seed 42 1 2           可一次给多个种子，依次跑。

固定的设置
----------
    FREQ_BINS=16   编码器保留子载波轴（旧结构全局平均会抹掉"哪个子载波凹陷"）。
    早停看验证集平均定位误差（不看损失）：分类时交叉熵会因过度自信早早上升，
                   旧判据下分类模型在第 2–4 轮就被停掉。

用法（项目根目录执行，需先 python -X utf8 -m backend.data.load_real_stage2 d1）：
    python -X utf8 -m backend.training.train_d1 --phase select --task cls --seed 42 1 2
    python -X utf8 -m backend.training.train_d1 --phase select --task reg --seed 42 1 2
    python -X utf8 -m backend.training.train_d1 --phase select --summary     # 汇总比较表
    python -X utf8 -m backend.training.train_d1 --phase final --task cls --seed 42   # 只跑一次

输出：backend/artifacts/d1/<phase>/<task>_<channels>_seed<种子>/results.json 及模型文件。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from backend import config
from backend.data import sessions
from backend.models.fusion_gat import _HAS_TORCH
from backend.models.metrics import accuracy, evaluate_xy, snap_to_label, per_point_accuracy
from backend.training.prepare import load_stage2_windows

SESSION = "d1"
PHASES = {
    "select": dict(val=["r6"], test=["r7"], exclude=["r8", "r9", "r10"]),
    "final": dict(val=[], test=["r9", "r10"], exclude=[]),
    # 开发阶段（2026-10-08 加，只用来开发健康分，不报模型精度）：piB 11:04 重启后的 r5–r6 训练、
    # r7 早停，在 piB 第一次开机的 r1–r3 上测。piA 全程同一次开机，于是 piB 两条链路跨了一次
    # 重启（指纹整体变了）而 piA 没有——一个答案已知的天然差模退化。r4 中途有 piB 重启，剔除。
    "dev": dict(val=["r7"], test=["r1", "r2", "r3"], exclude=["r4", "r8", "r9", "r10"]),
}
LINKS = config.all_link_ids()
MODEL_NAMES = (["Center"] + [f"KNN-{l}" for l in LINKS] + [f"CNN-{l}" for l in LINKS]
               + ["KNN", "GCN", "ConcatMLP", "ConcatMLP-mask"])
# 诊断模型：不在 --model all 里跑，只能用 --model 单独指定（2026-10-14 E1）
EXTRA_MODELS = ["ConcatMLP-sep"]
KNN_K = 5          # 单链路 / 四链路 KNN 与 GCN 建图共用（前人 Sun 2026 的 KNN 基线也是 5）
FAR_M = 1.6        # 口径纪律：误差 > 1.6 m（两个格距）的比例
FREQ_BINS = 16     # 编码器保留子载波轴，分 16 段（见 fusion_gat.CNNEncoder 文档）
CHANNEL_SETS = {"amp": (0,), "ampphase": (0, 1)}   # 0 = 幅度，1 = 相位
# 预处理版本，写进结果目录名，新旧结果分开保存、可并排比较：
#   pp1 = 2026-10-08 之前（子载波按原始 FFT 编号顺序，含垃圾编号 32/34/35，D1 正式结果用的就是它）
#   pp2 = 2026-10-08 晚修正（只用 55 个有效子载波，按物理频率排序；相位去趋势用物理子载波号）
PREPROC = "pp2"


def tag_of(task: str, channels: str, seed: int) -> str:
    return f"{task}_{channels}_{PREPROC}_seed{seed}"


def out_dir(phase: str, tag: str) -> Path:
    d = config.ARTIFACTS_DIR / "d1" / phase / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(phase: str, channels: str):
    p = PHASES[phase]
    data, pp = load_stage2_windows(
        T=config.STAGE2_TIME_WINDOW, mode="by_batch", session=SESSION,
        val_batches=p["val"] or None, test_batches=p["test"],
        exclude_batches=p["exclude"], subcarrier_mode="train_fold", return_batches=True)
    ch = list(CHANNEL_SETS[channels])
    for k in data:                       # 所有模型共用同一组输入通道
        data[k] = (data[k][0][:, :, ch],) + tuple(data[k][1:])
    return data, pp


def metrics_for(pred_xy, true_xy, true_label, grid) -> dict:
    err = np.linalg.norm(np.asarray(pred_xy) - np.asarray(true_xy), axis=1) * grid.spacing_m
    m = evaluate_xy(pred_xy, true_xy, grid=grid)
    m.pop("median_error_m", None)          # 口径纪律：不报中位数
    lab = snap_to_label(pred_xy, grid)
    m["accuracy"] = accuracy(true_label, lab)
    m["far_rate"] = float(np.mean(err > FAR_M))
    m["per_point_accuracy"] = per_point_accuracy(true_label, lab, grid=grid)
    return m


def by_round(pred_xy, data_test, grid) -> dict:
    _, yte, lte, bte = data_test
    return {r: metrics_for(pred_xy[bte == r], yte[bte == r], lte[bte == r], grid)
            for r in sorted(set(bte.tolist()), key=lambda s: int(s[1:]))}


class LinearModel:
    """线性基线：每条链路、每个通道、每个子载波在 16 帧内取均值 -> 标准化 -> 15 类逻辑回归。

    与 CNN 用同一批窗口、同一预处理、同一组输入通道，唯一差别是分类器本身。
    predict 返回概率最大格点的 (x,y)，与神经网络的输出格式一致。
    """

    def __init__(self, grid_cols: int, C: float = 0.1):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.grid_cols = grid_cols
        self.scaler, self.clf = StandardScaler(), LogisticRegression(max_iter=3000, C=C)

    @staticmethod
    def features(X) -> np.ndarray:            # X:[N,L,C,T,S] -> [N, L*C*S]
        return np.asarray(X).mean(axis=3).reshape(len(X), -1)

    def fit(self, X, labels):
        self.clf.fit(self.scaler.fit_transform(self.features(X)), labels)
        return self

    def predict(self, X) -> np.ndarray:
        lab = self.clf.predict(self.scaler.transform(self.features(X)))
        return np.stack([lab % self.grid_cols, lab // self.grid_cols], axis=1).astype(np.float32)

    def save(self, path):
        np.savez(f"{path}.npz", scaler_mean=self.scaler.mean_, scaler_scale=self.scaler.scale_,
                 coef=self.clf.coef_, intercept=self.clf.intercept_, classes=self.clf.classes_)


def make_model(name: str, seed: int, grid, task: str, n_channels: int):
    if name == "GCN":
        if task != "cls":
            raise SystemExit("GCN 只实现了分类（--task cls）")
        from backend.models.sample_gcn import SampleGCNModel
        return SampleGCNModel(n_classes=grid.num_classes, grid_cols=grid.cols, seed=seed, k=KNN_K), None
    from backend.models.fusion_baseline import (ConcatMLPMaskModel, ConcatMLPModel, ConcatMLPSepModel,
                                                SingleLinkCNNModel)
    kw = dict(lr=1e-3, seed=seed, freq_bins=FREQ_BINS, in_ch=n_channels,
              task=task, n_classes=grid.num_classes, grid_cols=grid.cols)
    if name == "ConcatMLP":
        return ConcatMLPModel(**kw), None
    if name == "ConcatMLP-sep":
        return ConcatMLPSepModel(**kw), None
    if name == "ConcatMLP-mask":
        return ConcatMLPMaskModel(**kw), None
    if name.startswith("CNN-"):
        return SingleLinkCNNModel(**kw), LINKS.index(name[4:])
    raise ValueError(name)


def run(phase: str, models: list[str], task: str, channels: str, seed: int,
        epochs: int = 150, patience: int = 20, data_pp=None) -> dict:
    grid = sessions.session_grid(SESSION)
    tag = tag_of(task, channels, seed)
    data, pp = data_pp if data_pp is not None else load(phase, channels)
    Xtr, ytr, ltr, btr = data["train"]
    Xte = data["test"][0]
    target = (lambda part: part[2]) if task == "cls" else (lambda part: part[1])
    rounds = lambda b: sorted(set(b.tolist()), key=lambda s: int(s[1:]))
    print(f"==== 阶段={phase}  设置={tag}  （神经网络={'15 类分类' if task == 'cls' else '坐标回归'}，"
          f"输入={'幅度' if channels == 'amp' else '幅度+相位'}）")
    print(f"训练轮次={rounds(btr)}  验证={PHASES[phase]['val']}  测试={PHASES[phase]['test']}  "
          f"剔除={PHASES[phase]['exclude']}")
    print(f"窗口数: train={len(ytr)} val={len(data['val'][1])} test={len(data['test'][1])}  "
          f"X 形状 {Xtr.shape} (窗口, 链路, 通道, 时间, 子载波)\n")

    fixed_epochs = {}
    if phase == "final":
        sel = out_dir("select", tag) / "results.json"
        if not sel.exists():
            raise SystemExit(f"先用同样的设置跑 --phase select（找不到 {sel}）：最终阶段的训练轮数取自选模型阶段。")
        sel_res = json.loads(sel.read_text(encoding="utf-8"))["models"]
        fixed_epochs = {k: v["best_epoch"] + 1 for k, v in sel_res.items() if "best_epoch" in v}

    res_path = out_dir(phase, tag) / "results.json"
    results = json.loads(res_path.read_text(encoding="utf-8"))["models"] if res_path.exists() else {}
    for name in models:
        t0 = time.time()
        if name == "Center":
            center = np.array([(grid.cols - 1) / 2, (grid.rows - 1) / 2], np.float32)
            row = {"by_round": by_round(np.tile(center, (len(Xte), 1)), data["test"], grid)}
        elif name == "Linear":
            model = LinearModel(grid.cols).fit(Xtr, ltr)
            row = {"by_round": by_round(model.predict(Xte), data["test"], grid)}
            model.save(out_dir(phase, tag) / name)
        elif name.startswith("KNN"):
            # 不训练、没有轮数；单链路切出一条，四链路用全部。特征与 GCN 相同（window_features）
            from backend.models.knn import KNNModel
            from backend.models.sample_gcn import window_features
            link = LINKS.index(name[4:]) if name != "KNN" else None
            sl = (lambda X: X[:, link:link + 1]) if link is not None else (lambda X: X)
            model = KNNModel(k=KNN_K, standardize=True).fit(window_features(sl(Xtr)), ltr)
            lab = model.predict(window_features(sl(Xte)))
            pred = np.stack([lab % grid.cols, lab // grid.cols], axis=1).astype(np.float32)
            row = {"by_round": by_round(pred, data["test"], grid)}
            model.save(out_dir(phase, tag) / f"{name}.joblib")
        else:
            model, link = make_model(name, seed, grid, task, Xtr.shape[2])
            sl = (lambda X: X[:, link:link + 1]) if link is not None else (lambda X: X)
            if phase != "final":
                ep_max, pat = (None, None) if name == "GCN" else (epochs, patience)   # GCN 用前人的 300 轮 / 15
                model.fit(sl(Xtr), target(data["train"]), epochs=ep_max, patience=pat,
                          X_val=sl(data["val"][0]), y_val=target(data["val"]), early_stop_on="error")
                extra = {"best_epoch": model.best_epoch,
                         "best_val_error_m": model.best_val_loss * grid.spacing_m}
            else:
                if name not in fixed_epochs:
                    raise SystemExit(f"选模型阶段没有 {name} 的结果，先跑 --phase select --model {name}")
                model.fit(sl(Xtr), target(data["train"]), epochs=fixed_epochs[name])
                extra = {"epochs_fixed": fixed_epochs[name]}
            row = {"by_round": by_round(model.predict(sl(Xte)), data["test"], grid), **extra}
            model.save(out_dir(phase, tag) / name)
        note = f"  最佳轮数={row['best_epoch']}" if "best_epoch" in row else ""
        row["seconds"] = round(time.time() - t0, 1)
        results[name] = row
        line = "  ".join(f"{r}: 误差 {m['mean_error_m']:.3f} m | P90 {m['p90_error_m']:.2f} m | "
                         f"准确率 {m['accuracy']:.0%} | >1.6m {m['far_rate']:.0%}"
                         for r, m in row["by_round"].items())
        print(f"{name:16s} {line}   ({row['seconds']:.0f}s){note}")

    pp.save(out_dir(phase, tag) / "preprocessor.json")
    res_path.write_text(json.dumps({"session": SESSION, "phase": phase, "tag": tag, "task": task,
                                    "preproc": PREPROC,
                                    "channels": channels, "seed": seed, "freq_bins": FREQ_BINS,
                                    "early_stop_on": "error", "protocol": PHASES[phase],
                                    "models": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {res_path}\n")
    return results


def summary(phase: str) -> None:
    """把该阶段所有设置、所有种子的结果汇成一张表：每个 (模型, 设置) 给出多个种子的平均和范围。"""
    root = config.ARTIFACTS_DIR / "d1" / phase
    files = sorted(root.glob("*/results.json"))
    if not files:
        raise SystemExit(f"{root} 下还没有结果")
    rows: dict[tuple, dict] = {}
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        setting = f"{r['task']}_{r['channels']}_{r.get('preproc', 'pp1')}"
        for name, v in r["models"].items():
            for rnd, m in v["by_round"].items():
                if name == "Center":
                    key = (rnd, name, "-")                 # 与任务、通道都无关
                elif name == "Linear":
                    key = (rnd, name, f"{r['channels']}_{r.get('preproc', 'pp1')}")   # 与 --task 无关
                else:
                    key = (rnd, name, setting)
                d = rows.setdefault(key, {"err": [], "p90": [], "acc": [], "far": [], "seeds": set()})
                if r["seed"] in d["seeds"]:
                    continue                     # Center / Linear 与种子无关，同一种子不重复计
                d["seeds"].add(r["seed"])
                d["err"].append(m["mean_error_m"]); d["p90"].append(m["p90_error_m"])
                d["acc"].append(m["accuracy"]); d["far"].append(m["far_rate"])
    order = {n: i for i, n in enumerate(MODEL_NAMES + ["Linear"] + EXTRA_MODELS)}
    fmt = lambda xs, f: (f.format(np.mean(xs)) + (f" ({f.format(min(xs))}–{f.format(max(xs))})"
                                                  if len(xs) > 1 else ""))
    for rnd in sorted({k[0] for k in rows}, key=lambda s: int(s[1:])):
        print(f"\n===== {phase} 阶段  测试轮 {rnd}  （多个种子时：平均（最小–最大））")
        print(f"{'模型':16s} {'设置':14s} {'种子数':>4s}  {'平均误差 m':22s} {'P90 m':20s} {'准确率':18s} {'>1.6m':16s}")
        for key in sorted((k for k in rows if k[0] == rnd), key=lambda k: (order.get(k[1], 99), k[2])):
            d = rows[key]
            print(f"{key[1]:16s} {key[2]:14s} {len(d['seeds']):>4d}  {fmt(d['err'], '{:.3f}'):22s} "
                  f"{fmt(d['p90'], '{:.2f}'):20s} {fmt([a * 100 for a in d['acc']], '{:.0f}'):18s} "
                  f"{fmt([a * 100 for a in d['far']], '{:.0f}'):16s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="D1 训练与评估（选模型 / 最终测试）")
    ap.add_argument("--phase", required=True, choices=list(PHASES))
    ap.add_argument("--summary", action="store_true", help="只汇总已有结果，不训练")
    ap.add_argument("--task", choices=["reg", "cls"], help="神经网络输出：reg 回归 / cls 分类（训练时必填）")
    ap.add_argument("--channels", default="ampphase", choices=list(CHANNEL_SETS))
    ap.add_argument("--model", default="all", choices=["all"] + MODEL_NAMES + ["Linear"] + EXTRA_MODELS)
    ap.add_argument("--seed", type=int, nargs="+", default=[config.RANDOM_SEED])
    args = ap.parse_args()
    if args.summary:
        summary(args.phase)
        raise SystemExit(0)
    if args.task is None:
        raise SystemExit("训练时必须写明 --task reg 或 --task cls")
    if not _HAS_TORCH and not (args.model in ("Center", "Linear") or args.model.startswith("KNN")):
        raise SystemExit("没有安装 PyTorch。先 pip install torch。")
    names = MODEL_NAMES if args.model == "all" else [args.model]
    shared = load(args.phase, args.channels)          # 数据只加载一次，多个种子共用
    for s in args.seed:
        run(args.phase, names, args.task, args.channels, s, data_pp=shared)
