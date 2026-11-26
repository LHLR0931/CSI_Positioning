"""跨日验证：健康分门控 vs 等权 —— 严格按预登记 report/健康分预登记.md（2026-10-11 18:05 冻结）实现。

本文件里每个"冻结"的选择都在预登记中有对应条款，注释里标出节号（§）。实现与预登记若有任何
不一致，以预登记为准，并记入预登记的"修订记录"。

修订 1（2026-10-13，解盲前）：导师决定不采 D+5。主验证只用 D+3 的 3 个测试轮，D+3 可单独解盲；
§7 判据改为"3 轮平均 Δ ≤ −0.08 m、3 轮全部 Δ < 0、没有一轮 Δ > +0.08 m"（只收紧、不放宽）。

三个子命令
----------
    train     训练当天模型（只用 D1 + 当天标定轮，不碰测试轮，不输出任何测试成绩，解盲前可以跑）
    evaluate  在测试轮上比较等权与门控（d3 需要 --unblind；§3、修订 1）
    report    汇总 D+3，按 §7（修订 1）判定"明确支持 / 方向性支持 / 不支持"

演练（D1 是开发数据，§3 允许）：把 D1 当成"一天"——标定轮 r8，测试轮 r9 / r10，基础训练集 D1 r1–r7：
    python -X utf8 -m backend.training.crossday train    --day d1
    python -X utf8 -m backend.training.crossday evaluate --day d1

正式（修订 1：只用 D+3）：
    python -X utf8 -m backend.training.crossday train    --day d3
    python -X utf8 -m backend.training.crossday evaluate --day d3 --unblind
    python -X utf8 -m backend.training.crossday report

产物：backend/artifacts/crossday/<day>/（模型、预处理器、results.json）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from backend import config
from backend.data import sessions
from backend.data.load import load_static_raw64
from backend.data.preprocess import Preprocessor, select_subcarriers
from backend.features import link_health as lh

LINKS = config.all_link_ids()
T_WIN = config.STAGE2_TIME_WINDOW                 # 16 帧，不重叠（§4）
BASE_SESSION = "d1"
BASE_ROUNDS = tuple(f"r{i}" for i in range(1, 9))  # D1 r1–r8（§4）
SEEDS = (42, 1, 2)                                 # §4
MASK_EPOCHS = {42: 39, 1: 15, 2: 18}               # §4 表：select pp2 的 best_epoch + 1
FREQ_BINS = 16
CHANNELS = (0, 1)                                  # 幅度 + 相位（ampphase）
FAR_M = 1.6
SUPPORT_M = 0.08                                   # §7：格点间距 0.8 m 的 10%
PREREG = config.PROJECT_ROOT / "report" / "健康分预登记.md"
PREREG_MD5_FROZEN = "72632ae088a8dabc8027d96fed847cdb"  # 冻结版（交接文档 §8g）
PREREG_MD5 = "b944c9af4e1dc05c689518df46bd3b83"         # 冻结版 + 修订 1（2026-10-13）
MAIN_DAYS = ("d3",)                                     # 修订 1：主验证只用 D+3
OUT_ROOT = config.ARTIFACTS_DIR / "crossday"


# ---------------------------------------------------------------------------
# 一天的协议
# ---------------------------------------------------------------------------
def day_protocol(day: str) -> dict:
    """返回 {calib, test, perturb, base_rounds, rehearsal}。D1 是演练：标定 r8、测试 r9/r10、基础集 r1–r7。"""
    if day == BASE_SESSION:
        return dict(calib="r8", test=["r9", "r10"], perturb=[],
                    base_rounds=[r for r in BASE_ROUNDS if r != "r8"], rehearsal=True)
    roles = sessions.SESSION_ROUND_ROLES.get(day)
    if not roles or len(roles.get("calib", ())) != 1 or not roles.get("test"):
        raise SystemExit(f"{day} 还没在 backend/data/sessions.py 登记轮次用途（需要 1 个 calib 和若干 test）。")
    return dict(calib=roles["calib"][0], test=list(roles["test"]), perturb=list(roles.get("perturb", ())),
                base_rounds=list(BASE_ROUNDS), rehearsal=False)


def out_dir(day: str) -> Path:
    d = OUT_ROOT / (f"{day}_rehearsal" if day == BASE_SESSION else day)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 窗口：与 prepare.load_stage2_windows / link_health.model_windows 完全相同的切法
# ---------------------------------------------------------------------------
def session_windows(session: str, rounds: list[str]):
    """原始 64 子载波窗口。返回 amp/phase [Nw,T,L,64]、label、batch、w（窗口序号），只保留 rounds。"""
    d = load_static_raw64(session)
    sel, kb, kl, kw = lh.model_windows(d["label"], d["batch"], T_WIN)
    keep = np.isin(kb, rounds)
    sel, kb, kl, kw = sel[keep], kb[keep], kl[keep], kw[keep]
    return dict(amp=d["amp"][sel], phase=d["phase"][sel], label=kl.astype(np.int64),
                batch=kb, w=kw, frame_amp=d["amp"][np.isin(d["batch"], rounds)])


def fit_preprocessor(train_parts: list[dict]) -> Preprocessor:
    """子载波选择与归一化统计只用训练集（与 load_stage2_windows 的 train_fold 模式一致）。"""
    sc_idx = select_subcarriers(np.concatenate([p["frame_amp"] for p in train_parts]))
    pp = Preprocessor(subcarrier_indices=sc_idx)
    a = np.concatenate([p["amp"][..., sc_idx] for p in train_parts])
    ph = np.concatenate([p["phase"][..., sc_idx] for p in train_parts])
    L, S = a.shape[2], a.shape[3]
    pp.fit(a.reshape(-1, L, S), ph.reshape(-1, L, S))
    return pp


def to_model_input(pp: Preprocessor, part: dict) -> np.ndarray:
    """原始窗口 -> 模型输入 [Nw,L,C,T,S]（与 prepare.load_stage2_windows.build 相同）。"""
    sc = pp.subcarrier_indices
    a, p = part["amp"][..., sc], part["phase"][..., sc]
    n, T, L, S = a.shape
    a, p = pp.transform(a.reshape(-1, L, S), p.reshape(-1, L, S))
    X = np.stack([a.reshape(n, T, L, S), p.reshape(n, T, L, S)], axis=3).transpose(0, 2, 3, 1, 4)
    return X[:, :, list(CHANNELS)].astype(np.float32)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def _grid(day: str):
    g = sessions.session_grid(day)
    gb = sessions.session_grid(BASE_SESSION)
    if (g.cols, g.rows, g.spacing_m) != (gb.cols, gb.rows, gb.spacing_m):
        raise SystemExit(f"{day} 的网格 {g} 与 D1 {gb} 不同，不能合并训练。")
    return g


def make_model(kind: str, seed: int, grid):
    from backend.models.fusion_baseline import ConcatMLPMaskModel, SingleLinkCNNModel
    kw = dict(lr=1e-3, seed=seed, freq_bins=FREQ_BINS, in_ch=len(CHANNELS), task="cls",
              n_classes=grid.num_classes, grid_cols=grid.cols)
    return ConcatMLPMaskModel(**kw) if kind == "mask" else SingleLinkCNNModel(**kw)


def select_epochs(seed: int) -> dict:
    """从 D1 select（pp2）结果读固定训练轮数 = best_epoch + 1，并核对 §4 的表。"""
    f = config.ARTIFACTS_DIR / "d1" / "select" / f"cls_ampphase_pp2_seed{seed}" / "results.json"
    res = json.loads(f.read_text(encoding="utf-8"))["models"]
    ep = {k: v["best_epoch"] + 1 for k, v in res.items() if "best_epoch" in v}
    if ep["ConcatMLP-mask"] != MASK_EPOCHS[seed]:
        raise SystemExit(f"{f} 里 ConcatMLP-mask 的轮数 {ep['ConcatMLP-mask']} 与预登记 §4 的 "
                         f"{MASK_EPOCHS[seed]} 不符——select 结果被改过？停止。")
    return ep


def logits(model, X: np.ndarray, W: np.ndarray | None = None, batch: int = 512) -> np.ndarray:
    """前向，可选逐窗口链路权重 W [Nw,L]（施加在编码特征上，§5 第 7 条）。"""
    import torch
    net = model.net.eval()
    dev = next(net.parameters()).device
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.tensor(X[s:s + batch], dtype=torch.float32, device=dev)
            if W is None:                       # 单链路 CNN 的 forward 没有 link_weights 参数
                out.append(net(xb).cpu().numpy())
            else:
                wb = torch.tensor(W[s:s + batch], dtype=torch.float32, device=dev)
                out.append(net(xb, link_weights=wb).cpu().numpy())
    return np.concatenate(out)


def label_xy(lab: np.ndarray, cols: int) -> np.ndarray:
    lab = np.asarray(lab)
    return np.stack([lab % cols, lab // cols], axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def cmd_train(day: str, seeds: list[int]) -> None:
    proto, grid = day_protocol(day), _grid(day)
    od = out_dir(day)
    base = session_windows(BASE_SESSION, proto["base_rounds"])
    cal = session_windows(day, [proto["calib"]])
    print(f"==== 训练 {day}{'（演练）' if proto['rehearsal'] else ''}：D1 {proto['base_rounds']} + {day} {proto['calib']}")
    print(f"窗口数：D1 {len(base['label'])}，{day} 标定轮 {len(cal['label'])}")

    # 当天标定模型（§4）+ S2 单链路 CNN + S7 线性基线
    fit_preprocessor([base, cal]).save(od / "preprocessor.json")
    pp = Preprocessor.load(od / "preprocessor.json")     # 训练与评估用同一份（存盘后的）统计量
    Xtr = np.concatenate([to_model_input(pp, base), to_model_input(pp, cal)])
    ytr = np.concatenate([base["label"], cal["label"]])
    print(f"子载波 {len(pp.subcarrier_indices)} 个；训练输入 {Xtr.shape}")

    # S4 "不标定"对照：只用 D1 基础集，预处理也只用 D1（与当天无关；每个 day 目录各存一份，保证自包含）
    fit_preprocessor([base]).save(od / "preprocessor_nocal.json")
    pp_nc = Preprocessor.load(od / "preprocessor_nocal.json")
    Xnc = to_model_input(pp_nc, base)

    from backend.training.train_d1 import LinearModel
    LinearModel(grid.cols).fit(Xtr, ytr).save(od / "Linear")

    for seed in seeds:
        ep = select_epochs(seed)
        t0 = time.time()
        m = make_model("mask", seed, grid).fit(Xtr, ytr, epochs=ep["ConcatMLP-mask"])
        m.save(od / f"mask_seed{seed}")
        m = make_model("mask", seed, grid).fit(Xnc, base["label"], epochs=ep["ConcatMLP-mask"])
        m.save(od / f"mask_nocal_seed{seed}")
        for l, lid in enumerate(LINKS):
            m = make_model("cnn", seed, grid).fit(Xtr[:, l:l + 1], ytr, epochs=ep[f"CNN-{lid}"])
            m.save(od / f"cnn_{lid}_seed{seed}")
        print(f"  种子 {seed}：ConcatMLP-mask {ep['ConcatMLP-mask']} 轮（标定 / 不标定各一个），"
              f"单链路 CNN {[ep[f'CNN-{l}'] for l in LINKS]} 轮  ({time.time() - t0:.0f}s)")

    (od / "train_meta.json").write_text(json.dumps(dict(
        day=day, protocol=proto, seeds=seeds, base_session=BASE_SESSION,
        n_train=int(len(ytr)), n_subcarriers=int(len(pp.subcarrier_indices)),
        trained_at=datetime.now().isoformat(timespec="seconds")), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存到 {od}（训练阶段不在测试轮上算任何成绩）")


# ---------------------------------------------------------------------------
# 健康分（§5、S1）
# ---------------------------------------------------------------------------
def health_scores(day: str, calib: str) -> dict:
    """当天全部窗口的健康分。返回 键(batch,label,w) 与 4 个链路得分 [Nw,L]（越大越健康）。"""
    feats = lh.features_for_session(day)
    cal = lh.Calibration(feats, feats["batch"] == calib)
    feats = cal.with_fp(feats)                          # 标定窗口用留一距离（§5 第 4 条）
    cm = feats["batch"] == calib
    ratio = feats["fp_dist"] / np.median(feats["fp_dist"][cm], axis=0, keepdims=True)   # §5 第 5 条
    ind = lh.indicators(cal.quantile(feats))
    q123 = 1.0 - (ind["phase"] + ind["mad"] + ind["timing"]) / 3.0                      # S1：只取①–③
    return dict(batch=feats["batch"], label=feats["label"], w=feats["w"],
                fp_ratio=ratio, q_fp=1.0 / ratio, q123=q123,
                q_phase=1 - ind["phase"], q_mad=1 - ind["mad"], q_timing=1 - ind["timing"])


def soft_weights(q: np.ndarray) -> np.ndarray:
    """§5 第 6 条：w = q / max(q 四条, 1e-6)，不封顶。"""
    return q / np.maximum(q.max(axis=1, keepdims=True), 1e-6)


# ---------------------------------------------------------------------------
# 指标（§6）
# ---------------------------------------------------------------------------
def round_metrics(err: np.ndarray, correct: np.ndarray, lab: np.ndarray) -> dict:
    """单轮：先按格点平均，再 15 个格点等权（MAE、准确率、>1.6 m）；P90 取全部窗口。"""
    labs = np.unique(lab)
    pt = lambda v: float(np.mean([v[lab == c].mean() for c in labs]))
    return dict(mae=pt(err), acc=pt(correct.astype(float)), far=pt((err > FAR_M).astype(float)),
                p90=float(np.percentile(err, 90)), n_windows=int(len(err)), n_points=int(len(labs)))


def errors_from_logits(z: np.ndarray, lab: np.ndarray, grid) -> tuple[np.ndarray, np.ndarray]:
    pred = z.argmax(1)
    err = np.linalg.norm(label_xy(pred, grid.cols) - label_xy(lab, grid.cols), axis=1) * grid.spacing_m
    return err, pred == lab


def concordance(q: np.ndarray, e: np.ndarray) -> float:
    """S2：窗口内两两比较，"得分高的那条单链路误差更小"的比例（= AUROC；q 相等记 0.5，e 相等不计）。"""
    num, den = 0.0, 0
    for i, j in combinations(range(q.shape[1]), 2):
        de, dq = e[:, i] - e[:, j], q[:, i] - q[:, j]
        v = de != 0
        den += int(v.sum())
        num += float((np.sign(dq[v]) == -np.sign(de[v])).sum() + 0.5 * (dq[v] == 0).sum())
    return num / den if den else float("nan")


def _utc(s: str) -> datetime:
    """PowerShell 的 "o" 格式（例如 2026-10-10T22:40:57.1920028Z，7 位小数）-> naive UTC datetime。"""
    s = s.strip().rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{''.join(ch for ch in frac if ch.isdigit())[:6]}"
    return datetime.fromisoformat(s)


def round_times(day: str, rounds: list[str], calib: str) -> dict:
    """S3：每轮中点距标定结束的分钟数（取 points_meta.csv 的首个 start 与最后 end）。"""
    def span(r):
        f = sessions.session_dir(day) / f"phase2_{r}" / "points_meta.csv"
        if not f.exists():
            return None
        with open(f, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        return min(_utc(x["start_utc"]) for x in rows), max(_utc(x["end_utc"]) for x in rows)
    c = span(calib)
    if c is None:
        return {}
    out = {}
    for r in rounds:
        s = span(r)
        if s:
            out[r] = round(((s[0] - c[1]) + (s[1] - c[1])).total_seconds() / 120.0, 1)
    return out


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def _check_unblind(day: str, unblind: bool) -> None:
    if day == BASE_SESSION:
        return
    if not unblind:
        raise SystemExit(f"{day} 是主验证数据，盲态中（预登记 §3）。确认可以解盲后加 --unblind。")
    if day not in MAIN_DAYS:
        raise SystemExit(f"预登记修订 1：主验证只用 {list(MAIN_DAYS)}，{day} 不在其中。")
    missing = [d for d in MAIN_DAYS if d not in sessions.SESSION_ROUND_ROLES]
    if missing:
        raise SystemExit(f"{missing} 还没在 sessions.py 登记，不能解盲。")
    md5 = hashlib.md5(PREREG.read_bytes()).hexdigest() if PREREG.exists() else None
    if md5 != PREREG_MD5:
        raise SystemExit(f"{PREREG.name} 的 md5 是 {md5}，与'冻结版 + 修订 1'（{PREREG_MD5}）不同。"
                         f"解盲前的修改必须写在预登记的'修订记录'里，并同步更新本文件的 PREREG_MD5。")


def cmd_evaluate(day: str, unblind: bool, seeds: list[int]) -> None:
    _check_unblind(day, unblind)
    proto, grid = day_protocol(day), _grid(day)
    od = out_dir(day)
    rounds = proto["test"] + proto["perturb"]
    part = session_windows(day, rounds)
    pp = Preprocessor.load(od / "preprocessor.json")
    pp_nc = Preprocessor.load(od / "preprocessor_nocal.json")
    X, Xnc, lab, bat = to_model_input(pp, part), to_model_input(pp_nc, part), part["label"], part["batch"]

    h = health_scores(day, proto["calib"])
    key = lambda b, l, w: list(zip(b.tolist(), l.tolist(), w.tolist()))
    idx = {k: i for i, k in enumerate(key(h["batch"], h["label"], h["w"]))}
    pos = np.array([idx[k] for k in key(bat, lab, part["w"])])          # 与模型窗口逐一对齐
    H = {k: v[pos] for k, v in h.items() if k.startswith("q") or k == "fp_ratio"}
    W_fp, W_123 = soft_weights(H["q_fp"]), soft_weights(H["q123"])

    lin = np.load(od / "Linear.npz")
    Xl = X.mean(axis=3).reshape(len(X), -1)
    zl = ((Xl - lin["scaler_mean"]) / lin["scaler_scale"]) @ lin["coef"].T + lin["intercept"]
    zl_lab = lin["classes"][zl.argmax(1)]

    per = {}            # per[r][方案] = 3 个种子的指标列表
    s2 = {}             # s2[r][指标] = 3 个种子的一致比例
    for seed in seeds:
        m = make_model("mask", seed, grid).load(od / f"mask_seed{seed}")
        mnc = make_model("mask", seed, grid).load(od / f"mask_nocal_seed{seed}")
        cnn_err = np.zeros((len(lab), len(LINKS)))
        for l, lid in enumerate(LINKS):
            c = make_model("cnn", seed, grid).load(od / f"cnn_{lid}_seed{seed}")
            cnn_err[:, l], _ = errors_from_logits(logits(c, X[:, l:l + 1]), lab, grid)
        z = {"等权": logits(m, X), "④门控": logits(m, X, W_fp), "S1 ①–③门控": logits(m, X, W_123),
             "S4 不标定·等权": logits(mnc, Xnc)}
        for r in rounds:
            mr = bat == r
            for name, zz in z.items():
                e, c = errors_from_logits(zz[mr], lab[mr], grid)
                per.setdefault(r, {}).setdefault(name, []).append(round_metrics(e, c, lab[mr]))
            for name, q in (("①相位", "q_phase"), ("②MAD", "q_mad"), ("③丢包/抖动", "q_timing"), ("④比值", "q_fp")):
                s2.setdefault(r, {}).setdefault(name, []).append(concordance(H[q][mr], cnn_err[mr]))
    for r in rounds:                                      # S7 线性基线（与种子无关）
        mr = bat == r
        e = np.linalg.norm(label_xy(zl_lab[mr], grid.cols) - label_xy(lab[mr], grid.cols), axis=1) * grid.spacing_m
        per[r]["S7 线性·等权"] = [round_metrics(e, zl_lab[mr] == lab[mr], lab[mr])]

    avg = lambda ms, k: float(np.mean([x[k] for x in ms]))
    res = {"day": day, "rehearsal": proto["rehearsal"], "protocol": proto, "seeds": seeds,
           "minutes_after_calib": round_times(day, rounds, proto["calib"]), "rounds": {}}
    for r in rounds:
        mr = bat == r
        res["rounds"][r] = {
            "role": "test" if r in proto["test"] else "perturb",
            "methods": {n: {k: avg(ms, k) for k in ("mae", "p90", "acc", "far")} for n, ms in per[r].items()},
            "delta_main": avg(per[r]["④门控"], "mae") - avg(per[r]["等权"], "mae"),
            "delta_s1": avg(per[r]["S1 ①–③门控"], "mae") - avg(per[r]["等权"], "mae"),
            "s2_concordance": {n: float(np.mean(v)) for n, v in s2[r].items()},
            "fp_ratio_median": {lid: float(np.median(H["fp_ratio"][mr, l])) for l, lid in enumerate(LINKS)},
        }
    (od / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print_day(res)
    print(f"\n已保存 {od / 'results.json'}")


def print_day(res: dict) -> None:
    tag = "（演练，D1 是开发数据）" if res["rehearsal"] else ""
    print(f"\n===== {res['day']}{tag}  {len(res['seeds'])} 个种子（{res['seeds']}）平均；Δ = ④门控 − 等权（负 = 门控更好）")
    t = res["minutes_after_calib"]
    for r, v in res["rounds"].items():
        mm = v["methods"]
        print(f"\n[{r}] {v['role']}" + (f"  中点 ≈ 标定后 {t[r]} 分钟" if r in t else ""))
        print(f"  {'方案':16s} {'MAE m':>8s} {'P90 m':>7s} {'准确率':>7s} {'>1.6m':>7s}")
        for n, x in mm.items():
            print(f"  {n:16s} {x['mae']:8.3f} {x['p90']:7.2f} {x['acc']:7.0%} {x['far']:7.0%}")
        print(f"  Δ主 = {v['delta_main']:+.3f} m    ΔS1 = {v['delta_s1']:+.3f} m")
        print("  S2 一致比例：" + "  ".join(f"{k} {x:.2f}" for k, x in v["s2_concordance"].items()))
        print("  ④比值中位数：" + "  ".join(f"{k} {x:.2f}" for k, x in v["fp_ratio_median"].items()))


# ---------------------------------------------------------------------------
# report（§7）
# ---------------------------------------------------------------------------
def verdict(days: dict) -> dict:
    """days = {day: {round: Δ}}，只含测试轮。按 §7（修订 1）判定：只有 D+3 一天、3 个测试轮。"""
    day_mean = {d: float(np.mean(list(v.values()))) for d, v in days.items()}
    all_d = [x for v in days.values() for x in v.values()]
    overall = float(np.mean(all_d))
    c = {"① 3 轮平均 Δ ≤ −0.08 m": overall <= -SUPPORT_M,
         "③ 3 轮全部 Δ < 0": len(all_d) == 3 and all(x < 0 for x in all_d),
         "④ 没有一轮 Δ > +0.08 m": all(x <= SUPPORT_M for x in all_d)}
    if all(c.values()):
        v = "明确支持"
    elif overall < 0:
        v = "方向性支持、证据有限"
    else:
        v = "不支持"
    return dict(day_mean=day_mean, overall=overall, n_rounds=len(all_d), conditions=c, verdict=v)


def cmd_report() -> None:
    days = {}
    for d in MAIN_DAYS:
        f = OUT_ROOT / d / "results.json"
        if not f.exists():
            raise SystemExit(f"缺少 {f}：先对 {d} 跑 evaluate --unblind。")
        r = json.loads(f.read_text(encoding="utf-8"))
        days[d] = {k: v["delta_main"] for k, v in r["rounds"].items() if v["role"] == "test"}
    out = verdict(days)
    print("\n===== 预登记 §7（修订 1）判定（Δ = ④门控 − 等权，单位 m）")
    for d, v in days.items():
        print(f"  {d}: " + "  ".join(f"{k} {x:+.3f}" for k, x in v.items()) + f"   日均 {out['day_mean'][d]:+.3f}")
    print(f"  总体 Δ = {out['overall']:+.3f} m（{out['n_rounds']} 轮）")
    for k, ok in out["conditions"].items():
        print(f"    {'✓' if ok else '✗'} {k}")
    print(f"  结论：{out['verdict']}")
    (OUT_ROOT / "report.json").write_text(json.dumps(dict(days=days, **out), ensure_ascii=False, indent=2),
                                          encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="跨日验证（按预登记 report/健康分预登记.md）")
    ap.add_argument("cmd", choices=["train", "evaluate", "report"])
    ap.add_argument("--day", help="d1（演练）/ d3")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--unblind", action="store_true", help="确认解盲（d3 的 evaluate 必须）")
    a = ap.parse_args()
    if a.cmd != "report" and not a.day:
        raise SystemExit("train / evaluate 需要 --day")
    if sorted(a.seeds) != sorted(SEEDS):
        print(f"!! 预登记 §4 规定种子 {list(SEEDS)}；现在用 {a.seeds}（只用于调试，不能当正式结果）")
    {"train": lambda: cmd_train(a.day, a.seeds),
     "evaluate": lambda: cmd_evaluate(a.day, a.unblind, a.seeds),
     "report": cmd_report}[a.cmd]()
