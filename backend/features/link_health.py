"""链路健康分 —— 每条链路、每个 16 帧时间窗，从**原始 CSI** 算出"这条链路此刻有多不正常"。

为什么必须用原始 CSI
--------------------
预处理（Hampel 去离群、S-G 平滑、逐子载波标准化）的目的恰恰是抹掉健康分要检测的异常。
在平滑之后算 MAD = 先擦掉证据再取证。所以这里只读 static_raw64.npz（未经任何预处理的
64 子载波幅度/相位 + 每帧每链路 pcap 时间戳），或直接读 pcap。

窗口与模型完全对齐
------------------
窗口切法与 backend/training/prepare.load_stage2_windows 逐行一致：按 label、再按 batch
（都用 np.unique 排序）遍历，每 16 帧一个不重叠窗口。每个窗口带键 (batch, label, w)，
可与模型的逐窗口预测一一对上。

四个指标（5 个原始量），越大越不正常
------------------------------------
    ① phase_resid  去趋势相位残差：每帧取 55 个有效子载波，**按真实频率排序**后展开相位、
                   最小二乘拟合直线（斜率=硬件 SFO/包边界偏移，截距=CFO 等公共相位），
                   残差均方根（rad）；16 帧取中位数。
    ② amp_mad      归一化时间 MAD：每个子载波 16 帧幅度的 MAD ÷ 该子载波 16 帧幅度中位数，
                   再对子载波取中位数。除以自身中位数，消掉链路之间、AGC 档位之间的增益差。
    ③ loss         丢包：AP 每 102.4 ms 发一个 beacon；按时间戳估计窗口内缺了几个。
                   loss = 缺失数 /（缺失数 + 16）。
       jitter      到达抖动：每个包间隔偏离 102.4 ms 整数倍的绝对值，取中位数（ms）。
    ④ fp_dist      指纹偏离度（2026-10-08 加）：窗口指纹 = 16 帧平均的对数幅度曲线（55 个子载波，
                   每帧先除以自身均值消 AGC）。标定时每条链路在 15 个格点各留一个标准指纹（类中心）。
                   偏离度 = 当前指纹到 15 个标准指纹中**最近那个**的距离（按标定轮类内标准差逐子载波缩放）。
                   不需要知道人站在哪；只用标定轮。标定窗口自身用"留一"中心算，避免自己和自己比。
                   为什么要加：①–③ 衡量"信号干不干净"，看不到"信号干净但指纹变了"（Pi 重启、设备
                   被碰歪 >λ/2=2.9 cm）。D1 上 piB 重启后精度退到地板，而①–③ 没有报警。

子载波（2026-10-08 实测确认）
------------------------------
原始 64 个子载波是 **FFT 顺序**：编号 1–27 = +1…+27，36–63 = −28…−1，0 = DC，
28–35 = 频带边缘（实测幅度 10315 / 34190 / 134 / 0 / 7428 / 7169 / 22 / 10844，全是垃圾）。
有效集合 = 1–27 ∪ 36–63，共 55 个。按物理频率排序后再拟合直线：r1 piA_apX 残差中位数
0.669 rad（编号顺序）→ 0.620 rad（物理顺序）。

标定与分位数
------------
每条链路、每个原始量，各自与该链路在标定轮里的分布比较，换算成分位数
p = 标定轮中"小于等于当前值"的窗口比例（0–1，越大越不正常）。链路之间从不直接比数值。
指标③ 取 loss 与 jitter 两个分位数的较大者。三个分位数如何合成一个健康分，见 combine()。

用法（项目根目录）：
    python -X utf8 -m backend.features.link_health d1 --calib r8
    python -X utf8 -m backend.features.link_health d1 --calib r8 --anomalies
输出：datasets/raw/<session>/link_health.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backend import config

T_WIN = 16                 # 与模型窗口一致（config.STAGE2_TIME_WINDOW）
BEACON_S = 0.1024          # 两台 AP 的 beacon 间隔，实测约 10 帧/秒/链路
VALID_SC = np.r_[1:28, 36:64]                                   # 55 个有效子载波（原始编号）
K_PHYS = np.where(VALID_SC < 32, VALID_SC, VALID_SC - 64)       # 对应的物理子载波号 −28…+27
_ORDER = np.argsort(K_PHYS)
SC_SORTED = VALID_SC[_ORDER]                                    # 按物理频率排好的原始编号
K_SORTED = K_PHYS[_ORDER].astype(np.float64)
RAW_FEATURES = ("phase_resid", "amp_mad", "loss", "jitter", "fp_dist")
INDICATORS = ("phase", "mad", "timing", "fp")                   # 四个指标


# ---------------------------------------------------------------------------
# 原始量：输入一组窗口 amp/phase [Nw,T,L,64]、ts [Nw,T,L]，输出 {名字: [Nw,L]}
# ---------------------------------------------------------------------------
def phase_residual(phase: np.ndarray) -> np.ndarray:
    """[..., 64] 原始相位 -> [...] 去趋势残差均方根（rad）。"""
    p = np.unwrap(phase[..., SC_SORTED].astype(np.float64), axis=-1)
    k = K_SORTED - K_SORTED.mean()
    pc = p - p.mean(axis=-1, keepdims=True)
    slope = (pc * k).sum(-1, keepdims=True) / (k ** 2).sum()
    resid = pc - slope * k
    return np.sqrt((resid ** 2).mean(-1))


def raw_features(amp: np.ndarray, phase: np.ndarray, ts: np.ndarray) -> dict[str, np.ndarray]:
    # ① 每帧残差，再取 16 帧中位数
    f_phase = np.median(phase_residual(phase), axis=1)                       # [Nw,L]
    # ② 每个子载波的时间 MAD / 时间中位数，再对子载波取中位数
    a = amp[..., VALID_SC].astype(np.float64)                                # [Nw,T,L,55]
    med = np.median(a, axis=1)
    mad = np.median(np.abs(a - med[:, None]), axis=1)
    f_mad = np.median(mad / (med + 1e-6), axis=-1)                           # [Nw,L]
    # ③ 包间隔
    gaps = np.diff(ts, axis=1)                                               # [Nw,T-1,L]
    n_int = np.rint(gaps / BEACON_S)
    missing = np.clip(n_int - 1, 0, None).sum(axis=1)
    f_loss = missing / (missing + ts.shape[1])
    dev = np.abs(gaps - n_int * BEACON_S) * 1e3                              # ms
    dev = np.where(n_int >= 1, dev, np.nan)          # 间隔不到半个 beacon 的是 AP 的非 beacon 帧，不计抖动
    f_jit = np.nan_to_num(np.nanmedian(dev, axis=1), nan=0.0) if np.isfinite(dev).any() else np.zeros_like(f_loss)
    # ④ 窗口指纹（距离要等标定后才能算，这里只给向量）
    an = a / (a.mean(axis=-1, keepdims=True) + 1e-6)                         # 每帧除以自身均值（消 AGC）
    fp_vec = np.log(an + 1e-3).mean(axis=1)                                  # [Nw,L,55]
    return {"phase_resid": f_phase, "amp_mad": f_mad, "loss": f_loss, "jitter": f_jit,
            "fp_vec": fp_vec}


# ---------------------------------------------------------------------------
# 与模型对齐的窗口
# ---------------------------------------------------------------------------
def model_windows(labels: np.ndarray, batches: np.ndarray, T: int = T_WIN):
    """逐行复刻 prepare.load_stage2_windows 的窗口切法。返回帧索引 [Nw,T] 与键。"""
    sel, kb, kl, kw = [], [], [], []
    for lbl in np.unique(labels):
        for b in np.unique(batches):
            idx = np.where((labels == lbl) & (batches == b))[0]
            for w in range(len(idx) // T):
                sel.append(idx[w * T:(w + 1) * T]); kb.append(b); kl.append(lbl); kw.append(w)
    return np.asarray(sel), np.asarray(kb), np.asarray(kl), np.asarray(kw)


def features_for_session(session: str) -> dict:
    d = np.load(config.DATASET_ARTIFACTS_DIR / session / "static_raw64.npz")
    sel, kb, kl, kw = model_windows(d["label"], d["batch"])
    amp, phase, ts = d["amp"][sel], d["phase"][sel], d["timestamp"][sel]    # [Nw,T,L,64] / [Nw,T,L]
    out = raw_features(amp, phase, ts)
    out.update(batch=kb, label=kl, w=kw, links=d["links"])
    return out


# ---------------------------------------------------------------------------
# 标定：每条链路、每个原始量，与该链路自己在标定轮的分布比较 -> 分位数
# ---------------------------------------------------------------------------
class Calibration:
    def __init__(self, feats: dict, calib_mask: np.ndarray):
        if calib_mask.sum() < 20:
            raise ValueError(f"标定窗口太少（{calib_mask.sum()} 个）")
        # ④ 标准指纹：每条链路、每个格点一个类中心；按类内标准差逐子载波缩放
        v = feats["fp_vec"][calib_mask]                                      # [Nc,L,55]
        lab = feats["label"][calib_mask]
        self.classes = np.unique(lab)
        self.cnt = np.array([(lab == c).sum() for c in self.classes])        # [K]
        self.cent = np.stack([v[lab == c].mean(axis=0) for c in self.classes])   # [K,L,55]
        resid = v - self.cent[np.searchsorted(self.classes, lab)]
        self.scale = resid.std(axis=0) + 1e-3                                # [L,55]
        d_cal = self.fp_distance(v, loo_labels=lab)                          # 标定窗口用留一中心
        feats_cal = {f: feats[f][calib_mask] for f in RAW_FEATURES if f != "fp_dist"}
        feats_cal["fp_dist"] = d_cal
        self.ref = {f: np.sort(feats_cal[f], axis=0) for f in RAW_FEATURES}  # [Nc,L]
        self.n = int(calib_mask.sum())
        self.calib_mask = calib_mask

    def fp_distance(self, vec: np.ndarray, loo_labels: np.ndarray | None = None) -> np.ndarray:
        """vec [N,L,55] -> 到最近标准指纹的距离 [N,L]。loo_labels 给出时，自己所在类用留一中心。"""
        diff = (vec[:, None] - self.cent[None]) / self.scale[None, None]     # [N,K,L,55]
        d2 = (diff ** 2).mean(axis=-1)                                       # [N,K,L]
        if loo_labels is not None:
            k = np.searchsorted(self.classes, loo_labels)
            n = self.cnt[k][:, None, None].astype(np.float64)                # [N,1,1]
            c_loo = (n * self.cent[k] - vec) / np.maximum(n - 1, 1)          # [N,L,55]
            d2[np.arange(len(vec)), k] = (((vec - c_loo) / self.scale) ** 2).mean(axis=-1)
        return np.sqrt(d2.min(axis=1))

    def with_fp(self, feats: dict) -> dict:
        """补上 fp_dist。若 feats 就是整场数据（含标定窗口），标定窗口用留一距离。"""
        if "fp_dist" in feats:
            return feats
        out = dict(feats)
        d = self.fp_distance(feats["fp_vec"])
        if "label" in feats and len(feats["fp_vec"]) == len(self.calib_mask):
            m = self.calib_mask
            d[m] = self.fp_distance(feats["fp_vec"][m], loo_labels=feats["label"][m])
        out["fp_dist"] = d
        return out

    def quantile(self, feats: dict) -> dict[str, np.ndarray]:
        """p = 标定轮中（< 当前值的比例）+ 0.5 ×（= 当前值的比例），即"中位秩"。每条链路各自比较。

        为什么用中位秩：丢包和抖动在干净数据里绝大多数窗口恰好是 0。若用"≤ 当前值的比例"，
        一个值为 0 的健康窗口会得到 p≈0.98，被误判成"比 98% 的标定窗口都差"。
        中位秩下它得到 p≈0.5，与连续量（相位残差、MAD）在干净数据上的期望一致。
        """
        feats = self.with_fp(feats)
        out = {}
        for f in RAW_FEATURES:
            ref, x = self.ref[f], feats[f]
            p = np.empty_like(x, dtype=np.float64)
            for l in range(x.shape[1]):
                lo = np.searchsorted(ref[:, l], x[:, l], side="left")
                hi = np.searchsorted(ref[:, l], x[:, l], side="right")
                p[:, l] = (lo + 0.5 * (hi - lo)) / self.n
            out[f] = p
        return out


def indicators(p: dict) -> dict[str, np.ndarray]:
    """5 个原始量的分位数 -> 4 个指标的分位数（③ 取丢包与抖动中较差的一个）。"""
    return {"phase": p["phase_resid"], "mad": p["amp_mad"],
            "timing": np.maximum(p["loss"], p["jitter"]), "fp": p["fp_dist"]}


def combine(ind: dict, how: str = "mean") -> np.ndarray:
    """各指标分位数 -> 健康分 q（0–1，越大越健康）。合成方式在 D+3 之前用开发数据定，两种都给出。

    mean：q = 1 − 各指标平均。干净数据上 q 的期望约 0.5。
    max ：q = 1 − 各指标最大（任何一个指标异常就判不健康）。
    """
    stack = np.stack([ind[k] for k in INDICATORS], axis=0)
    return 1.0 - (stack.mean(axis=0) if how == "mean" else stack.max(axis=0))


# ---------------------------------------------------------------------------
# 直接从一对 pcap 算（异常样本不在 static_raw64.npz 里）
# ---------------------------------------------------------------------------
def features_from_pcaps(pi1: Path, pi2: Path, T: int = T_WIN) -> dict:
    """一个点的两份 pcap -> 四条链路各自切 16 帧窗口（不做四链路截断对齐），返回每链路的原始量。

    异常样本里某条链路可能只剩几帧，四链路按最短截断会把整个样本截没，所以这里各链路独立切窗。
    返回 {链路: {原始量: [Nw]}}，以及每链路的总帧数。
    """
    from backend.data.load_real_stage2 import _read_pcap_split, LINK_ORDER
    streams = {}
    for pi_id, fp in (("piA", pi1), ("piB", pi2)):
        with open(fp, "rb") as f:
            for ap_id, s in _read_pcap_split(f).items():
                streams[f"{pi_id}_{ap_id}"] = s
    out, counts = {}, {}
    for lid in LINK_ORDER:
        csi, ts = streams[lid]["csi"], streams[lid]["timestamp"]
        counts[lid] = len(ts)
        nw = len(ts) // T
        if nw == 0:
            out[lid] = None
            continue
        c = csi[:nw * T].reshape(nw, T, 1, -1)
        feats = raw_features(np.abs(c), np.angle(c), ts[:nw * T].reshape(nw, T, 1))
        out[lid] = {k: v[:, 0] for k, v in feats.items()}          # fp_vec 变成 [Nw,55]
    return {"features": out, "counts": counts}


# ---------------------------------------------------------------------------
# 命令行：算全部窗口、保存、打印概览
# ---------------------------------------------------------------------------
def _round_key(b: str) -> int:
    return int(b[1:])


def main(session: str, calib: list[str], show_anomalies: bool) -> None:
    feats = features_for_session(session)
    links = [str(v) for v in feats["links"]]
    cmask = np.isin(feats["batch"], calib)
    cal = Calibration(feats, cmask)
    feats = cal.with_fp(feats)
    p = cal.quantile(feats)          # 注意：标定轮只对同一次开机的轮次有意义（相位残差基线随重启改变）
    ind = indicators(p)
    q_mean, q_max = combine(ind, "mean"), combine(ind, "max")

    out = config.DATASET_ARTIFACTS_DIR / session / "link_health.npz"
    np.savez(out, batch=feats["batch"], label=feats["label"], w=feats["w"], links=feats["links"],
             calib_rounds=np.array(calib),
             **{f"raw_{k}": feats[k] for k in RAW_FEATURES}, fp_vec=feats["fp_vec"],
             **{f"p_{k}": p[k] for k in RAW_FEATURES},
             **{f"ind_{k}": ind[k] for k in INDICATORS},
             q_mean=q_mean, q_max=q_max)
    print(f"session={session}  标定轮={calib}（{cal.n} 个窗口）  共 {len(feats['batch'])} 个窗口 × {len(links)} 条链路")
    print(f"已保存 {out}\n")

    rounds = sorted(set(feats["batch"].tolist()), key=_round_key)
    print("各轮各链路原始量的中位数（相位残差 rad | 时间 MAD | 丢包率 | 抖动 ms | 指纹偏离度）")
    print(f"{'轮':5s}" + "".join(f"{l:>38s}" for l in links))
    for r in rounds:
        m = feats["batch"] == r
        cells = [f"{np.median(feats['phase_resid'][m, i]):.3f} | {np.median(feats['amp_mad'][m, i]):.3f} | "
                 f"{np.mean(feats['loss'][m, i]):.3f} | {np.median(feats['jitter'][m, i]):.2f} | "
                 f"{np.median(feats['fp_dist'][m, i]):.2f}"
                 for i in range(len(links))]
        print(f"{r:5s}" + "".join(f"{c:>38s}" for c in cells) + ("   ← 标定" if r in calib else ""))

    print("\n各轮各链路「指纹偏离度」分位数的中位数；标定轮约 0.5，越高越不像标定时")
    print(f"{'轮':5s}" + "".join(f"{l:>10s}" for l in links))
    for r in rounds:
        m = feats["batch"] == r
        print(f"{r:5s}" + "".join(f"{np.median(ind['fp'][m, i]):>10.2f}" for i in range(len(links)))
              + ("   ← 标定" if r in calib else ""))

    print("\n各轮各链路健康分 q（mean 合成）的中位数；标定轮约 0.5，越低越不正常")
    print(f"{'轮':5s}" + "".join(f"{l:>10s}" for l in links))
    for r in rounds:
        m = feats["batch"] == r
        print(f"{r:5s}" + "".join(f"{np.median(q_mean[m, i]):>10.2f}" for i in range(len(links)))
              + ("   ← 标定" if r in calib else ""))

    if show_anomalies:
        root = config.SOURCE_DATA_DIR / session
        print("\n异常样本（不在训练数据里）：与**同一轮**的干净数据比较（同一次开机，基线可比）。")
        print("各链路四个指标分位数的中位数 [相位, MAD, 丢包/抖动, 指纹] 与 q_mean（同轮干净数据上 q 约 0.5）")
        for adir in sorted(root.glob("anomaly_*")):
            same_round = adir.name.split("_")[1]                     # anomaly_r3_... -> r3
            cal = Calibration(feats, feats["batch"] == same_round)
            for pi1 in sorted(adir.glob("label*_pi1.pcap")):
                pi2 = pi1.with_name(pi1.name.replace("_pi1", "_pi2"))
                r = features_from_pcaps(pi1, pi2)
                print(f"\n  {adir.name}/{pi1.name.replace('_pi1.pcap', '')}（对照 {same_round}）  帧数 "
                      + "  ".join(f"{l}={r['counts'][l]}" for l in links))
                for i, l in enumerate(links):
                    f = r["features"][l]
                    if f is None:
                        print(f"    {l:8s} 帧数不足 16，无法成窗 → 视为最不健康（q=0）")
                        continue
                    one = {k: np.repeat(f[k][:, None], len(links), axis=1)
                           for k in RAW_FEATURES if k != "fp_dist"}
                    one["fp_vec"] = np.repeat(f["fp_vec"][:, None, :], len(links), axis=1)
                    pp = cal.quantile(one)
                    pi = indicators({k: v[:, i:i + 1] for k, v in pp.items()})
                    qm = combine(pi, "mean")[:, 0]
                    print(f"    {l:8s} [{np.median(pi['phase']):.2f}, {np.median(pi['mad']):.2f}, "
                          f"{np.median(pi['timing']):.2f}, {np.median(pi['fp']):.2f}]  "
                          f"q_mean={np.median(qm):.2f}  （{len(qm)} 个窗口）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="链路健康分：原始量、标定分位数、健康分")
    ap.add_argument("session")
    ap.add_argument("--calib", nargs="+", required=True, help="标定轮，例如 r8")
    ap.add_argument("--anomalies", action="store_true", help="同时检查 data/<session>/anomaly_* 异常样本")
    a = ap.parse_args()
    main(a.session, a.calib, a.anomalies)
