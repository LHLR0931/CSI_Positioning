"""F-MLP 的训练与推理（实时后端用）。

训练协议与 D+3 验证（backend/training/frozen_fusion_d3.py）相同，只是标定轮换成当天的：
    训练集    D1 r1–r8 + 当天标定轮（session 的第 round 轮）
    预处理    crossday.fit_preprocessor：子载波选择与归一化统计只用这份训练集
    编码器    4 个单链路 CNN（crossday.make_model("cnn")），每个只用自己那条链路训练，
              轮数 = D1 select 的 best_epoch + 1（crossday.select_epochs）；训练完冻结
    融合头    frozen_fusion.make_mlp：Linear(256,64) → ReLU → Linear(64,15)，
              输入为训练集标准化后的 4×64 维编码特征；轮数 = D1 select 的 F-MLP best_epoch + 1
    种子      默认 42（实时演示只用一个种子）
所有函数都直接调用原脚本里的实现，不重写，保证与论文结果是同一个模型。

产物目录 backend/artifacts/live/<session>/：
    preprocessor.json  cnn_<链路>.pt  head.pt  feat_norm.npz  train_meta.json
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from backend import config
from backend.data.preprocess import Preprocessor
from backend.live.linkbuf import LINKS

MODEL_ID = "F-MLP"
OUT_ROOT = config.ARTIFACTS_DIR / "live"


def out_dir(session: str) -> Path:
    return OUT_ROOT / session


def fmlp_epochs(seed: int) -> int:
    f = config.ARTIFACTS_DIR / "d1" / "select" / f"cls_ampphase_pp2_seed{seed}" / "frozen_fusion.json"
    if not f.exists():
        raise SystemExit(f"找不到 {f}：F-MLP 的训练轮数取自 D1 select 阶段的 frozen_fusion 结果。")
    return int(json.loads(f.read_text(encoding="utf-8"))["models"]["F-MLP"]["best_epoch"]) + 1


def train(session: str, calib_round: str, seed: int = 42, log=print) -> Path:
    from backend.training import crossday as C
    from backend.training.frozen_fusion import encode, make_mlp

    grid = C._grid(session)
    od = out_dir(session)
    od.mkdir(parents=True, exist_ok=True)
    base = C.session_windows(C.BASE_SESSION, list(C.BASE_ROUNDS))
    cal = C.session_windows(session, [calib_round])
    if len(cal["label"]) == 0:
        raise SystemExit(f"{session} {calib_round} 没有任何窗口，检查标定轮是否采完、是否已解析。")
    missing = sorted(set(range(grid.num_classes)) - set(np.unique(cal["label"]).tolist()))
    log(f"训练集：D1 {list(C.BASE_ROUNDS)} {len(base['label'])} 窗口 + {session} {calib_round} "
        f"{len(cal['label'])} 窗口" + (f"；⚠ 标定轮缺格点 {missing}" if missing else "；标定轮 15 个格点齐全"))

    C.fit_preprocessor([base, cal]).save(od / "preprocessor.json")
    pp = Preprocessor.load(od / "preprocessor.json")          # 训练与推理用同一份存盘后的统计量
    Xtr = np.concatenate([C.to_model_input(pp, base), C.to_model_input(pp, cal)])
    ytr = np.concatenate([base["label"], cal["label"]])
    log(f"子载波 {len(pp.subcarrier_indices)} 个；训练输入 {Xtr.shape}")

    ep = C.select_epochs(seed)
    t0 = time.time()
    encs = []
    for l, lid in enumerate(LINKS):
        m = C.make_model("cnn", seed, grid).fit(Xtr[:, l:l + 1], ytr, epochs=ep[f"CNN-{lid}"])
        m.save(od / f"cnn_{lid}")
        encs.append(m)
        log(f"  单链路 CNN {lid}：{ep[f'CNN-{lid}']} 轮  ({time.time() - t0:.0f}s)")
    F = encode(encs, Xtr)
    mu, sd = F.mean(0), F.std(0) + 1e-6
    n_ep = fmlp_epochs(seed)
    head = make_mlp(seed, F.shape[1], grid).fit((F - mu) / sd, ytr, epochs=n_ep)
    head.save(od / "head")
    np.savez(od / "feat_norm.npz", mu=mu, sd=sd)
    log(f"  F-MLP 融合头：{n_ep} 轮  ({time.time() - t0:.0f}s)")

    # 只是自检：模型在自己的训练数据（标定轮）上拟合得怎样。这不是测试成绩，不能写进论文。
    ncal = len(cal["label"])
    pred = head._forward_np((F[-ncal:] - mu) / sd).argmax(1)
    fit_acc = float((pred == cal["label"]).mean())
    log(f"  自检（训练集拟合，不是测试成绩）：标定轮窗口准确率 {fit_acc:.0%}")

    meta = dict(model_id=MODEL_ID, session=session, calib_round=calib_round, seed=seed,
                base_session=C.BASE_SESSION, base_rounds=list(C.BASE_ROUNDS),
                n_train=int(len(ytr)), n_calib=int(ncal), calib_missing_labels=missing,
                n_subcarriers=int(len(pp.subcarrier_indices)),
                cnn_epochs={lid: ep[f"CNN-{lid}"] for lid in LINKS}, head_epochs=n_ep,
                calib_fit_acc=fit_acc, grid=dict(cols=grid.cols, rows=grid.rows, spacing_m=grid.spacing_m),
                trained_at=datetime.now().astimezone().isoformat(timespec="seconds"))
    (od / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已保存到 {od}")
    return od


class LiveFMLP:
    """加载 train() 的产物，对一个原始窗口（64 个子载波）输出 15 类概率。"""

    def __init__(self, session: str):
        import torch
        from backend.training import crossday as C
        from backend.training.frozen_fusion import make_mlp

        od = out_dir(session)
        self.meta = json.loads((od / "train_meta.json").read_text(encoding="utf-8"))
        self.grid = C._grid(session)
        self.pp = Preprocessor.load(od / "preprocessor.json")
        seed = int(self.meta["seed"])
        self.encs = [C.make_model("cnn", seed, self.grid).load(od / f"cnn_{lid}") for lid in LINKS]
        nf = np.load(od / "feat_norm.npz")
        self.mu, self.sd = nf["mu"], nf["sd"]
        self.head = make_mlp(seed, len(self.mu), self.grid).load(od / "head")
        self._torch = torch
        self._to_input = C.to_model_input

    def features(self, X: np.ndarray) -> np.ndarray:
        from backend.training.frozen_fusion import encode
        return encode(self.encs, X)

    def predict_proba_raw(self, amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """amp / phase：[T, L, 64] 或 [N, T, L, 64]（原始 64 个子载波，链路顺序 = LINKS）。返回 [N, 15]。"""
        if amp.ndim == 3:
            amp, phase = amp[None], phase[None]
        X = self._to_input(self.pp, {"amp": amp, "phase": phase})       # [N,L,C,T,S]，与训练同一函数
        F = (self.features(X) - self.mu) / self.sd
        z = self.head._forward_np(F)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)
