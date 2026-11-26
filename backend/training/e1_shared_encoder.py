"""E1：冻结单链路编码器（F-MLP）为什么比端到端 ConcatMLP 好？（2026-10-14，跑之前写下判据）

问题
----
ConcatMLP 的 4 条链路共用 1 个编码器；F-MLP 是 4 个编码器，每个先用自己那条链路单独训练、再冻结。
两者同时差了两件事，D1 上 F-MLP 在 r10 好了 0.43 m（0.397 vs 0.826），但不知道功劳归谁：
    (a) 每条链路有自己的编码器（不共享权重）
    (b) 每条链路单独训练（再冻结）

E1 加一个只有 (a)、没有 (b) 的模型 ConcatMLP-sep：4 个不共享的编码器 + 与 ConcatMLP 相同的头，
从零端到端联合训练，不加遮挡（backend/models/fusion_baseline.py::ConcatMLPSep）。
训练协议与 ConcatMLP 完全相同（train_d1：select 在 r6 上早停 → final 用 r1–r8、固定轮数，测 r9 / r10）。

三个模型的关系：
    ConcatMLP      共享编码器 + 联合训练
    ConcatMLP-sep  独立编码器 + 联合训练            ← 只比 ConcatMLP 多了 (a)
    F-MLP          独立编码器 + 单独训练后冻结      ← 只比 ConcatMLP-sep 多了 (b)

跑之前写下的判据（D1 r9 / r10 之前看过，所以是诊断分析，不是预登记检验；D+3 不再动）
--------------------------------------------------------------------------------
只看 r10（标定后约 47 分钟，差距出现的那一轮），3 种子平均，阈值 0.10 m：
    gain_a = ConcatMLP − ConcatMLP-sep      （不共享权重带来的改善）
    gain_b = ConcatMLP-sep − F-MLP          （单独训练再冻结带来的改善）
    结论：
      gain_a ≥ 0.10 且 gain_b < 0.10  → "主要是 (a) 不共享权重"
      gain_a < 0.10 且 gain_b ≥ 0.10  → "主要是 (b) 单独训练"（modality laziness 一类解释成立）
      两者都 ≥ 0.10                   → "(a)、(b) 都有贡献"
      两者都 < 0.10                   → "无法区分"（ConcatMLP 与 F-MLP 相差 0.43 m，理论上不会出现）
r9 只报告，不进判据。逐种子数字一并打印。

用法（项目根目录，先训练 ConcatMLP-sep，3 个种子在本机 GPU 上几分钟）：
    python -X utf8 -m backend.training.train_d1 --phase select --task cls --model ConcatMLP-sep --seed 42 1 2
    python -X utf8 -m backend.training.train_d1 --phase final  --task cls --model ConcatMLP-sep --seed 42 1 2
    python -X utf8 -m backend.training.e1_shared_encoder
输出：backend/artifacts/d1/final/e1_shared_encoder.json
"""
from __future__ import annotations

import json

import numpy as np

from backend import config
from backend.training import train_d1 as T

SEEDS = (42, 1, 2)
THR = 0.10
ROWS = (("ConcatMLP", "共享编码器 + 联合训练"), ("ConcatMLP-sep", "独立编码器 + 联合训练"),
        ("F-MLP", "独立编码器 + 单独训练后冻结"), ("ConcatMLP-mask", "（参考）共享编码器 + 联合训练 + 遮挡"))
OUT = config.ARTIFACTS_DIR / "d1" / "final" / "e1_shared_encoder.json"


def load_seed(seed: int) -> dict:
    d = config.ARTIFACTS_DIR / "d1" / "final" / T.tag_of("cls", "ampphase", seed)
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))["models"]
    if "ConcatMLP-sep" not in res:
        raise SystemExit(f"种子 {seed} 的 final 还没有 ConcatMLP-sep：先跑文件开头的两条 train_d1 命令。")
    ff = d / "frozen_fusion.json"
    if not ff.exists():
        raise SystemExit(f"找不到 {ff}：先跑 frozen_fusion --phase select / final。")
    res["F-MLP"] = json.loads(ff.read_text(encoding="utf-8"))["models"]["F-MLP"]
    return res


def main() -> None:
    res = {s: load_seed(s) for s in SEEDS}
    get = lambda m, r, k="mean_error_m": [res[s][m]["by_round"][r][k] for s in SEEDS]
    out = {"seeds": list(SEEDS), "rounds": {}}
    for r in ("r9", "r10"):
        print(f"\n===== final  {r}  3 个种子：平均（逐种子 42 / 1 / 2）；单位 m")
        print(f"  {'模型':16s} {'结构':28s} {'平均误差':>8s}   {'逐种子':22s} {'P90':>6s} {'准确率':>6s} {'>1.6m':>6s}")
        out["rounds"][r] = {}
        for m, desc in ROWS:
            e = get(m, r)
            p90, acc, far = (float(np.mean(get(m, r, k))) for k in ("p90_error_m", "accuracy", "far_rate"))
            print(f"  {m:16s} {desc:28s} {np.mean(e):8.3f}   {' / '.join(f'{v:.3f}' for v in e):22s} "
                  f"{p90:6.2f} {acc:6.0%} {far:6.0%}")
            out["rounds"][r][m] = {"mae": float(np.mean(e)), "mae_per_seed": e, "p90": p90, "acc": acc, "far": far}
        ep = [res[s]["ConcatMLP-sep"].get("epochs_fixed") for s in SEEDS]
        print(f"  （ConcatMLP-sep 固定训练轮数，种子 42 / 1 / 2：{ep}）")

    m10 = lambda m: out["rounds"]["r10"][m]["mae"]
    gain_a = m10("ConcatMLP") - m10("ConcatMLP-sep")
    gain_b = m10("ConcatMLP-sep") - m10("F-MLP")
    per_a = [a - b for a, b in zip(get("ConcatMLP", "r10"), get("ConcatMLP-sep", "r10"))]
    per_b = [a - b for a, b in zip(get("ConcatMLP-sep", "r10"), get("F-MLP", "r10"))]
    A, B = gain_a >= THR, gain_b >= THR
    verdict = {(True, False): "主要是 (a) 不共享权重", (False, True): "主要是 (b) 单独训练",
               (True, True): "(a)、(b) 都有贡献", (False, False): "无法区分"}[(A, B)]
    print("\n===== 判据（跑之前写下，见文件开头；只看 r10）")
    print(f"  gain_a = ConcatMLP − ConcatMLP-sep = {gain_a:+.3f} m（逐种子 "
          + " / ".join(f"{v:+.3f}" for v in per_a) + f"）  {'≥' if A else '<'} {THR}")
    print(f"  gain_b = ConcatMLP-sep − F-MLP     = {gain_b:+.3f} m（逐种子 "
          + " / ".join(f"{v:+.3f}" for v in per_b) + f"）  {'≥' if B else '<'} {THR}")
    print(f"  结论：{verdict}")
    out.update(gain_a=gain_a, gain_b=gain_b, gain_a_per_seed=per_a, gain_b_per_seed=per_b, verdict=verdict)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
