"""D+3 post-hoc factor ablation for the F-MLP comparison.

Purpose
-------
F-MLP and ConcatMLP-mask differ in three design choices: encoder sharing,
training protocol, and random link masking.  This diagnostic analysis adds the
two missing controlled comparisons on the already collected D+3 data.

Evaluation protocol (fixed on 2026-11-04)
-----------------------------------------
Models:
    ConcatMLP       shared encoder + joint end-to-end training + no masking
    ConcatMLP-mask  shared encoder + joint end-to-end training + masking
    ConcatMLP-sep   independent encoders + joint end-to-end training + no masking
    F-MLP           independent encoders + single-link pretraining, then frozen

All models use the existing cross-day protocol: D1 r1-r8 plus D+3 r2 for
training, the saved D+3 preprocessor, seeds 42/1/2, and D+3 r3/r4/r5 for test.
Model-specific epoch counts are fixed from D1 model selection and are never
tuned on D+3.

Controlled gains (positive means that the named factor lowered MAE):
    mask_gain = MAE(ConcatMLP) - MAE(ConcatMLP-mask)
    independent_encoder_gain = MAE(ConcatMLP) - MAE(ConcatMLP-sep)
    staged_training_gain = MAE(ConcatMLP-sep) - MAE(F-MLP)

For each model and round, first average MAE across the three seeds.  Then give
r3/r4/r5 equal weight; do not pool time windows as independent units.

For each factor:
    consistent support: mean gain across r3/r4/r5 >= 0.10 m and every round > 0
    directional support: mean gain > 0 but the first rule is not fully met
    no support: mean gain <= 0

MAE is the primary metric.  P90, accuracy, and error > 1.6 m rate are reported
but cannot change the verdict.  Negative and inconsistent results are retained.

Evidence status and limits
--------------------------
This is post-hoc diagnostic evidence because the D+3 test results had already
been inspected.  ``staged_training_gain`` identifies the bundle of single-link
pretraining plus frozen fusion; it does not separate pretraining from freezing.
The design is not a full factorial experiment and cannot estimate interactions.

Integrity checks
----------------
The script re-evaluates the saved ConcatMLP-mask and re-trains F-MLP exactly as
the earlier frozen-fusion script did.  If either three-seed round mean differs
from the locked result by more than 0.005 m, execution stops without a verdict.

Output
------
backend/artifacts/crossday/d3/fmlp_factor_ablation_results.json
"""
from __future__ import annotations

import json
import time

import numpy as np

from backend import config
from backend.data.preprocess import Preprocessor
from backend.training import crossday as C
from backend.training.frozen_fusion import encode, make_mlp


DAY = "d3"
SEEDS = (42, 1, 2)
ROUNDS = ("r3", "r4", "r5")
MODELS = ("ConcatMLP", "ConcatMLP-mask", "ConcatMLP-sep", "F-MLP")
GAIN_THRESHOLD_M = 0.10
CHECK_TOL_M = 0.005
OUT = C.OUT_ROOT / DAY / "fmlp_factor_ablation_results.json"


def d1_epochs(seed: int) -> dict[str, int]:
    d = config.ARTIFACTS_DIR / "d1" / "select" / f"cls_ampphase_pp2_seed{seed}"
    main = json.loads((d / "results.json").read_text(encoding="utf-8"))["models"]
    frozen = json.loads((d / "frozen_fusion.json").read_text(encoding="utf-8"))["models"]
    needed = ("ConcatMLP", "ConcatMLP-mask", "ConcatMLP-sep")
    missing = [name for name in needed if "best_epoch" not in main.get(name, {})]
    if missing:
        raise SystemExit(f"Missing D1 select epochs for seed {seed}: {missing}")
    if "best_epoch" not in frozen.get("F-MLP", {}):
        raise SystemExit(f"Missing D1 select F-MLP epoch for seed {seed}: {d / 'frozen_fusion.json'}")
    return {
        **{name: int(main[name]["best_epoch"]) + 1 for name in needed},
        "F-MLP": int(frozen["F-MLP"]["best_epoch"]) + 1,
    }


def make_joint_model(name: str, seed: int, grid):
    from backend.models.fusion_baseline import ConcatMLPModel, ConcatMLPSepModel

    kw = dict(
        lr=1e-3,
        seed=seed,
        freq_bins=C.FREQ_BINS,
        in_ch=len(C.CHANNELS),
        task="cls",
        n_classes=grid.num_classes,
        grid_cols=grid.cols,
    )
    if name == "ConcatMLP":
        return ConcatMLPModel(**kw)
    if name == "ConcatMLP-sep":
        return ConcatMLPSepModel(**kw)
    raise ValueError(name)


def round_metrics(pred: np.ndarray, lab: np.ndarray, grid) -> dict:
    err = np.linalg.norm(C.label_xy(pred, grid.cols) - C.label_xy(lab, grid.cols), axis=1) * grid.spacing_m
    return C.round_metrics(err, pred == lab, lab)


def verdict(gains: dict[str, float]) -> dict:
    values = [float(gains[r]) for r in ROUNDS]
    mean_gain = float(np.mean(values))
    all_positive = all(value > 0 for value in values)
    if mean_gain >= GAIN_THRESHOLD_M and all_positive:
        label = "consistent_support"
    elif mean_gain > 0:
        label = "directional_support"
    else:
        label = "no_support"
    return {
        "gain_by_round_m": {r: float(gains[r]) for r in ROUNDS},
        "mean_gain_m": mean_gain,
        "all_rounds_positive": all_positive,
        "threshold_m": GAIN_THRESHOLD_M,
        "verdict": label,
    }


def main() -> None:
    proto, grid = C.day_protocol(DAY), C._grid(DAY)
    if tuple(proto["test"]) != ROUNDS:
        raise SystemExit(f"Unexpected D+3 test rounds: {proto['test']} (expected {list(ROUNDS)})")

    od = C.OUT_ROOT / DAY
    pp = Preprocessor.load(od / "preprocessor.json")
    base = C.session_windows(C.BASE_SESSION, proto["base_rounds"])
    cal = C.session_windows(DAY, [proto["calib"]])
    Xtr = np.concatenate([C.to_model_input(pp, base), C.to_model_input(pp, cal)])
    ytr = np.concatenate([base["label"], cal["label"]])

    meta = json.loads((od / "train_meta.json").read_text(encoding="utf-8"))
    if len(ytr) != meta["n_train"]:
        raise SystemExit(f"Training windows {len(ytr)} != locked cross-day count {meta['n_train']}")

    part = C.session_windows(DAY, list(ROUNDS))
    Xte = C.to_model_input(pp, part)
    lab, bat = part["label"], part["batch"]
    per = {r: {name: [] for name in MODELS} for r in ROUNDS}
    epochs_used = {}

    print(
        f"==== D+3 F-MLP factor ablation: train D1 {proto['base_rounds']} + {DAY} {proto['calib']} "
        f"({len(ytr)} windows); test {list(ROUNDS)}"
    )

    for seed in SEEDS:
        t0 = time.time()
        ep = d1_epochs(seed)
        epochs_used[str(seed)] = ep
        predictions = {}

        mask = C.make_model("mask", seed, grid).load(od / f"mask_seed{seed}")
        predictions["ConcatMLP-mask"] = C.logits(mask, Xte).argmax(1)

        for name in ("ConcatMLP", "ConcatMLP-sep"):
            model = make_joint_model(name, seed, grid).fit(Xtr, ytr, epochs=ep[name])
            model.save(od / f"factor_{name.lower()}_seed{seed}")
            predictions[name] = C.logits(model, Xte).argmax(1)

        encoders = [
            C.make_model("cnn", seed, grid).load(od / f"cnn_{link}_seed{seed}")
            for link in config.all_link_ids()
        ]
        Ftr, Fte = encode(encoders, Xtr), encode(encoders, Xte)
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        fmlp = make_mlp(seed, Ftr.shape[1], grid).fit(
            (Ftr - mu) / sd, ytr, epochs=ep["F-MLP"]
        )
        predictions["F-MLP"] = fmlp._forward_np((Fte - mu) / sd).argmax(1)

        for round_id in ROUNDS:
            take = bat == round_id
            for name in MODELS:
                per[round_id][name].append(round_metrics(predictions[name][take], lab[take], grid))
        print(
            f"  seed {seed}: ConcatMLP {ep['ConcatMLP']} epochs, mask {ep['ConcatMLP-mask']}, "
            f"ConcatMLP-sep {ep['ConcatMLP-sep']}, F-MLP {ep['F-MLP']} "
            f"({time.time() - t0:.0f}s)"
        )

    avg = lambda r, name, key: float(np.mean([row[key] for row in per[r][name]]))

    locked = json.loads((od / "frozen_fusion_results.json").read_text(encoding="utf-8"))["rounds"]
    for round_id in ROUNDS:
        for name, locked_name in (("ConcatMLP-mask", "mask 等权"), ("F-MLP", "F-MLP")):
            got = avg(round_id, name, "mae")
            want = float(locked[round_id]["methods"][locked_name]["mae"])
            if abs(got - want) > CHECK_TOL_M:
                raise SystemExit(
                    f"Integrity check failed for {name} {round_id}: recomputed {got:.3f} m, "
                    f"locked {want:.3f} m (tolerance {CHECK_TOL_M:.3f} m)"
                )
    print("  Integrity checks passed for ConcatMLP-mask and F-MLP.")

    results = {
        "evidence_status": "post_hoc_diagnostic",
        "day": DAY,
        "seeds": list(SEEDS),
        "training_protocol": {
            "base_session": C.BASE_SESSION,
            "base_rounds": proto["base_rounds"],
            "calibration_round": proto["calib"],
            "test_rounds": list(ROUNDS),
            "epochs_source": "D1 select only",
            "epochs_used": epochs_used,
        },
        "primary_metric": "round-level three-seed mean MAE; r3/r4/r5 equally weighted",
        "rounds": {},
    }

    print("\n===== Round results (three-seed mean; MAE in m)")
    for round_id in ROUNDS:
        results["rounds"][round_id] = {}
        print(f"[{round_id}]")
        for name in MODELS:
            seed_rows = per[round_id][name]
            row = {
                "mae": avg(round_id, name, "mae"),
                "mae_per_seed": [float(x["mae"]) for x in seed_rows],
                "p90": avg(round_id, name, "p90"),
                "accuracy": avg(round_id, name, "acc"),
                "far_rate": avg(round_id, name, "far"),
            }
            results["rounds"][round_id][name] = row
            print(
                f"  {name:15s} MAE={row['mae']:.3f}  P90={row['p90']:.3f}  "
                f"acc={row['accuracy']:.1%}  >1.6m={row['far_rate']:.1%}"
            )

    mae = lambda r, name: results["rounds"][r][name]["mae"]
    gains = {
        "mask": {r: mae(r, "ConcatMLP") - mae(r, "ConcatMLP-mask") for r in ROUNDS},
        "independent_encoder": {r: mae(r, "ConcatMLP") - mae(r, "ConcatMLP-sep") for r in ROUNDS},
        "staged_training": {r: mae(r, "ConcatMLP-sep") - mae(r, "F-MLP") for r in ROUNDS},
    }
    results["contrasts"] = {name: verdict(values) for name, values in gains.items()}
    results["limitations"] = [
        "D+3 outcomes were inspected before this analysis; results are post-hoc diagnostic.",
        "Staged training combines single-link pretraining and frozen fusion.",
        "The design is not fully factorial and does not estimate interactions.",
    ]

    print("\n===== Fixed verdict rules")
    for name, row in results["contrasts"].items():
        gains_text = ", ".join(f"{r}={value:+.3f}" for r, value in row["gain_by_round_m"].items())
        print(f"  {name:19s} {gains_text}; mean={row['mean_gain_m']:+.3f} m -> {row['verdict']}")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
