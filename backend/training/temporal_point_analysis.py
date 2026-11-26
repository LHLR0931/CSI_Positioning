"""Round-level temporal figure and per-grid-point diagnostic analysis.

Analysis and display protocol fixed before running (2026-11-04)
----------------------------------------------------------------
Scope:
    Models: ConcatMLP-mask and F-MLP only.
    D1 rounds: r9 and r10, at their measured midpoints after calibration r8.
    D+3 rounds: r3, r4, and r5, at their measured midpoints after calibration r2.

Primary descriptive metric:
    For each seed and round, compute mean localisation error separately for
    each of the 15 grid points, then average those 15 point means.  This gives
    every grid point equal weight.  Do not pool windows across rounds or treat
    windows as independent experimental units.

Main temporal figure:
    Two panels (D1 and D+3), x = measured minutes after calibration, y = the
    equal-point-weighted round MAE.  Lines are three-seed means and error bars
    are the seed minimum-to-maximum range, not confidence intervals.  No slope,
    significance test, or causal fit is reported.

Per-grid-point figures:
    One figure for D1 and one for D+3.  Rows are the two models.  Columns are
    the earliest test round, latest test round, and latest-minus-earliest MAE.
    The endpoint rule is fixed before seeing the maps.  Every round, including
    D+3 r4, is retained in the JSON and CSV outputs even when not shown in the
    endpoint heatmap.  Absolute maps within a day share one full-range colour
    scale; change maps use a symmetric full-range scale.  Cells show MAE (m)
    and the verified fixed serpentine acquisition order in brackets.

Interpretation limits:
    The figures are descriptive.  Round, elapsed time, grid position, and a
    fixed serpentine acquisition order are not independently randomized.  The
    plots cannot establish that elapsed time caused an error change.  D1 and
    D+3 use different calibration/training sessions and are shown in separate
    panels rather than pooled.

Integrity checks:
    Predictions are reconstructed from saved models and deterministic F-MLP
    training.  D1 window-weighted MAE and D+3 locked equal-point MAE must match
    existing result files within 0.005 m before any output is written.

Outputs:
    backend/artifacts/analysis/temporal_point_analysis.json
    backend/artifacts/analysis/temporal_round_summary.csv
    backend/artifacts/analysis/temporal_per_point.csv
    backend/artifacts/analysis/figures/post_calibration_error_by_round.{pdf,png}
    backend/artifacts/analysis/figures/d1_per_point_error.{pdf,png}
    backend/artifacts/analysis/figures/d3_per_point_error.{pdf,png}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from backend import config
from backend.data import sessions
from backend.data.preprocess import Preprocessor
from backend.training import crossday as C
from backend.training import train_d1 as T
from backend.training.frozen_fusion import encode, load_encoders, make_mlp


SEEDS = (42, 1, 2)
MODELS = ("ConcatMLP-mask", "F-MLP")
DAY_ROUNDS = {"d1": ("r9", "r10"), "d3": ("r3", "r4", "r5")}
CALIBRATION_ROUND = {"d1": "r8", "d3": "r2"}
CHECK_TOL_M = 0.005
OUT_DIR = config.ARTIFACTS_DIR / "analysis"
FIG_DIR = OUT_DIR / "figures"


def label_metrics(pred: np.ndarray, lab: np.ndarray, grid) -> dict:
    pred = np.asarray(pred, dtype=np.int64)
    lab = np.asarray(lab, dtype=np.int64)
    err = np.linalg.norm(C.label_xy(pred, grid.cols) - C.label_xy(lab, grid.cols), axis=1) * grid.spacing_m
    correct = pred == lab
    summary = C.round_metrics(err, correct, lab)
    points = []
    for point in range(grid.num_classes):
        take = lab == point
        if not np.any(take):
            raise SystemExit(f"Round is missing grid point {point}")
        points.append(
            {
                "label": point,
                "x": point % grid.cols,
                "y": point // grid.cols,
                "mae_m": float(err[take].mean()),
                "accuracy": float(correct[take].mean()),
                "far_rate": float((err[take] > C.FAR_M).mean()),
                "n_windows": int(take.sum()),
            }
        )
    return {
        "mae_m": float(summary["mae"]),
        "window_weighted_mae_m": float(err.mean()),
        "p90_m": float(summary["p90"]),
        "accuracy": float(summary["acc"]),
        "far_rate": float(summary["far"]),
        "n_windows": int(len(err)),
        "points": points,
    }


def acquisition_order(day: str, rounds: tuple[str, ...]) -> dict[int, int]:
    reference = None
    for round_id in rounds:
        path = sessions.session_dir(day) / f"phase2_{round_id}" / "points_meta.csv"
        with path.open(encoding="utf-8-sig", newline="") as fh:
            labels = [int(row["label"]) for row in csv.DictReader(fh) if row.get("status") == "ok"]
        if len(labels) != 15 or len(set(labels)) != 15:
            raise SystemExit(f"Unexpected acquisition sequence in {path}: {labels}")
        if reference is None:
            reference = labels
        elif labels != reference:
            raise SystemExit(f"Acquisition order differs across {day} rounds: {round_id} has {labels}")
    return {label: index + 1 for index, label in enumerate(reference or [])}


def frozen_epochs(seed: int) -> int:
    path = (
        config.ARTIFACTS_DIR
        / "d1"
        / "select"
        / T.tag_of("cls", "ampphase", seed)
        / "frozen_fusion.json"
    )
    row = json.loads(path.read_text(encoding="utf-8"))["models"]["F-MLP"]
    return int(row["best_epoch"]) + 1


def evaluate_d1() -> tuple[dict, dict[str, float]]:
    day = "d1"
    rounds = DAY_ROUNDS[day]
    grid = sessions.session_grid(day)
    data, _ = T.load("final", "ampphase")
    Xtr, Xte = data["train"][0], data["test"][0]
    lab, batch = data["test"][2], data["test"][3]
    stats = {r: {name: {} for name in MODELS} for r in rounds}

    for seed in SEEDS:
        d = config.ARTIFACTS_DIR / "d1" / "final" / T.tag_of("cls", "ampphase", seed)
        mask, _ = T.make_model("ConcatMLP-mask", seed, grid, "cls", Xtr.shape[2])
        mask.load(d / "ConcatMLP-mask")
        mask_pred = T.snap_to_label(mask.predict(Xte), grid)

        encoders = load_encoders("final", seed, grid, Xtr.shape[2], data)
        Ftr, Fte = encode(encoders, Xtr), encode(encoders, Xte)
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        fmlp = make_mlp(seed, Ftr.shape[1], grid).fit(
            (Ftr - mu) / sd, data["train"][2], epochs=frozen_epochs(seed)
        )
        fmlp_pred = fmlp._forward_np((Fte - mu) / sd).argmax(1)

        predictions = {"ConcatMLP-mask": mask_pred, "F-MLP": fmlp_pred}
        locked_main = json.loads((d / "results.json").read_text(encoding="utf-8"))["models"]
        locked_frozen = json.loads((d / "frozen_fusion.json").read_text(encoding="utf-8"))["models"]
        for round_id in rounds:
            take = batch == round_id
            for name in MODELS:
                row = label_metrics(predictions[name][take], lab[take], grid)
                stats[round_id][name][str(seed)] = row
                locked = (
                    locked_main[name]["by_round"][round_id]["mean_error_m"]
                    if name == "ConcatMLP-mask"
                    else locked_frozen[name]["by_round"][round_id]["mean_error_m"]
                )
                if abs(row["window_weighted_mae_m"] - locked) > CHECK_TOL_M:
                    raise SystemExit(
                        f"D1 integrity check failed: seed={seed}, {name}, {round_id}, "
                        f"recomputed={row['window_weighted_mae_m']:.3f}, locked={locked:.3f}"
                    )
    return stats, C.round_times(day, list(rounds), CALIBRATION_ROUND[day])


def evaluate_d3() -> tuple[dict, dict[str, float]]:
    day = "d3"
    rounds = DAY_ROUNDS[day]
    proto, grid = C.day_protocol(day), C._grid(day)
    od = C.OUT_ROOT / day
    pp = Preprocessor.load(od / "preprocessor.json")
    base = C.session_windows(C.BASE_SESSION, proto["base_rounds"])
    cal = C.session_windows(day, [proto["calib"]])
    Xtr = np.concatenate([C.to_model_input(pp, base), C.to_model_input(pp, cal)])
    ytr = np.concatenate([base["label"], cal["label"]])
    part = C.session_windows(day, list(rounds))
    Xte = C.to_model_input(pp, part)
    lab, batch = part["label"], part["batch"]
    stats = {r: {name: {} for name in MODELS} for r in rounds}

    locked = json.loads((od / "frozen_fusion_results.json").read_text(encoding="utf-8"))["rounds"]
    for seed in SEEDS:
        mask = C.make_model("mask", seed, grid).load(od / f"mask_seed{seed}")
        mask_pred = C.logits(mask, Xte).argmax(1)

        encoders = [
            C.make_model("cnn", seed, grid).load(od / f"cnn_{link}_seed{seed}")
            for link in config.all_link_ids()
        ]
        Ftr, Fte = encode(encoders, Xtr), encode(encoders, Xte)
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        fmlp = make_mlp(seed, Ftr.shape[1], grid).fit(
            (Ftr - mu) / sd, ytr, epochs=frozen_epochs(seed)
        )
        fmlp_pred = fmlp._forward_np((Fte - mu) / sd).argmax(1)

        predictions = {"ConcatMLP-mask": mask_pred, "F-MLP": fmlp_pred}
        for round_id in rounds:
            take = batch == round_id
            for name in MODELS:
                row = label_metrics(predictions[name][take], lab[take], grid)
                stats[round_id][name][str(seed)] = row
                locked_name = "mask 等权" if name == "ConcatMLP-mask" else "F-MLP"
                locked_value = locked[round_id]["methods"][locked_name]["mae_per_seed"][SEEDS.index(seed)]
                if abs(row["mae_m"] - locked_value) > CHECK_TOL_M:
                    raise SystemExit(
                        f"D+3 integrity check failed: seed={seed}, {name}, {round_id}, "
                        f"recomputed={row['mae_m']:.3f}, locked={locked_value:.3f}"
                    )
    return stats, C.round_times(day, list(rounds), CALIBRATION_ROUND[day])


def seed_values(stats: dict, day: str, round_id: str, model: str, key: str) -> list[float]:
    return [float(stats[day][round_id][model][str(seed)][key]) for seed in SEEDS]


def point_seed_values(stats: dict, day: str, round_id: str, model: str, label: int) -> list[float]:
    return [
        float(stats[day][round_id][model][str(seed)]["points"][label]["mae_m"])
        for seed in SEEDS
    ]


def write_tables(stats: dict, times: dict, orders: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    round_path = OUT_DIR / "temporal_round_summary.csv"
    with round_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "day", "round", "minutes_after_calibration", "model", "seed", "mae_m",
            "window_weighted_mae_m", "p90_m", "accuracy", "far_rate", "n_windows",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for day, rounds in DAY_ROUNDS.items():
            for round_id in rounds:
                for model in MODELS:
                    for seed in SEEDS:
                        row = stats[day][round_id][model][str(seed)]
                        writer.writerow(
                            {
                                "day": day,
                                "round": round_id,
                                "minutes_after_calibration": times[day][round_id],
                                "model": model,
                                "seed": seed,
                                **{key: row[key] for key in fields if key in row},
                            }
                        )

    point_path = OUT_DIR / "temporal_per_point.csv"
    with point_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "day", "round", "minutes_after_calibration", "model", "seed", "label", "x", "y",
            "acquisition_order", "mae_m", "accuracy", "far_rate", "n_windows",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for day, rounds in DAY_ROUNDS.items():
            for round_id in rounds:
                for model in MODELS:
                    for seed in SEEDS:
                        for point in stats[day][round_id][model][str(seed)]["points"]:
                            writer.writerow(
                                {
                                    "day": day,
                                    "round": round_id,
                                    "minutes_after_calibration": times[day][round_id],
                                    "model": model,
                                    "seed": seed,
                                    "acquisition_order": orders[day][point["label"]],
                                    **point,
                                }
                            )


def save_figure(fig, stem: str, *, raster_pdf: bool = False) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIG_DIR / f"{stem}.png"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    if raster_pdf:
        from PIL import Image

        with Image.open(png_path) as image:
            image.convert("RGB").save(pdf_path, "PDF", resolution=400.0)
    else:
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )


def plot_temporal(stats: dict, times: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
    styles = {
        "ConcatMLP-mask": dict(color="#D55E00", marker="s", label="ConcatMLP-mask"),
        "F-MLP": dict(color="#0072B2", marker="o", label="F-MLP"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharey=True)
    max_seed_mae = 0.0
    for ax, day, title in zip(axes, ("d1", "d3"), ("(a) D1", "(b) D+3")):
        rounds = DAY_ROUNDS[day]
        x = np.array([times[day][r] for r in rounds])
        for model in MODELS:
            values = [seed_values(stats, day, r, model, "mae_m") for r in rounds]
            max_seed_mae = max(max_seed_mae, max(max(v) for v in values))
            y = np.array([np.mean(v) for v in values])
            lo = y - np.array([min(v) for v in values])
            hi = np.array([max(v) for v in values]) - y
            ax.errorbar(x, y, yerr=np.vstack([lo, hi]), linewidth=1.8, capsize=3, **styles[model])
        for xi, round_id in zip(x, rounds):
            ax.annotate(round_id, (xi, 0), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Minutes after calibration (round midpoint)")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylim(0, max_seed_mae * 1.08)
    axes[0].set_ylabel("Equal-point-weighted mean error (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.01, "Error bars show the minimum and maximum across three seeds.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save_figure(fig, "post_calibration_error_by_round")
    plt.close(fig)


def point_map(stats: dict, day: str, round_id: str, model: str, grid) -> np.ndarray:
    values = [np.mean(point_seed_values(stats, day, round_id, model, label)) for label in range(grid.num_classes)]
    return np.asarray(values).reshape(grid.rows, grid.cols)


def plot_point_maps(stats: dict, times: dict, orders: dict, day: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
    grid = sessions.session_grid(day)
    early, late = DAY_ROUNDS[day][0], DAY_ROUNDS[day][-1]
    maps = {
        model: {
            "early": point_map(stats, day, early, model, grid),
            "late": point_map(stats, day, late, model, grid),
        }
        for model in MODELS
    }
    for model in MODELS:
        maps[model]["delta"] = maps[model]["late"] - maps[model]["early"]
    abs_max = max(float(maps[m][k].max()) for m in MODELS for k in ("early", "late"))
    delta_max = max(float(np.abs(maps[m]["delta"]).max()) for m in MODELS) or 1e-6

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.8), constrained_layout=True)
    abs_images, delta_images = [], []
    for row, model in enumerate(MODELS):
        for col, key in enumerate(("early", "late", "delta")):
            ax = axes[row, col]
            arr = maps[model][key]
            if key == "delta":
                im = ax.imshow(arr, origin="lower", cmap="coolwarm", vmin=-delta_max, vmax=delta_max)
                delta_images.append(im)
            else:
                im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=0, vmax=abs_max)
                abs_images.append(im)
            for y in range(grid.rows):
                for x in range(grid.cols):
                    label = y * grid.cols + x
                    order = orders[day][label]
                    value = arr[y, x]
                    ax.text(x, y, f"{value:.2f}\n[{order}]", ha="center", va="center", fontsize=7,
                            color="white" if (key != "delta" and value > abs_max * 0.55) else "black")
            ax.set_xticks(range(grid.cols))
            ax.set_yticks(range(grid.rows))
            ax.set_xlabel("x grid index")
            if col == 0:
                ax.set_ylabel(f"{model}\ny grid index")
            if row == 0:
                if key == "early":
                    ax.set_title(f"{early}: {times[day][early]:.1f} min")
                elif key == "late":
                    ax.set_title(f"{late}: {times[day][late]:.1f} min")
                else:
                    ax.set_title(f"Change: {late} - {early}")
    fig.colorbar(abs_images[0], ax=axes[:, :2], shrink=0.85, label="Per-point MAE (m)")
    fig.colorbar(delta_images[0], ax=axes[:, 2], shrink=0.85, label="MAE change (m)")
    day_title = "D+3" if day == "d3" else day.upper()
    fig.suptitle(f"{day_title} per-grid-point localisation error", fontsize=12)
    fig.text(0.5, -0.02, "Cell text: MAE in metres; [n] is the fixed serpentine acquisition order.",
             ha="center", fontsize=9)
    save_figure(fig, f"{day}_per_point_error", raster_pdf=(day == "d3"))
    plt.close(fig)


def main() -> None:
    print("Reconstructing D1 predictions...")
    d1_stats, d1_times = evaluate_d1()
    print("D1 integrity checks passed.")
    print("Reconstructing D+3 predictions...")
    d3_stats, d3_times = evaluate_d3()
    print("D+3 integrity checks passed.")

    stats = {"d1": d1_stats, "d3": d3_stats}
    times = {"d1": d1_times, "d3": d3_times}
    orders = {day: acquisition_order(day, DAY_ROUNDS[day]) for day in DAY_ROUNDS}
    if orders["d1"] != orders["d3"]:
        raise SystemExit(f"D1 and D+3 acquisition orders differ: {orders}")

    write_tables(stats, times, orders)
    plot_temporal(stats, times)
    plot_point_maps(stats, times, orders, "d1")
    plot_point_maps(stats, times, orders, "d3")

    output = {
        "evidence_status": "post_hoc_descriptive",
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "rounds": {day: list(rounds) for day, rounds in DAY_ROUNDS.items()},
        "minutes_after_calibration": times,
        "acquisition_order_by_label": {str(k): v for k, v in orders["d1"].items()},
        "metric": "per-point MAE followed by equal weighting of 15 points",
        "error_bars": "minimum-to-maximum across three seeds",
        "stats": stats,
        "limitations": [
            "Round and elapsed time were not randomized.",
            "Every round used the same serpentine acquisition order.",
            "Grid position and within-round acquisition time are confounded.",
            "D1 and D+3 are separate calibrated training/evaluation sessions and are not pooled.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "temporal_point_analysis.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nThree-seed round means (equal point weighting):")
    for day, rounds in DAY_ROUNDS.items():
        for round_id in rounds:
            text = "  ".join(
                f"{model}={np.mean(seed_values(stats, day, round_id, model, 'mae_m')):.3f} m"
                for model in MODELS
            )
            print(f"  {day} {round_id} ({times[day][round_id]:.1f} min): {text}")
    print(f"Wrote analysis tables to {OUT_DIR}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
