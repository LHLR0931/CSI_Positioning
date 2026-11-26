"""Validation-selected single-link CNN baseline for the D1 experiment.

Purpose
-------
This analysis compares four-link models with single-link CNN results.
This check constructs a stronger and reproducible single-link baseline without
using either final test round to choose the link.

Protocol fixed before running
-----------------------------
1. Candidates are the four single-link CNNs already trained by ``train_d1.py``
   with task=cls, channels=ampphase, preprocessor=pp2, and seeds 42/1/2.
2. Model-selection data are D1 r7 only.  For each link, compute the arithmetic
   mean of r7 mean localisation error across the three seeds.
3. Select one common link: the candidate with the lowest mean r7 error.  The
   same selected link is then used for every seed and every final test round.
4. Final evaluation reads the existing D1 r9 and r10 results for that link and
   reports each round separately.  r9/r10 are never used for selection or
   re-selection.
5. Report mean localisation error, P90, accuracy, and error > 1.6 m rate.  The
   script aggregates seeds within each round and never pools time windows as
   independent statistical units.
6. Evidence status: post-hoc diagnostic.  D1 r7/r9/r10 results had already been
   inspected before this check was proposed, so this is a fairer baseline, not
   a new confirmatory test.  Negative or unfavourable results must be retained.

Inputs
------
backend/artifacts/d1/{select,final}/cls_ampphase_pp2_seed{42,1,2}/results.json

Output
------
backend/artifacts/d1/best_single_link/results.json
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

from backend import config


SEEDS = (42, 1, 2)
SELECT_ROUND = "r7"
FINAL_ROUNDS = ("r9", "r10")
METRICS = ("mean_error_m", "p90_error_m", "accuracy", "far_rate")
TAG = "cls_ampphase_pp2_seed{seed}"


def result_path(phase: str, seed: int) -> Path:
    return config.ARTIFACTS_DIR / "d1" / phase / TAG.format(seed=seed) / "results.json"


def load_results(phase: str, seed: int) -> dict:
    path = result_path(phase, seed)
    if not path.exists():
        raise SystemExit(f"Missing required result file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(values: list[float]) -> dict:
    return {
        "values_by_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
        "mean": mean(values),
        "sample_std": stdev(values),
    }


def main() -> None:
    links = config.all_link_ids()
    select_results = {seed: load_results("select", seed) for seed in SEEDS}

    selection = {}
    for link in links:
        model = f"CNN-{link}"
        values = [
            float(select_results[seed]["models"][model]["by_round"][SELECT_ROUND]["mean_error_m"])
            for seed in SEEDS
        ]
        selection[link] = aggregate(values)

    selected_link = min(links, key=lambda link: selection[link]["mean"])
    selected_model = f"CNN-{selected_link}"

    final_results = {seed: load_results("final", seed) for seed in SEEDS}
    final = {}
    for round_id in FINAL_ROUNDS:
        final[round_id] = {}
        for metric in METRICS:
            values = [
                float(final_results[seed]["models"][selected_model]["by_round"][round_id][metric])
                for seed in SEEDS
            ]
            final[round_id][metric] = aggregate(values)

    output = {
        "evidence_status": "post_hoc_diagnostic",
        "selection_rule": (
            "Choose one common single-link CNN with the lowest three-seed average "
            "of r7 mean localisation error; do not use r9 or r10 for selection."
        ),
        "settings": {
            "session": "d1",
            "task": "cls",
            "channels": "ampphase",
            "preprocessor": "pp2",
            "seeds": list(SEEDS),
            "selection_round": SELECT_ROUND,
            "final_rounds": list(FINAL_ROUNDS),
        },
        "selection": selection,
        "selected_link": selected_link,
        "selected_model": selected_model,
        "final": final,
    }

    out_dir = config.ARTIFACTS_DIR / "d1" / "best_single_link"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Selection round: r7 (three-seed mean localisation error)")
    for link in links:
        row = selection[link]
        print(f"  {link:9s} mean={row['mean']:.3f} m  std={row['sample_std']:.3f} m")
    print(f"Selected link: {selected_link}")
    for round_id in FINAL_ROUNDS:
        row = final[round_id]
        print(
            f"  {round_id}: mean error={row['mean_error_m']['mean']:.3f} m, "
            f"P90={row['p90_error_m']['mean']:.3f} m, "
            f"accuracy={row['accuracy']['mean']:.1%}, "
            f">1.6 m={row['far_rate']['mean']:.1%}"
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
