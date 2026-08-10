"""
Experiment: HIF F1 at a matched operating point (held-out error types).

Table 2 (see 05_heldout_error_baselines.py) reports HIF F1 at the fixed
non-parametric threshold H > 0.5, while IF/LOF are given a tuned
``contamination`` parameter equal to the known corruption rate. This script
recomputes HIF F1 at the *same* operating point as those baselines: it flags the
top ``n_corrupt = max(1, int(n_rows * level))`` rows by HIF penalty, matching
the oracle calibration advantage granted to the baselines.

HIF detection is re-run with the current library code, which is verified to
reproduce the committed 05 outputs exactly (see tests). The script writes
``outputs/heldout_errors_matched.csv`` with per-seed H > 0.5 and matched-rate
F1 for HIF alongside the committed IF/LOF F1 at contamination = level.

Run:
    python scripts/05_heldout_matched_threshold.py --dataset census_acs --rows 2000 \
      --seeds 42 --levels 0.4
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

# ruff: noqa: E402
from _exp_utils import generate, load_real

_spec = importlib.util.spec_from_file_location(
    "heldout_baselines", SCRIPTS_DIR / "05_heldout_error_baselines.py"
)
heldout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(heldout)

CORRUPTION_STRATEGIES = {
    "random_injection": heldout.corrupt_random_injection,
    "semantic_hallucination": heldout.corrupt_semantic_hallucination,
    "row_duplication": heldout.corrupt_row_duplication,
    "feature_dropout": heldout.corrupt_feature_dropout,
    "covariate_shift": heldout.corrupt_covariate_shift,
}


def compute_matched_f1(
    scores: np.ndarray, true_labels: np.ndarray, level: float
) -> float:
    """F1 flagging the top ``ceil(n * level)`` records by score (oracle rate)."""
    n_flag = max(1, int(len(scores) * level))
    order = np.argsort(scores)[::-1]
    preds = np.zeros(len(scores), dtype=int)
    preds[order[:n_flag]] = 1
    return float(f1_score(true_labels, preds, zero_division=0.0))


def run_experiment(
    dataset_id: str,
    n_rows: int,
    seeds: list[int],
    levels: list[float],
    output_dir: Path,
):
    print(f"Loading dataset: {dataset_id}")
    real = load_real(dataset_id, n=n_rows)

    committed = pd.read_csv(output_dir / "heldout_errors_raw.csv")
    committed = committed[committed["method"] == "HIF"]

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        print(f"Seed {seed}...")
        syn_clean = generate(real, n_rows, seed, "gaussian_copula")
        syn_clean = syn_clean[real.columns.intersection(syn_clean.columns).tolist()]

        gmm = heldout.fit_gmm(real, seed=seed)

        for strategy_name, corrupt_fn in CORRUPTION_STRATEGIES.items():
            for level in levels:
                if strategy_name == "row_duplication":
                    corrupted, labels = corrupt_fn(syn_clean, level, rng)
                else:
                    corrupted, labels = corrupt_fn(syn_clean, real, level, rng)

                hif_preds, hif_scores = heldout.detect_hif(real, corrupted, seed=seed)
                gmm_preds, gmm_scores = heldout.detect_gmm(
                    gmm, real, corrupted, contamination=level
                )
                f1_default = float(f1_score(labels, hif_preds, zero_division=0.0))
                f1_matched = compute_matched_f1(hif_scores, labels, level)
                gmm_f1_default = float(f1_score(labels, gmm_preds, zero_division=0.0))
                gmm_f1_matched = compute_matched_f1(gmm_scores, labels, level)

                comm = committed[
                    (committed["seed"] == seed)
                    & (committed["error_type"] == strategy_name)
                    & (committed["corruption_level"] == level)
                ]
                comm_f1 = float(comm["f1"].iloc[0]) if len(comm) else float("nan")
                if not np.isclose(f1_default, comm_f1, atol=1e-6):
                    print(
                        f"  MISMATCH seed={seed} {strategy_name}: "
                        f"recomputed {f1_default:.6f} vs committed {comm_f1:.6f}"
                    )

                rows.append(
                    {
                        "seed": seed,
                        "error_type": strategy_name,
                        "corruption_level": level,
                        "f1_threshold_default": f1_default,
                        "f1_matched_rate": f1_matched,
                        "gmm_f1_threshold_default": gmm_f1_default,
                        "gmm_f1_matched_rate": gmm_f1_matched,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "heldout_errors_matched.csv", index=False)

    summary = (
        df.groupby(["error_type", "corruption_level"])
        .agg(
            f1_threshold_mean=("f1_threshold_default", "mean"),
            f1_threshold_sem=("f1_threshold_default", sem),
            f1_matched_mean=("f1_matched_rate", "mean"),
            f1_matched_sem=("f1_matched_rate", sem),
            gmm_f1_threshold_mean=("gmm_f1_threshold_default", "mean"),
            gmm_f1_threshold_sem=("gmm_f1_threshold_default", sem),
            gmm_f1_matched_mean=("gmm_f1_matched_rate", "mean"),
            gmm_f1_matched_sem=("gmm_f1_matched_rate", sem),
        )
        .round(3)
        .reset_index()
    )
    summary.to_csv(output_dir / "heldout_errors_matched_summary.csv", index=False)

    baseline = pd.read_csv(output_dir / "heldout_errors_raw.csv")
    base = (
        baseline[
            (baseline["corruption_level"].isin(levels)) & (baseline["method"] != "HIF")
        ]
        .groupby(["error_type", "method"])["f1"]
        .mean()
        .unstack()
        .round(3)
    )

    print("\n\n" + "=" * 80)
    print("HIF F1 at matched operating point (top n*level rows) vs baselines")
    print("=" * 80)
    merged = summary.merge(base, left_on="error_type", right_index=True, how="left")
    for _, row in merged.iterrows():
        print(
            f"{row['error_type']:<24} HIF@0.5: {row['f1_threshold_mean']:.3f} "
            f"| HIF@matched: {row['f1_matched_mean']:.3f} "
            f"| IF: {row.get('IsolationForest', float('nan')):.3f} "
            f"| LOF: {row.get('LOF', float('nan')):.3f} "
            f"| GMM@contam: {row.get('gmm_f1_threshold_mean', float('nan')):.3f} "
            f"| GMM@matched: {row.get('gmm_f1_matched_mean', float('nan')):.3f}"
        )

    print(f"\nResults saved to {output_dir / 'heldout_errors_matched.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HIF F1 at a matched operating point (held-out error types)"
    )
    parser.add_argument("--dataset", default="census_acs")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--corruption-levels", default="0.4")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    levels = [float(x) for x in args.corruption_levels.split(",")]
    run_experiment(
        dataset_id=args.dataset,
        n_rows=args.rows,
        seeds=seeds,
        levels=levels,
        output_dir=Path(args.output_dir),
    )
