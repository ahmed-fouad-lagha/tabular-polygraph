"""
Experiment: Component Ablation Study.

Q: "What contribution does each layer (symbolic rules, neural sentinels, NIC continuity) make to overall performance?"

Tests each HIF component individually and compares against the full ensemble,
plus additive vs geometric aggregation.

Addresses:
  - Q1: "Extend by showing individual performance by rejecting
    using each component / ablations of components from the ensemble vs HIF"
  - Q2: "Show mathematically why the algebraic intersection
    (geometric mean) is superior to an additive metric"

Run:
    python scripts/08_ablation_study.py --dataset census_acs --rows 2000 --seeds 5
"""

from __future__ import annotations

import argparse

# ruff: noqa: E402
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tabular_polygraph.dataset import load_dataset
from tabular_polygraph.fidelity import hif_score
from tabular_polygraph.fidelity.downstream import tstr_score
from tabular_polygraph.generators import (
    BaseGenerator,
    CTGANGenerator,
    GaussianCopulaGenerator,
)

ABLATION_CONFIGS = [
    {"name": "LSE-only", "ablation_mode": "lse_only", "aggregation": "geometric"},
    {"name": "NIC-only", "ablation_mode": "nic_only", "aggregation": "geometric"},
    {"name": "Rules-only", "ablation_mode": "rules_only", "aggregation": "geometric"},
    {"name": "LSE + NIC", "ablation_mode": "lse_nic", "aggregation": "geometric"},
    {"name": "Full HIF (geom.)", "ablation_mode": "full", "aggregation": "geometric"},
    {"name": "Full HIF (arith.)", "ablation_mode": "full", "aggregation": "arithmetic"},
]


def _compute_utility(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target: str,
    seed: int,
) -> dict:
    """Compute TSTR F1 for a given real/synthetic pair."""
    num_cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]
    cat_cols = [c for c in real.columns if not pd.api.types.is_numeric_dtype(real[c])]

    # Build feature columns (one-hot encode low-cardinality categoricals)
    real_util = real.copy()
    syn_util = syn.copy()
    encoded_cols = []

    for col in cat_cols:
        if col == target or col not in syn.columns:
            continue
        if real[col].nunique() > 50:
            continue
        dummies = pd.get_dummies(real[col], prefix=f"ohe__{col}").astype(float)
        real_util = pd.concat([real_util, dummies], axis=1)
        syn_dummies = pd.get_dummies(syn[col], prefix=f"ohe__{col}").astype(float)
        for d_col in dummies.columns:
            if d_col in syn_dummies.columns:
                syn_util[d_col] = syn_dummies[d_col]
            else:
                syn_util[d_col] = 0.0
        encoded_cols.extend(dummies.columns)

    feature_cols = [
        c for c in num_cols if c != target and c in syn.columns
    ] + encoded_cols

    if not feature_cols:
        return {"f1": np.nan, "accuracy": np.nan}

    # Discretize target to binary median split
    u_real, u_syn = real_util.copy(), syn_util.copy()
    if pd.api.types.is_numeric_dtype(u_real[target]) and u_real[target].nunique() > 2:
        m = u_real[target].median()
        u_real[target] = (u_real[target] > m).astype(int)
        u_syn[target] = (u_syn[target] > m).astype(int)

    try:
        result = tstr_score(
            u_real,
            u_syn,
            target_col=target,
            feature_cols=feature_cols,
            task="classification",
            seed=seed,
        )
        if "error" in result:
            return {"f1": np.nan, "accuracy": np.nan}
        return {"f1": float(result.get("tstr_score", np.nan)), "accuracy": np.nan}
    except Exception:
        return {"f1": np.nan, "accuracy": np.nan}


def run_ablation(
    dataset_id: str,
    target: str,
    n_rows: int,
    n_seeds: int,
    generator_type: str,
    output_dir: Path,
):
    print(f"Loading dataset: {dataset_id}")
    real = load_dataset(dataset_id, n=n_rows)

    print(f"Fitting generator: {generator_type}")
    gen: BaseGenerator
    if generator_type == "ctgan":
        gen = CTGANGenerator()
    else:
        gen = GaussianCopulaGenerator()
    gen.fit(real)

    all_cols = real.columns.tolist()
    all_results = []

    for seed_i in range(n_seeds):
        seed = 42 + seed_i
        print(f"\n{'=' * 60}")
        print(f"Seed {seed} ({seed_i + 1}/{n_seeds})")
        print(f"{'=' * 60}")

        syn = gen.generate(n_rows, seed=seed)
        syn = syn.drop(columns=["syn_id"], errors="ignore")
        common_cols = [c for c in all_cols if c in syn.columns]
        syn = syn[common_cols]

        # Full synthetic utility (no filtering)
        util_full = _compute_utility(real, syn, target, seed)
        all_results.append(
            {
                "seed": seed,
                "ablation": "No filtering",
                "retention_pct": 100.0,
                "f1": util_full["f1"],
                "accuracy": util_full["accuracy"],
                "violation_rate": 0.0,
                "hif_score": np.nan,
            }
        )

        for config in ABLATION_CONFIGS:
            name = config["name"]
            print(f"  [{name}] ...", end="", flush=True)

            try:
                result = hif_score(
                    real,
                    syn,
                    columns=common_cols,
                    random_state=seed,
                    verbose=False,
                    ablation_mode=config["ablation_mode"],
                    aggregation=config["aggregation"],
                )
                penalties = result["row_penalties"]
                mask = penalties < 0.5
                syn_filtered = syn[mask]
                retention = len(syn_filtered) / len(syn) * 100

                if len(syn_filtered) > 10:
                    util = _compute_utility(real, syn_filtered, target, seed)
                else:
                    util = {"f1": np.nan, "accuracy": np.nan}

                all_results.append(
                    {
                        "seed": seed,
                        "ablation": name,
                        "retention_pct": round(retention, 1),
                        "f1": util["f1"],
                        "accuracy": util["accuracy"],
                        "violation_rate": result["violation_rate"],
                        "hif_score": result["hif_score"],
                    }
                )
                print(
                    f" Done. (retention={retention:.1f}%, F1={util['f1']:.3f})"
                    if not np.isnan(util["f1"])
                    else " Done."
                )

            except Exception as e:
                print(f" Failed: {e}")
                all_results.append(
                    {
                        "seed": seed,
                        "ablation": name,
                        "retention_pct": np.nan,
                        "f1": np.nan,
                        "accuracy": np.nan,
                        "violation_rate": np.nan,
                        "hif_score": np.nan,
                    }
                )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / f"ablation_{dataset_id}_raw.csv", index=False)

    # Summary table
    summary = (
        df.groupby("ablation")
        .agg(
            retention_mean=("retention_pct", "mean"),
            f1_mean=("f1", "mean"),
            f1_sem=("f1", sem),
            accuracy_mean=("accuracy", "mean"),
            violation_rate_mean=("violation_rate", "mean"),
            hif_score_mean=("hif_score", "mean"),
        )
        .round(4)
        .reset_index()
    )

    # Reorder to match paper presentation
    order = [
        "No filtering",
        "LSE-only",
        "NIC-only",
        "Rules-only",
        "LSE + NIC",
        "Full HIF (arith.)",
        "Full HIF (geom.)",
    ]
    summary["ablation"] = pd.Categorical(
        summary["ablation"], categories=order, ordered=True
    )
    summary = summary.sort_values("ablation").reset_index(drop=True)
    summary.to_csv(output_dir / f"ablation_{dataset_id}_summary.csv", index=False)

    # Print markdown table
    print("\n\n" + "=" * 80)
    print(f"Ablation Study ({dataset_id})")
    print("=" * 80)
    print("| Ablation | Retention% | F1 (mean ± SEM) | Violation Rate |")
    print("|---|---|---|---|")
    for _, row in summary.iterrows():
        f1_str = (
            f"{row['f1_mean']:.3f} ± {row['f1_sem']:.3f}"
            if not np.isnan(row["f1_mean"])
            else "—"
        )
        vr_str = (
            f"{row['violation_rate_mean']:.1%}"
            if not np.isnan(row["violation_rate_mean"])
            else "—"
        )
        print(
            f"| {row['ablation']} | {row['retention_mean']:.1f}% | {f1_str} | {vr_str} |"
        )

    # Write markdown
    with open(output_dir / f"ablation_{dataset_id}_summary.md", "w") as f:
        f.write(f"# Ablation Study: {dataset_id}\n\n")
        f.write(f"Generator: {generator_type} | Seeds: {n_seeds} | Rows: {n_rows}\n\n")
        f.write("| Ablation | Retention% | F1 (mean ± SEM) | Violation Rate |\n")
        f.write("|---|---|---|---|\n")
        for _, row in summary.iterrows():
            f1_str = (
                f"{row['f1_mean']:.3f} ± {row['f1_sem']:.3f}"
                if not np.isnan(row["f1_mean"])
                else "—"
            )
            vr_str = (
                f"{row['violation_rate_mean']:.1%}"
                if not np.isnan(row["violation_rate_mean"])
                else "—"
            )
            f.write(
                f"| {row['ablation']} | {row['retention_mean']:.1f}% | {f1_str} | {vr_str} |\n"
            )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HIF Component Ablation Study")
    parser.add_argument("--dataset", default="census_acs")
    parser.add_argument("--target", default="household_income")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--generator", default="gaussian_copula", choices=["gaussian_copula", "ctgan"]
    )
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    run_ablation(
        dataset_id=args.dataset,
        target=args.target,
        n_rows=args.rows,
        n_seeds=args.seeds,
        generator_type=args.generator,
        output_dir=Path(args.output_dir),
    )
