"""
Experiment: Component Ablation Study (ported to current API).

Tests each HIF component individually and compares against the full ensemble,
plus arithmetic vs geometric aggregation.

Run:
    python scripts/06_component_ablation_study.py --dataset census_acs --rows 2000 \
      --seeds 5 --generator gaussian_copula
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ruff: noqa: E402
from _exp_utils import audit_hif, generate, load_real, utility_metrics

ABLATION_CONFIGS = [
    {"name": "LSE-only", "ablation_mode": "lse_only", "aggregation": "geometric"},
    {"name": "NIC-only", "ablation_mode": "nic_only", "aggregation": "geometric"},
    {"name": "Rules-only", "ablation_mode": "rules_only", "aggregation": "geometric"},
    {"name": "LSE + NIC", "ablation_mode": "lse_nic", "aggregation": "geometric"},
    {
        "name": "Full HIF (multiplicative)",
        "ablation_mode": "full",
        "aggregation": "geometric",
    },
    {"name": "Full HIF (arith.)", "ablation_mode": "full", "aggregation": "arithmetic"},
]


def run_ablation(
    dataset_id: str,
    target: str,
    n_rows: int,
    n_seeds: int,
    generator_type: str,
    output_dir: Path,
):
    print(f"Loading dataset: {dataset_id}")
    real = load_real(dataset_id, n=n_rows)
    print(f"Fitting generator: {generator_type}")

    all_results = []
    for seed_i in range(n_seeds):
        seed = 42 + seed_i
        print(f"\n{'=' * 60}\nSeed {seed} ({seed_i + 1}/{n_seeds})\n{'=' * 60}")

        syn = generate(real, n_rows, seed, generator_type)
        syn = syn[real.columns.intersection(syn.columns).tolist()]

        util_full = utility_metrics(real, syn, target, seed)
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
                result = audit_hif(
                    real,
                    syn,
                    seed=seed,
                    ablation_mode=config["ablation_mode"],
                    aggregation=config["aggregation"],
                )
                penalties = result["row_penalties"]
                mask = penalties < 0.5
                syn_filtered = syn[mask]
                retention = len(syn_filtered) / len(syn) * 100

                util = (
                    utility_metrics(real, syn_filtered, target, seed)
                    if len(syn_filtered) > 10
                    else {"f1": np.nan, "accuracy": np.nan, "trr": np.nan}
                )
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

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df["generator"] = generator_type
    df.to_csv(
        output_dir / f"ablation_{dataset_id}_{generator_type}_raw.csv", index=False
    )

    summary = (
        df.groupby("ablation")
        .agg(
            retention_mean=("retention_pct", "mean"),
            f1_mean=("f1", "mean"),
            f1_sem=("f1", sem),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sem=("accuracy", sem),
            violation_rate_mean=("violation_rate", "mean"),
            hif_score_mean=("hif_score", "mean"),
        )
        .round(4)
        .reset_index()
    )

    order = [
        "No filtering",
        "LSE-only",
        "NIC-only",
        "Rules-only",
        "LSE + NIC",
        "Full HIF (arith.)",
        "Full HIF (multiplicative)",
    ]
    summary["order"] = summary["ablation"].map({o: i for i, o in enumerate(order)})
    summary = summary.sort_values("order").drop(columns="order")

    summary.to_csv(
        output_dir / f"ablation_{dataset_id}_{generator_type}_summary.csv", index=False
    )

    print("\n\n" + "=" * 80)
    print(f"Component Ablation ({dataset_id}, {generator_type}, N={n_seeds})")
    print("=" * 80)
    print(
        "| Ablation | Retention (%) | F1 (mean ± SEM) | Accuracy (mean ± SEM) | Violation Rate |"
    )
    print("|---|---|---|---|---|")
    for _, row in summary.iterrows():
        print(
            f"| {row['ablation']} | {row['retention_mean']:.1f} | "
            f"{row['f1_mean']:.3f} ± {row['f1_sem']:.3f} | "
            f"{row['accuracy_mean']:.3f} ± {row['accuracy_sem']:.3f} | "
            f"{row['violation_rate_mean'] * 100:.1f}% |"
        )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Component Ablation Study")
    parser.add_argument("--dataset", default="census_acs")
    parser.add_argument("--target", default="household_income")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--generator",
        default="gaussian_copula",
        choices=["gaussian_copula", "ctgan", "tvae", "vine"],
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
