"""
Experiment: Privacy Filtering Audit - TAMIS Privacy Preservation under HIF Filtering
(ported to current API).

Q: "Does filtering out logically inconsistent synthetic rows increase or decrease privacy risk?"

Flow:
  1. Load dataset and split into train/test holdouts
  2. Train generator on real train data
  3. Generate synthetic data
  4. Perform HIF auditing and filter cohort (Full vs Rule-Only vs HIF Oracle)
  5. Evaluate TAMIS Privacy Oracle (MIA AUC, Linkability Risk, Exact Copies) across variants

Run:
    python scripts/09_privacy_filtering.py --dataset supermarket_sales --generator gaussian --seeds 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ruff: noqa: E402
from _exp_utils import audit_hif, generate, load_real
from tabular_polygraph.privacy import privacy_audit


def run_privacy_filtering(real_full, gen_type, rows, seed, epochs):
    train_size = int(len(real_full) * 0.7)
    train_df = real_full.iloc[:train_size].reset_index(drop=True)
    test_df = real_full.iloc[train_size:].reset_index(drop=True)

    syn_full = generate(train_df, rows, seed, gen_type, epochs=epochs)

    # HIF Audit (seeded for reproducibility)
    hif_res = audit_hif(train_df, syn_full, seed=seed)
    row_penalties = hif_res["row_penalties"]
    rule_violation_mask = hif_res.get(
        "rule_violation_mask", np.zeros(len(syn_full), dtype=bool)
    )

    # Variants:
    # 1. Full synthetic
    syn_full_var = syn_full.copy()

    # 2. Rule-only baseline (drop rule violations)
    valid_rule_indices = np.where(~rule_violation_mask)[0]
    syn_rule_var = (
        syn_full.iloc[valid_rule_indices].reset_index(drop=True)
        if len(valid_rule_indices) > 0
        else syn_full
    )

    # 3. HIF Oracle (drop rows with penalty > 0.5)
    valid_hif_indices = np.where(row_penalties < 0.5)[0]
    syn_hif_var = (
        syn_full.iloc[valid_hif_indices].reset_index(drop=True)
        if len(valid_hif_indices) > 0
        else syn_full
    )

    variants = {
        "Full synthetic": syn_full_var,
        "Rule-only Baseline": syn_rule_var,
        "HIF Oracle": syn_hif_var,
    }

    results = {}
    for var_name, syn_var in variants.items():
        audit = privacy_audit(
            real=train_df,
            synthetic=syn_var,
            real_holdout=test_df,
            n_attacks=min(200, len(syn_var)),
            seed=seed,
        )
        retention = float(len(syn_var) / len(syn_full))
        mia_auc = audit.get("membership_inference", {}).get("attack_auc", 0.5)
        link_risk = audit.get("linkability", {}).get("risk_level", "low")
        exact_copies = audit.get("exact_copies", {}).get("count", 0)

        results[var_name] = {
            "retention_pct": round(retention * 100, 1),
            "mia_auc": round(mia_auc, 4),
            "exact_copies": exact_copies,
            "linkability_risk": link_risk,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Privacy filtering audit")
    parser.add_argument("--dataset", type=str, default="supermarket_sales")
    parser.add_argument("--generator", type=str, default="gaussian")
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_full = load_real(args.dataset)
    print(
        f"Privacy Filtering Audit: {args.generator} on {args.dataset} ({args.seeds} seeds)"
    )

    all_res = []
    for s in range(42, 42 + args.seeds):
        res = run_privacy_filtering(
            real_full, args.generator, args.rows, s, args.epochs
        )
        for var_name, metrics in res.items():
            all_res.append(
                {
                    "dataset": args.dataset,
                    "generator": args.generator,
                    "variant": var_name,
                    "seed": s,
                    **metrics,
                }
            )

    df = pd.DataFrame(all_res)
    print("\n" + "=" * 70)
    print("PRIVACY AUDIT UNDER FILTERING (Mean across seeds)")
    print("=" * 70)
    summary = (
        df.groupby("variant")
        .agg(
            {
                "retention_pct": "mean",
                "mia_auc": "mean",
                "exact_copies": "mean",
            }
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    csv_path = out_dir / "privacy_filtering_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved privacy filtering results to {csv_path}")


if __name__ == "__main__":
    main()
