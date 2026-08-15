"""
Experiment: External Arithmetic-Identity Verification of HIF-Flagged Records (Q1).

Answers: on real generator cohorts (no injected corruption), what fraction of
HIF-flagged records are independently confirmed as invalid via an externally
verifiable arithmetic identity, and what is the rate among unflagged records?

Verifiable identities (hold exactly in the real data to ~1e-13 relative error):
  - Supermarket Sales:  total = unit_price * quantity * 1.05
                        total = cogs + gross_income
  - Online Purchases:   item_total = item_subtotal + item_tax
                        item_subtotal = purchase_price * quantity

A synthetic record is "confirmed invalid" if its identity relative error
exceeds a strict tolerance (1e-3, ~1e10x above the real-data residual).

Run:
    python scripts/09_arithmetic_identity_verification.py --seeds 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ruff: noqa: E402
from _exp_utils import audit_hif, generate, load_real

TOLERANCE = 1e-3

DATASET_IDENTITIES = {
    "supermarket_sales": {
        "n": 2000,
        "cols": ["total", "unit_price", "quantity", "cogs", "gross_income"],
        "error": lambda d: np.maximum(
            (d["total"] - d["unit_price"] * d["quantity"] * 1.05).abs()
            / (d["total"].abs() + 1e-9),
            (d["total"] - d["cogs"] - d["gross_income"]).abs()
            / (d["total"].abs() + 1e-9),
        ),
    },
    "online_purchases": {
        "n": 664,
        "cols": [
            "item_total",
            "item_subtotal",
            "item_tax",
            "purchase_price",
            "quantity",
        ],
        "error": lambda d: np.maximum(
            (d["item_total"] - (d["item_subtotal"] + d["item_tax"])).abs()
            / (d["item_total"].abs() + 1e-9),
            (d["item_subtotal"] - d["purchase_price"] * d["quantity"]).abs()
            / (d["item_subtotal"].abs() + 1e-9),
        ),
    },
}

GENERATORS = ["gaussian", "vine", "ctgan", "tvae"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "generator"])
        .agg(
            flag_rate=("flag_rate", "mean"),
            base_violation_rate=("base_violation_rate", "mean"),
            confirmation_rate_flagged=("confirmation_rate_flagged", "mean"),
            confirmation_rate_unflagged=("confirmation_rate_unflagged", "mean"),
            median_severity_flagged=("median_severity_flagged", "mean"),
            median_severity_unflagged=("median_severity_unflagged", "mean"),
            spearman_penalty_severity=("spearman_penalty_severity", "mean"),
            spearman_sem=("spearman_penalty_severity", sem),
        )
        .round(4)
        .reset_index()
    )


def run_experiment(n_seeds: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for ds_id, spec in DATASET_IDENTITIES.items():
        cols = spec["cols"]
        print(f"\n{'=' * 70}\nDataset: {ds_id}\n{'=' * 70}")

        for gen in GENERATORS:
            for seed_i in range(n_seeds):
                seed = 42 + seed_i
                real = load_real(ds_id, n=spec["n"], seed=seed).reset_index(drop=True)
                syn = generate(real, spec["n"], seed, gen)
                syn = syn[cols].reset_index(drop=True)

                hif = audit_hif(real, syn, seed=seed)
                pen = hif["row_penalties"]
                flagged = pen > 0.5
                err = spec["error"](syn)
                viol = err > TOLERANCE

                n_flagged = int(flagged.sum())
                n_unflagged = int((~flagged).sum())
                rho, _ = spearmanr(pen, err)
                rho = float(rho) if np.isfinite(rho) else np.nan

                rows.append(
                    {
                        "dataset": ds_id,
                        "generator": gen,
                        "seed": seed,
                        "n_rows": len(syn),
                        "flag_rate": round(n_flagged / len(syn), 4),
                        "base_violation_rate": round(float(viol.mean()), 4),
                        "confirmation_rate_flagged": round(
                            float(viol[flagged].mean()) if n_flagged else np.nan, 4
                        ),
                        "confirmation_rate_unflagged": round(
                            float(viol[~flagged].mean()) if n_unflagged else np.nan, 4
                        ),
                        "n_flagged": n_flagged,
                        "n_unflagged": n_unflagged,
                        "median_severity_flagged": round(
                            float(np.median(err[flagged])) if n_flagged else np.nan, 4
                        ),
                        "median_severity_unflagged": round(
                            float(np.median(err[~flagged])) if n_unflagged else np.nan,
                            4,
                        ),
                        "spearman_penalty_severity": round(rho, 4),
                    }
                )
            print(f"  {gen}: done ({n_seeds} seeds)")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "identity_verification.csv", index=False)

    summary = summarize(df)
    summary.to_csv(output_dir / "identity_verification_summary.csv", index=False)

    print("\n\n" + "=" * 100)
    print("External Arithmetic-Identity Verification (mean over seeds)")
    print("=" * 100)
    for _, r in summary.iterrows():
        print(
            f"| {r['dataset']:18s} | {r['generator']:8s} | flag={r['flag_rate']:.3f} "
            f"| base_viol={r['base_violation_rate']:.3f} "
            f"| confirm@flagged={r['confirmation_rate_flagged']:.3f} "
            f"| confirm@unflagged={r['confirmation_rate_unflagged']:.3f} "
            f"| rho={r['spearman_penalty_severity']:.3f}±{r['spearman_sem']:.3f} |"
        )
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External identity verification (Q1)")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_experiment(n_seeds=args.seeds, output_dir=Path(args.output_dir))
