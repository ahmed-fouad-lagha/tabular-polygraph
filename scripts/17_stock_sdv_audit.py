"""
Experiment: Stock SDV Gaussian Copula Audit on Constrained Domains (Q1).

Q: How does a stock, deployed synthesizer (SDV GaussianCopulaSynthesizer,
which models categorical-numeric dependence via Gaussianization) behave on
the arithmetic-constraint domains, relative to the repo's custom arms?

For each seed: fit SDV's synthesizer on the real cohort, sample len(real)
rows, then (a) compute the Table-1 metric suite and the full-feature HIF
audit, and (b) run the identity-verification projection protocol of
script 09.

Run:
    python scripts/17_stock_sdv_audit.py --seeds 10
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
from _exp_utils import aggregate_metrics, audit_hif, load_real

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


def generate_sdv_gaussian(real: pd.DataFrame) -> pd.DataFrame:
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer

    from tabular_polygraph._utils import set_seed

    set_seed(0)
    meta = Metadata.detect_from_dataframe(data=real, table_name="t")
    synth = GaussianCopulaSynthesizer(meta)
    synth.fit(real)
    set_seed(0)
    syn = synth.sample(len(real))
    return syn[real.columns.intersection(syn.columns).tolist()]


def run_experiment(n_seeds: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for ds_id, spec in DATASET_IDENTITIES.items():
        cols = spec["cols"]
        print(f"\n{'=' * 70}\nDataset: {ds_id}\n{'=' * 70}")
        for seed_i in range(n_seeds):
            seed = 42 + seed_i
            print(f"  Seed {seed} ({seed_i + 1}/{n_seeds})")
            real = load_real(ds_id, n=spec["n"], seed=seed).reset_index(drop=True)
            syn_full = generate_sdv_gaussian(real)

            agg = aggregate_metrics(real, syn_full, seed)
            hif = audit_hif(real, syn_full, seed=seed)

            # Identity verification on the projection onto identity columns.
            syn_proj = syn_full[cols].reset_index(drop=True)
            hif_proj = audit_hif(real, syn_proj, seed=seed)
            pen_proj = hif_proj["row_penalties"]
            flagged = pen_proj > 0.5
            err = spec["error"](syn_proj)
            viol = err > TOLERANCE
            n_flagged = int(flagged.sum())
            n_unflagged = int((~flagged).sum())
            rho, _ = spearmanr(pen_proj, err)
            rho = float(rho) if np.isfinite(rho) else np.nan

            rows.append(
                {
                    "dataset": ds_id,
                    "generator": "sdv_gaussian_copula",
                    "seed": seed,
                    "ks": agg.get("ks", np.nan),
                    "alpha_precision": agg.get("alpha_precision", np.nan),
                    "beta_recall": agg.get("beta_recall", np.nan),
                    "hif_score": hif.get("hif_score", np.nan),
                    "violation_rate": hif.get("violation_rate", np.nan),
                    "lse_violation_rate": hif.get("lse_violation_rate", np.nan),
                    "nic_violation_rate": hif.get("nic_violation_rate", np.nan),
                    "rule_violation_rate": hif.get("rule_violation_rate", np.nan),
                    "flag_rate": round(n_flagged / len(syn_proj), 4),
                    "base_violation_rate": round(float(viol.mean()), 4),
                    "confirmation_rate_flagged": round(
                        float(viol[flagged].mean()) if n_flagged else np.nan, 4
                    ),
                    "confirmation_rate_unflagged": round(
                        float(viol[~flagged].mean()) if n_unflagged else np.nan, 4
                    ),
                    "spearman_penalty_severity": round(rho, 4),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "stock_sdv_audit_raw.csv", index=False)

    summary = (
        df.groupby(["dataset", "generator"])
        .agg(
            ks_mean=("ks", "mean"),
            ks_sd=("ks", "std"),
            alpha_mean=("alpha_precision", "mean"),
            alpha_sd=("alpha_precision", "std"),
            beta_mean=("beta_recall", "mean"),
            beta_sd=("beta_recall", "std"),
            hif_mean=("hif_score", "mean"),
            hif_sd=("hif_score", "std"),
            viol_mean=("violation_rate", "mean"),
            viol_sd=("violation_rate", "std"),
            flag_rate=("flag_rate", "mean"),
            base_viol=("base_violation_rate", "mean"),
            conf_flagged=("confirmation_rate_flagged", "mean"),
            conf_unflagged=("confirmation_rate_unflagged", "mean"),
            spearman_mean=("spearman_penalty_severity", "mean"),
            spearman_sem=("spearman_penalty_severity", sem),
        )
        .round(4)
        .reset_index()
    )
    summary.to_csv(output_dir / "stock_sdv_audit_summary.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_experiment(args.seeds, Path(args.output_dir))


if __name__ == "__main__":
    main()
