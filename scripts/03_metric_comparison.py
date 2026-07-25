"""
Comparison experiment: HIF vs distributional metrics.

Q:"Do standard distribution matching metrics miss logical hallucinations?"

Runs 3 generators × 5 datasets × N seeds and computes:
  - KS (marginal fidelity)
  - Correlation distance (joint fidelity)
  - alpha-precision / beta-recall (Alaa et al., ICML 2022)
  - HIF (structural integrity)

Goal: determine whether HIF catches failures that distributional metrics miss.

python scripts/03_metric_comparison.py --rows 500 --seeds 1 --epochs 20
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tabular_polygraph.fidelity import (  # noqa: E402
    alpha_precision_beta_recall,
    hif_score,
    ks_distribution_scores,
)
from tabular_polygraph.generators import (  # noqa: E402
    CTGANGenerator,
    GaussianCopulaGenerator,
    TVAEGenerator,
)

try:
    from tabular_polygraph.generators import VineCopulaGenerator

    HAS_VINE = True
except ImportError:
    HAS_VINE = False

# ── Correlation distance (numeric only) ─────────────────────────────────────


def _numeric_correlation_distance(real: pd.DataFrame, syn: pd.DataFrame) -> float:
    """Spearman correlation distance on numeric columns only."""
    num_real = real.select_dtypes(include=[np.number])
    num_syn = syn.select_dtypes(include=[np.number])
    common = list(set(num_real.columns) & set(num_syn.columns))
    if len(common) < 2:
        return np.nan
    corr_real = num_real[common].corr(method="spearman").values
    corr_syn = num_syn[common].corr(method="spearman").values
    return float(np.linalg.norm(corr_real - corr_syn, "fro"))


# ── Main experiment ──────────────────────────────────────────────────────────


def _sanitize_categoricals(real: pd.DataFrame, syn: pd.DataFrame) -> pd.DataFrame:
    """Replace categorical values not present in real data with a random valid category."""
    syn = syn.copy()
    for col in real.select_dtypes(include=["object", "category"]).columns:
        valid = real[col].dropna().unique()
        mask = ~syn[col].isin(valid)
        if mask.any():
            syn.loc[mask, col] = np.random.choice(valid, size=mask.sum())
    return syn


def run_single(real_df, gen_type, rows, seed, epochs):
    np.random.seed(seed)
    if gen_type == "gaussian":
        gen = GaussianCopulaGenerator()
    elif gen_type == "vine":
        if not HAS_VINE:
            raise ImportError("Vine requires: pip install .[vine]")
        gen = VineCopulaGenerator()
    elif gen_type == "ctgan":
        gen = CTGANGenerator(epochs=epochs, batch_size=min(100, len(real_df)))
    elif gen_type == "tvae":
        gen = TVAEGenerator(epochs=epochs)
    else:
        raise ValueError(f"Unknown generator: {gen_type}")

    gen.fit(real_df)
    syn = gen.generate(rows, seed=seed).drop(columns=["syn_id"], errors="ignore")
    syn = _sanitize_categoricals(real_df, syn)
    return syn


def compute_metrics(real, syn):
    results = {}

    # KS (mean across all columns)
    ks = ks_distribution_scores(real, syn)
    results["ks_score"] = float(np.mean(list(ks.values()))) if ks else np.nan

    # Correlation distance
    results["corr_distance"] = _numeric_correlation_distance(real, syn)

    # Alpha-precision / beta-recall
    ap_br = alpha_precision_beta_recall(real, syn)
    results["alpha_precision"] = ap_br["alpha_precision"]
    results["beta_recall"] = ap_br["beta_recall"]
    results["authenticity"] = ap_br["authenticity"]

    # HIF
    cat_cols = real.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        hif_res = hif_score(real, syn, verbose=False, hif_epochs=5)
        results["hif_score"] = hif_res["hif_score"]
        results["hif_violation_rate"] = hif_res["violation_rate"]
        results["hif_rule_violation_rate"] = hif_res["rule_violation_rate"]
    else:
        results["hif_score"] = np.nan
        results["hif_violation_rate"] = np.nan

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["supermarket_sales", "online_purchases", "credit", "adult"]
    generators = ["gaussian", "vine", "ctgan", "tvae"]
    seeds = list(range(42, 42 + args.seeds))

    print(
        f"Experiment: {len(generators)} generators × {len(datasets)} datasets × {len(seeds)} seeds"
    )
    print(f"Rows per experiment: {args.rows}, CTGAN/TVAE epochs: {args.epochs}")

    all_results = []

    from tabular_polygraph.dataset import load_dataset

    for ds_id in datasets:
        try:
            real_full = load_dataset(ds_id)
        except Exception as err_load:
            print(f"  Skipping {ds_id} (error loading: {err_load})")
            continue

        drop_cols = [c for c in real_full.columns if real_full[c].isnull().mean() > 0.3]
        real_full = real_full.drop(columns=drop_cols, errors="ignore")

        print(f"\n{'=' * 70}")
        print(
            f"  Dataset: {ds_id} ({len(real_full)} rows × {len(real_full.columns)} cols)"
        )
        print(f"{'=' * 70}")

        for gen_type in generators:
            if gen_type == "vine" and not HAS_VINE:
                print("\n  ── vine (skipped: pyvinecopulib not installed) ──")
                continue
            print(f"\n  ── {gen_type} ──")
            for seed in seeds:
                real = real_full.sample(
                    min(args.rows, len(real_full)), random_state=seed
                ).reset_index(drop=True)

                try:
                    t0 = time.time()
                    syn = run_single(real, gen_type, len(real), seed, args.epochs)
                    metrics = compute_metrics(real, syn)
                    dt = time.time() - t0

                    row = {
                        "dataset": ds_id,
                        "generator": gen_type,
                        "seed": seed,
                        "rows": len(real),
                        **metrics,
                        "time_s": round(dt, 1),
                    }
                    all_results.append(row)

                    print(
                        f"    seed={seed}  "
                        f"KS={metrics['ks_score']:.1f}  "
                        f"α={metrics['alpha_precision']:.3f}  "
                        f"β={metrics['beta_recall']:.3f}  "
                        f"HIF={metrics.get('hif_score', 'N/A')}  "
                        f"Viol={metrics.get('hif_violation_rate', 'N/A')}  "
                        f"({dt:.0f}s)"
                    )
                except Exception as e:
                    print(f"    seed={seed}  ERROR: {e}")
                    all_results.append(
                        {
                            "dataset": ds_id,
                            "generator": gen_type,
                            "seed": seed,
                            "error": str(e),
                        }
                    )

    df = pd.DataFrame(all_results)
    csv_path = out_dir / "metric_comparison.csv"
    df.to_csv(csv_path, index=False)

    # Summary table
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY: Mean ± Std across seeds")
    print(f"{'=' * 70}")
    metric_cols = [
        "ks_score",
        "alpha_precision",
        "beta_recall",
        "authenticity",
        "hif_score",
        "hif_violation_rate",
    ]
    summary_rows = []
    for ds_id in df["dataset"].unique():
        for gen_type in df["generator"].unique():
            sub = df[(df["dataset"] == ds_id) & (df["generator"] == gen_type)]
            if sub.empty:
                continue
            row = {"dataset": ds_id, "generator": gen_type}
            for m in metric_cols:
                if m in sub.columns and sub[m].notna().any():
                    vals = sub[m].dropna()
                    row[f"{m}_mean"] = round(vals.mean(), 3)
                    row[f"{m}_std"] = round(vals.std(), 3)
                else:
                    row[f"{m}_mean"] = "N/A"
                    row[f"{m}_std"] = "N/A"
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Correlation analysis
    print(f"\n{'=' * 70}")
    print("  CORRELATION: HIF vs distributional metrics")
    print(f"{'=' * 70}")
    for m in ["ks_score", "alpha_precision", "beta_recall", "corr_distance"]:
        valid = df[["hif_score", m]].dropna()
        if len(valid) > 3:
            corr = valid["hif_score"].corr(valid[m])
            print(f"  HIF vs {m}: ρ = {corr:.3f}")

    print(f"\n  Results saved → {csv_path}")


if __name__ == "__main__":
    main()
