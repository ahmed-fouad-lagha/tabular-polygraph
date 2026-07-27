"""
Experiment: Aggregate vs Row-Level Structural Fidelity

Core question: Can aggregate metrics give a "good" verdict while HIF reveals
that many individual rows are structurally broken?

This tests whether row-level diagnostics (HIF) provide information that
aggregate metrics (correlation distance, KS, α-precision) miss.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path("/home/lagha/repos/tabular-polygraph")
sys.path.insert(0, str(PROJECT_ROOT))

from tabular_polygraph.dataset import load_dataset  # noqa: E402
from tabular_polygraph.fidelity import (  # noqa: E402
    correlation_distance_score,
    hif_score,
    mean_moment_matching_score,
    moment_matching_scores,
)
from tabular_polygraph.fidelity.alpha_beta import (  # noqa: E402
    alpha_precision_beta_recall,
)
from tabular_polygraph.generators import (  # noqa: E402
    CTGANGenerator,
    GaussianCopulaGenerator,
    TVAEGenerator,
)


def load_real_data(dataset_id: str, n: int = 5000) -> pd.DataFrame:
    return load_dataset(dataset_id, n=n)


def generate_synthetic(
    real: pd.DataFrame, gen_type: str, rows: int, seed: int
) -> pd.DataFrame:
    if gen_type == "gaussian_copula":
        gen = GaussianCopulaGenerator()
    elif gen_type == "ctgan":
        gen = CTGANGenerator()
    elif gen_type == "tvae":
        gen = TVAEGenerator()
    else:
        raise ValueError(f"Unknown generator: {gen_type}")
    gen.fit(real)
    syn = gen.generate(rows, seed=seed)
    return syn.drop(columns=["syn_id"], errors="ignore")


def compute_structural_fidelity_score(real: pd.DataFrame, syn: pd.DataFrame) -> dict:
    """Compute a composite aggregate structural fidelity score (similar to TabStruct's approach)."""
    num_cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]

    # 1. Correlation distance (pairwise structural)
    corr_dist = correlation_distance_score(real, syn, num_cols)

    # 2. Moment matching (aggregate distributional)
    mm_scores = moment_matching_scores(real, syn, num_cols)
    mm_mean = mean_moment_matching_score(mm_scores)

    # 3. α-precision and β-recall (support coverage)
    # alpha_precision_beta_recall requires same row count — subsample real to match syn
    try:
        real_sub = real.sample(n=len(syn), random_state=42)
        ab = alpha_precision_beta_recall(real_sub, syn, num_cols)
        alpha = ab.get("alpha_precision", 0)
        beta = ab.get("beta_recall", 0)
    except Exception:
        alpha, beta = 0, 0

    # 4. Pairwise KS (marginal fidelity)
    ks_scores = []
    for col in num_cols:
        if col in real.columns and col in syn.columns:
            from scipy.stats import ks_2samp

            stat, _ = ks_2samp(real[col].dropna(), syn[col].dropna())
            ks_scores.append(1 - stat)  # Convert to similarity
    ks_mean = np.mean(ks_scores) if ks_scores else 0

    # Composite score (weighted average, all components normalised to 0-1)
    aggregate_score = (
        0.3 * (corr_dist / 100) + 0.25 * (mm_mean / 100) + 0.25 * alpha + 0.2 * ks_mean
    )

    return {
        "aggregate_composite": round(aggregate_score, 4),
        "correlation_distance": round(corr_dist, 2),
        "moment_matching": round(mm_mean, 4),
        "alpha_precision": round(alpha, 4),
        "beta_recall": round(beta, 4),
        "ks_similarity": round(ks_mean, 4),
    }


def run_experiment():
    print("=" * 80)
    print("EXPERIMENT: AGGREGATE vs ROW-LEVEL STRUCTURAL FIDELITY")
    print("=" * 80)

    datasets = ["adult"]
    generators = ["gaussian_copula", "ctgan"]
    seeds = [42]
    rows = 3000

    results = []

    for dataset_id in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_id}")
        print(f"{'=' * 60}")

        real = load_real_data(dataset_id, n=5000)
        num_cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]
        cat_cols = [
            c for c in real.columns if not pd.api.types.is_numeric_dtype(real[c])
        ]
        print(
            f"  Real shape: {real.shape} | Num cols: {len(num_cols)} | Cat cols: {len(cat_cols)}"
        )

        for gen_type in generators:
            for seed in seeds:
                print(f"\n  Generator: {gen_type} | Seed: {seed}")

                try:
                    syn = generate_synthetic(real, gen_type, rows, seed)
                except Exception as e:
                    print(f"    FAILED: {e}")
                    continue

                # 1. Aggregate structural fidelity
                agg = compute_structural_fidelity_score(real, syn)

                # 2. HIF row-level
                all_cols = num_cols + cat_cols
                all_cols = [c for c in all_cols if c in syn.columns]

                hif_result = hif_score(
                    real,
                    syn,
                    columns=all_cols,
                    hif_hubs=5,
                    random_state=seed,
                    verbose=False,
                )

                hif_score_val = hif_result["hif_score"]
                violation_rate = hif_result["violation_rate"]
                row_penalties = hif_result["row_penalties"]

                # 3. Row-level statistics
                median_penalty = float(np.median(row_penalties))
                p90_penalty = float(np.percentile(row_penalties, 90))
                p99_penalty = float(np.percentile(row_penalties, 99))
                n_broken = int((row_penalties > 0.5).sum())
                n_suspicious = int((row_penalties > 0.3).sum())

                row = {
                    "dataset": dataset_id,
                    "generator": gen_type,
                    "seed": seed,
                    "real_rows": len(real),
                    "syn_rows": len(syn),
                    **agg,
                    "hif_score": round(hif_score_val, 4),
                    "hif_violation_rate": round(violation_rate, 4),
                    "median_penalty": round(median_penalty, 4),
                    "p90_penalty": round(p90_penalty, 4),
                    "p99_penalty": round(p99_penalty, 4),
                    "n_broken_rows": n_broken,
                    "n_suspicious_rows": n_suspicious,
                }
                results.append(row)

                print(f"    Aggregate composite: {agg['aggregate_composite']:.4f}")
                print(f"    HIF score:           {hif_score_val:.4f}")
                print(f"    Violation rate:      {violation_rate:.1%}")
                print(f"    Broken rows:         {n_broken}/{len(syn)}")
                print(f"    Median penalty:      {median_penalty:.4f}")
                print(f"    P90 penalty:         {p90_penalty:.4f}")
                print(f"    P99 penalty:         {p99_penalty:.4f}")

    # Analysis
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("ANALYSIS: DISCREPANCY BETWEEN AGGREGATE AND ROW-LEVEL")
    print("=" * 80)

    # Key finding: cases where aggregate says "good" but HIF says "bad"
    print("\n--- Cases where aggregate composite > 0.5 BUT violation rate > 30% ---")
    misleading = df[
        (df["aggregate_composite"] > 0.5) & (df["hif_violation_rate"] > 0.3)
    ]
    if len(misleading) > 0:
        print(
            misleading[
                [
                    "dataset",
                    "generator",
                    "aggregate_composite",
                    "hif_score",
                    "hif_violation_rate",
                    "n_broken_rows",
                ]
            ].to_string(index=False)
        )
        print(
            f"\n  >>> FOUND {len(misleading)} CASES where aggregate metrics are MISLEADING <<<"
        )
    else:
        print("  None found")

    print("\n--- Cases where aggregate composite > 0.7 BUT violation rate > 50% ---")
    misleading2 = df[
        (df["aggregate_composite"] > 0.7) & (df["hif_violation_rate"] > 0.5)
    ]
    if len(misleading2) > 0:
        print(
            misleading2[
                [
                    "dataset",
                    "generator",
                    "aggregate_composite",
                    "hif_score",
                    "hif_violation_rate",
                    "n_broken_rows",
                ]
            ].to_string(index=False)
        )
        print(f"\n  >>> FOUND {len(misleading2)} SEVERELY MISLEADING CASES <<<")
    else:
        print("  None found")

    # Correlation between aggregate and HIF
    valid = df.dropna(subset=["aggregate_composite", "hif_score"])
    if len(valid) > 2:
        rho, p = spearmanr(valid["aggregate_composite"], valid["hif_score"])
        print(
            f"\n--- Rank correlation (aggregate vs HIF): rho={rho:.3f}, p={p:.4f} ---"
        )
        if abs(rho) < 0.3:
            print(
                "  >>> WEAK CORRELATION: aggregate and HIF measure DIFFERENT things <<<"
            )
        elif abs(rho) < 0.6:
            print("  >>> MODERATE CORRELATION: aggregate and HIF partially overlap <<<")
        else:
            print("  >>> STRONG CORRELATION: aggregate and HIF largely agree <<<")

    # Show the full table
    print("\n--- Full results ---")
    print(
        df[
            [
                "dataset",
                "generator",
                "aggregate_composite",
                "correlation_distance",
                "hif_score",
                "hif_violation_rate",
                "median_penalty",
                "p90_penalty",
                "p99_penalty",
            ]
        ].to_string(index=False)
    )

    # Save
    out_path = PROJECT_ROOT / "outputs" / "aggregate_vs_rowlevel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    return df


if __name__ == "__main__":
    df = run_experiment()
