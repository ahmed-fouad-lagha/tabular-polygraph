"""
Experiment 2: Adversarial synthetic data — globally good, locally broken.

Create a synthetic dataset where:
- Marginals match perfectly (KS ≈ 0)
- Correlations match perfectly
- BUT specific rows violate logical rules (e.g., age=25 AND occupation=Retired)

This is the strongest proof that aggregate metrics MISS row-level failures.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

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


def create_adversarial_synthetic(
    real: pd.DataFrame, n: int = 3000, frac_bad: float = 0.3, seed: int = 42
) -> pd.DataFrame:
    """
    Create synthetic data that looks good globally but has row-level rule violations.

    Strategy:
    1. Start with a copy of real data (resampled) — guarantees marginal fidelity
    2. Inject rule violations into ~frac_bad fraction of rows
    """
    rng = np.random.RandomState(seed)
    syn = real.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    n_bad = int(n * frac_bad)
    bad_idx = rng.choice(n, size=n_bad, replace=False)

    # Find categorical columns to corrupt
    cat_cols = [c for c in syn.columns if not pd.api.types.is_numeric_dtype(syn[c])]
    num_cols = [c for c in syn.columns if pd.api.types.is_numeric_dtype(syn[c])]

    # Violation 1: Shuffle categorical values within rows (breaks inter-column deps)
    for col in cat_cols:
        vals = syn[col].values.copy()
        shuffled = rng.permutation(vals)
        syn.loc[syn.index[bad_idx], col] = shuffled[bad_idx]

    # Violation 2: For numeric columns, swap values between distant rows
    for col in num_cols:
        vals = syn[col].values.copy()
        permuted = rng.permutation(vals)
        syn.loc[syn.index[bad_idx], col] = permuted[bad_idx]

    return syn


def compute_aggregate_verdict(real, syn, num_cols):
    """Simulate a typical aggregate-only evaluation."""
    corr = correlation_distance_score(real, syn, num_cols)
    mm = moment_matching_scores(real, syn, num_cols)
    mm_mean = mean_moment_matching_score(mm)

    ks_scores = []
    for col in num_cols:
        s, _ = ks_2samp(real[col].dropna(), syn[col].dropna())
        ks_scores.append(1 - s)
    ks_mean = np.mean(ks_scores) if ks_scores else 0

    try:
        real_sub = real.sample(n=len(syn), random_state=42)
        ab = alpha_precision_beta_recall(real_sub, syn)
        alpha = ab.get("alpha_precision", 0)
    except Exception:
        alpha = 0

    composite = (
        0.3 * (corr / 100) + 0.25 * (mm_mean / 100) + 0.25 * alpha + 0.2 * ks_mean
    )
    return {
        "composite": round(composite, 4),
        "corr_dist": round(corr, 2),
        "mm_mean": round(mm_mean, 2),
        "ks_mean": round(ks_mean, 4),
        "alpha": round(alpha, 4),
    }


def run():
    print("=" * 80)
    print("EXPERIMENT 2: ADVERSARIAL — GLOBAL METRICS SAY GOOD, ROWS ARE BROKEN")
    print("=" * 80)

    real = load_dataset("adult", n=5000)
    num_cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]
    cat_cols = [c for c in real.columns if not pd.api.types.is_numeric_dtype(real[c])]
    all_cols = [c for c in real.columns if c in real.columns]

    print(f"\nReal: {real.shape} | Num: {len(num_cols)} | Cat: {len(cat_cols)}")

    # Conditions: 0%, 10%, 20%, 30%, 40%, 50% corrupted
    fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    print(
        f"\n{'frac_bad':>8} {'aggregate':>10} {'corr_dist':>10} {'ks_sim':>8} {'alpha':>8} {'hif':>8} {'viol_rate':>10} {'broken':>8} {'p90':>8} {'p99':>8}"
    )
    print("-" * 110)

    for frac in fracs:
        syn = create_adversarial_synthetic(real, n=3000, frac_bad=frac, seed=42)

        agg = compute_aggregate_verdict(real, syn, num_cols)
        hif = hif_score(
            real, syn, columns=all_cols, hif_hubs=5, random_state=42, verbose=False
        )

        rp = hif["row_penalties"]
        n_broken = int((rp > 0.5).sum())

        print(
            f"{frac:>8.0%} {agg['composite']:>10.4f} {agg['corr_dist']:>10.2f} {agg['ks_mean']:>8.4f} {agg['alpha']:>8.4f} "
            f"{hif['hif_score']:>8.4f} {hif['violation_rate']:>10.1%} {n_broken:>8} "
            f"{np.percentile(rp, 90):>8.4f} {np.percentile(rp, 99):>8.4f}"
        )

    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("  If aggregate stays ~constant while violation rate climbs,")
    print("  that proves aggregate metrics CANNOT detect row-level corruption.")
    print("=" * 80)


if __name__ == "__main__":
    run()
