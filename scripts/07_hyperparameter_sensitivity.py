"""
Example 07: Hyperparameter Sensitivity Analysis.

Sweeps three HIF hyperparameters on Census ACS data:
  1. Hub count K:          {1, 3, 5, 10, 15}
  2. Confidence percentile: {1, 3, 5, 10}
  3. Violation threshold:   {0.3, 0.5, 0.7}

Uses N=10 seeds per configuration and reports mean +/- SEM.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tabular_polygraph.dataset import load_dataset
from tabular_polygraph.fidelity import hif_score
from tabular_polygraph.utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_single_config(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    hif_hubs: int,
    confidence_percentile: float,
    violation_threshold: float,
    seed: int,
) -> dict:
    set_seed(seed)
    res = hif_score(
        real_df,
        syn_df,
        hif_hubs=hif_hubs,
        confidence_percentile=confidence_percentile,
        violation_threshold=violation_threshold,
        random_state=seed,
        verbose=False,
    )
    return {
        "hif_score": res["hif_score"],
        "violation_rate": res["violation_rate"],
        "mean_penalty": res["mean_penalty"],
        "lse_violation_rate": res["lse_violation_rate"],
        "nic_violation_rate": res["nic_violation_rate"],
        "rule_violation_rate": res["rule_violation_rate"],
    }


def make_corrupted_synthetic(
    real_df: pd.DataFrame, p_hallucination: float = 0.05, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    syn_df = real_df.copy()
    n_corrupt = int(len(syn_df) * p_hallucination)
    if n_corrupt > 0:
        cat_cols = [
            c for c in syn_df.columns if not pd.api.types.is_numeric_dtype(syn_df[c])
        ]
        corrupt_idx = rng.choice(syn_df.index, size=n_corrupt, replace=False)
        for col in cat_cols:
            pool = real_df[col].dropna().unique()
            syn_df.loc[corrupt_idx, col] = rng.choice(
                pool, size=n_corrupt, replace=True
            )
    return syn_df


if __name__ == "__main__":
    N_SEEDS = 5
    BASE_SEED = 42
    N_RECORDS = 2000

    print("[Sensitivity] Loading Census ACS data...")
    real_df = load_dataset("census_acs", n=N_RECORDS).dropna()
    syn_df = make_corrupted_synthetic(real_df, p_hallucination=0.05, seed=BASE_SEED)
    print(f"[Sensitivity] Real: {real_df.shape}, Synthetic: {syn_df.shape}")

    hub_counts = [1, 3, 5, 10, 15]
    percentiles = [1, 3, 5, 10]
    thresholds = [0.3, 0.5, 0.7]

    all_results = []

    # Sweep 1: Hub count K
    print("\n[Sweep 1] Hub count K...")
    for k in hub_counts:
        scores, rates = [], []
        for s in range(N_SEEDS):
            r = run_single_config(real_df, syn_df, k, 5.0, 0.5, BASE_SEED + s)
            scores.append(r["hif_score"])
            rates.append(r["violation_rate"])
        all_results.append(
            {
                "sweep": "hub_count",
                "param": "K",
                "value": k,
                "hif_score_mean": np.mean(scores),
                "hif_score_se": np.std(scores, ddof=1) / np.sqrt(N_SEEDS),
                "violation_rate_mean": np.mean(rates),
                "violation_rate_se": np.std(rates, ddof=1) / np.sqrt(N_SEEDS),
            }
        )
        print(
            f"  K={k:>2d}: HIF={np.mean(scores):.4f} +/- {np.std(scores, ddof=1) / np.sqrt(N_SEEDS):.4f}"
        )

    # Sweep 2: Confidence percentile
    print("\n[Sweep 2] Confidence floor percentile...")
    for p in percentiles:
        scores, rates = [], []
        for s in range(N_SEEDS):
            r = run_single_config(real_df, syn_df, 5, float(p), 0.5, BASE_SEED + s)
            scores.append(r["hif_score"])
            rates.append(r["violation_rate"])
        all_results.append(
            {
                "sweep": "confidence_percentile",
                "param": "percentile",
                "value": p,
                "hif_score_mean": np.mean(scores),
                "hif_score_se": np.std(scores, ddof=1) / np.sqrt(N_SEEDS),
                "violation_rate_mean": np.mean(rates),
                "violation_rate_se": np.std(rates, ddof=1) / np.sqrt(N_SEEDS),
            }
        )
        print(
            f"  P={p:>2d}: HIF={np.mean(scores):.4f} +/- {np.std(scores, ddof=1) / np.sqrt(N_SEEDS):.4f}"
        )

    # Sweep 3: Violation threshold
    print("\n[Sweep 3] Violation threshold...")
    for t in thresholds:
        scores, rates = [], []
        for s in range(N_SEEDS):
            r = run_single_config(real_df, syn_df, 5, 5.0, t, BASE_SEED + s)
            scores.append(r["hif_score"])
            rates.append(r["violation_rate"])
        all_results.append(
            {
                "sweep": "violation_threshold",
                "param": "threshold",
                "value": t,
                "hif_score_mean": np.mean(scores),
                "hif_score_se": np.std(scores, ddof=1) / np.sqrt(N_SEEDS),
                "violation_rate_mean": np.mean(rates),
                "violation_rate_se": np.std(rates, ddof=1) / np.sqrt(N_SEEDS),
            }
        )
        print(
            f"  T={t:.1f}: HIF={np.mean(scores):.4f} +/- {np.std(scores, ddof=1) / np.sqrt(N_SEEDS):.4f}"
        )

    df = pd.DataFrame(all_results)
    out_path = PROJECT_ROOT / "results" / "hyperparameter_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[Done] Results saved to {out_path}")
    print(df.to_string(index=False))
