"""
Experiment: Semantic Sensitivity vs. Statistical Dilution.

This script demonstrates that the Hybrid Integrity Framework (HIF) identifies
'Logical Viruses' (hallucinations) that aggregate distributional metrics miss
when they are sparse (low-p).

The Continuous Semantic Severity Penalty (CSSP) provides a deterministic signal
for combinatorial ruptures, even when they occupy <1% of the generated volume.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ruff: noqa: E402
PROJECT_ROOT = Path(__file__).resolve().parents[1]


from tabular_polygraph.dataset import load_dataset
from tabular_polygraph.fidelity import hif_score


def joint_correlation_distance(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Measure the distance between two correlation matrices."""
    # Only numeric columns
    num_cols = [c for c in df1.columns if pd.api.types.is_numeric_dtype(df1[c])]
    c1 = df1[num_cols].corr().fillna(0).values.flatten()
    c2 = df2[num_cols].corr().fillna(0).values.flatten()
    return np.linalg.norm(c1 - c2)


def run_sensitivity_test(p_hallucination: float = 0.01):
    print(
        f"\n[Sensitivity Test] Injecting {p_hallucination * 100:.1f}% Hallucinations..."
    )

    # 1. Load Real Data (Adult dataset)
    real_df = load_dataset("census_acs", n=5000).dropna()

    # 2. Perfect Distributional Mimic (Baseline)
    # We use the real data itself as the 'perfect' synthetic data
    syn_df = real_df.copy()

    # 3. Inject Semantic Hallucinations
    # Corrupt categorical columns by randomly swapping values
    n_corrupt = int(len(syn_df) * p_hallucination)
    if n_corrupt > 0:
        rng = np.random.default_rng(42)
        cat_cols = [
            c for c in syn_df.columns if not pd.api.types.is_numeric_dtype(syn_df[c])
        ]
        corrupt_idx = rng.choice(syn_df.index, size=n_corrupt, replace=False)
        for col in cat_cols:
            pool = real_df[col].dropna().unique()
            syn_df.loc[corrupt_idx, col] = rng.choice(
                pool, size=n_corrupt, replace=True
            )

    # 4. Measure Metrics
    print("  Evaluating HIF...")
    hif_res = hif_score(real_df, syn_df, verbose=False)
    hif_val = hif_res["hif_score"]

    print("  Evaluating JCD...")
    jcd_val = joint_correlation_distance(real_df, syn_df)

    return {
        "p_hallucination": p_hallucination,
        "hif_integrity": hif_val,
        "jcd_distance": jcd_val,
    }


if __name__ == "__main__":
    np.random.seed(42)

    results = []
    for p in [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]:
        res = run_sensitivity_test(p)
        results.append(res)

    df = pd.DataFrame(results)

    # Normalize for comparison (Inverse HIF change vs JCD change)
    hif_0 = df.loc[df["p_hallucination"] == 0, "hif_integrity"].iloc[0]
    df["hif_drop"] = (hif_0 - df["hif_integrity"]) / hif_0 if hif_0 > 0 else 0

    print("\n" + "=" * 72)
    print("SENSITIVITY ANALYSIS: SEMANTIC INTEGRITY VS. JOINT CORRELATION")
    print("=" * 72)
    print(df[["p_hallucination", "hif_integrity", "jcd_distance", "hif_drop"]])
    print("=" * 72)

    # Save for appendix
    output_path = Path("outputs/sensitivity_analysis.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
