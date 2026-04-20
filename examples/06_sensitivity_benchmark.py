"""
Example 06: Linear Sensitivity vs. Structural Dilution.

This script proves that HIF catches 'Logical Viruses' (hallucinations) that
standard joint statistical metrics (like Joint Correlation Distance) miss
when they are sparse (low-p).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ruff: noqa: E402
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.catalog import load_dataset
from src.fidelity import hif_score
from scipy.spatial.distance import correlation

def joint_correlation_distance(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Measure the distance between two correlation matrices."""
    # Only numeric columns
    num_cols = [c for c in df1.columns if pd.api.types.is_numeric_dtype(df1[c])]
    c1 = df1[num_cols].corr().fillna(0).values.flatten()
    c2 = df2[num_cols].corr().fillna(0).values.flatten()
    return np.linalg.norm(c1 - c2)

def run_sensitivity_test(p_hallucination: float = 0.01):
    print(f"\n[Sensitivity Test] Injecting {p_hallucination*100:.1f}% Hallucinations...")
    
    # 1. Load Real Data
    real_df = load_dataset("census_acs", n=2000).dropna()
    
    # 2. Perfect Distributional Mimic (Baseline)
    # We use the real data itself as the 'perfect' synthetic data
    syn_df = real_df.copy()
    
    # 3. Inject 'Impossible Tracts' (HIF Hallucinations)
    # Law: owner_occupied + total_renter_units <= total_housing_units
    # Corruption: set owner_occupied = total_housing_units * 2
    n_corrupt = int(len(syn_df) * p_hallucination)
    corrupt_idx = np.random.choice(syn_df.index, size=n_corrupt, replace=False)
    
    syn_df.loc[corrupt_idx, 'owner_occupied'] = syn_df.loc[corrupt_idx, 'total_housing_units'] * 2
    
    # 4. Measure Metrics
    print("  Evaluating HIF...")
    hif_res = hif_score(real_df, syn_df, verbose=False)
    hif_val = hif_res['hif_score']
    
    print("  Evaluating JCD...")
    jcd_val = joint_correlation_distance(real_df, syn_df)
    
    return {
        "p": p_hallucination,
        "hif": hif_val,
        "jcd": jcd_val
    }

if __name__ == "__main__":
    np.random.seed(42)
    
    results = []
    for p in [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]:
        res = run_sensitivity_test(p)
        results.append(res)
        
    df = pd.DataFrame(results)
    
    # Normalize for comparison (Inverse HIF change vs JCD change)
    hif_0 = df.loc[df['p'] == 0, 'hif'].iloc[0]
    df['hif_drop'] = (hif_0 - df['hif']) / hif_0 if hif_0 > 0 else 0
    
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS: HIF vs. JOINT CORRELATION DISTANCE")
    print("="*60)
    print(df[['p', 'hif', 'jcd', 'hif_drop']])
    print("="*60)
    
    # Save for appendix
    df.to_csv("results/sensitivity_analysis.csv", index=False)
    print("\nResults saved to results/sensitivity_analysis.csv")
