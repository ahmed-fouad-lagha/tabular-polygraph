"""
Joint distribution fidelity — captures how well the generator
preserves inter-column relationships, not just marginals.

Metrics
-------
correlation_distance : Frobenius norm between real and synthetic
                       correlation matrices, normalised to 0–100.
pairwise_mi_score    : Average mutual information ratio across column pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tabular_polygraph.utils import numeric_columns


def correlation_distance_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> float:
    """
    Score based on Frobenius distance between Spearman correlation matrices.
    Score = 100 means identical correlation structure.
    """
    if columns is None:
        cols = [c for c in numeric_columns(real) if c in synthetic.columns]
    else:
        cols = columns
    if len(cols) < 2:
        return 100.0

    R_real = _spearman_matrix(real[cols])
    R_syn = _spearman_matrix(synthetic[cols])

    # Max Frobenius norm when every off-diagonal pair is flipped (±1):
    # diagonal diff is always 0, off-diagonal max diff is 2, squared = 4
    max_possible = 2.0 * np.sqrt(len(cols) * (len(cols) - 1))
    dist = np.linalg.norm(R_real - R_syn, "fro")
    score = max(0.0, 1 - dist / max(max_possible, 1e-8)) * 100
    return round(float(score), 2)


def _spearman_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute pairwise Spearman correlation matrix using pairwise complete observations."""
    num_df = df.select_dtypes(include="number")
    n = num_df.shape[1]
    cols = num_df.columns.tolist()
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            mask = num_df[cols[i]].notna() & num_df[cols[j]].notna()
            if mask.sum() < 2:
                continue
            r, _ = spearmanr(
                num_df.loc[mask, cols[i]].values, num_df.loc[mask, cols[j]].values
            )
            mat[i, j] = mat[j, i] = r if np.isfinite(r) else 0.0
    return mat


def pairwise_correlation_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, float]:
    """
    Per-pair Spearman correlation delta (real − synthetic).
    Returns dict of 'col_a × col_b' → delta.
    """
    if columns is None:
        cols = [c for c in numeric_columns(real) if c in synthetic.columns]
    else:
        cols = columns
    # Ensure we only use truly numeric columns
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(real[c])]
    result = {}
    for i, ca in enumerate(cols):
        for cb in cols[i + 1 :]:
            # Properly handle NaNs: drop them for each pair independently
            # This preserves the actual correlation structure without arbitrary imputation
            real_pair = real[[ca, cb]].dropna()
            syn_pair = synthetic[[ca, cb]].dropna()

            if len(real_pair) < 2 or len(syn_pair) < 2:
                continue

            r_real, _ = spearmanr(
                real_pair[ca].astype(float), real_pair[cb].astype(float)
            )
            r_syn, _ = spearmanr(syn_pair[ca].astype(float), syn_pair[cb].astype(float))

            # Handle non-finite correlations (e.g., constant columns)
            r_real = r_real if np.isfinite(r_real) else 0.0
            r_syn = r_syn if np.isfinite(r_syn) else 0.0

            delta = round(float(abs(r_real - r_syn)), 4)
            result[f"{ca} × {cb}"] = delta
    return result
