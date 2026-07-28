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
from scipy.stats import chi2_contingency, spearmanr


def correlation_distance_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> float:
    """
    Score based on Frobenius distance between Universal Association matrices.
    Score = 100 means identical correlation structure.
    """
    if columns is None:
        cols = [c for c in real.columns if c in synthetic.columns]
    else:
        cols = columns
    if len(cols) < 2:
        return 100.0

    R_real = _association_matrix(real[cols])
    R_syn = _association_matrix(synthetic[cols])

    # Association values are bounded [0, 1].
    # Max Frobenius norm when every off-diagonal pair is diametrically opposite (diff is 1), squared = 1
    max_possible = 1.0 * np.sqrt(len(cols) * (len(cols) - 1))
    dist = np.linalg.norm(R_real - R_syn, "fro")
    score = max(0.0, 1 - float(dist) / max(float(max_possible), 1e-8)) * 100
    return round(float(score), 2)


def _cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """Bias-corrected Cramer's V for two categorical arrays."""
    crosstab = pd.crosstab(x, y)
    if crosstab.size == 0:
        return 0.0
    chi2 = chi2_contingency(crosstab, correction=False)[0]
    n = crosstab.sum().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = crosstab.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def _correlation_ratio(categories: np.ndarray, measurements: np.ndarray) -> float:
    """Correlation Ratio (eta) between a categorical and continuous array."""
    df = pd.DataFrame({"c": categories, "m": measurements})
    y_mean = df["m"].mean()
    if pd.isna(y_mean):
        return 0.0
    grouped = df.groupby("c")["m"]
    numerator = sum(len(g) * (g.mean() - y_mean) ** 2 for _, g in grouped)
    denominator = sum((df["m"] - y_mean) ** 2)
    if denominator == 0:
        return 0.0
    eta2 = float(numerator / denominator)
    return float(np.sqrt(max(0.0, min(1.0, eta2))))


def _association_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute pairwise association matrix (Spearman, Cramer's V, or Correlation Ratio)."""
    n = df.shape[1]
    cols = df.columns.tolist()
    mat = np.eye(n)
    is_num = {c: pd.api.types.is_numeric_dtype(df[c]) for c in cols}

    for i in range(n):
        for j in range(i + 1, n):
            ca, cb = cols[i], cols[j]
            mask = df[ca].notna() & df[cb].notna()
            if mask.sum() < 2:
                continue

            xa = df.loc[mask, ca].values
            xb = df.loc[mask, cb].values

            if is_num[ca] and is_num[cb]:
                r, _ = spearmanr(xa.astype(float), xb.astype(float))
                val = abs(r) if np.isfinite(r) else 0.0
            elif not is_num[ca] and not is_num[cb]:
                val = _cramers_v(xa, xb)
            else:
                if is_num[ca]:
                    val = _correlation_ratio(xb, xa.astype(float))
                else:
                    val = _correlation_ratio(xa, xb.astype(float))

            mat[i, j] = mat[j, i] = val
    return mat


def pairwise_correlation_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, float]:
    """
    Per-pair universal association delta (real − synthetic).
    Returns dict of 'col_a × col_b' → delta.
    """
    if columns is None:
        cols = [c for c in real.columns if c in synthetic.columns]
    else:
        cols = columns

    if len(cols) < 2:
        return {}

    R_real = _association_matrix(real[cols])
    R_syn = _association_matrix(synthetic[cols])

    result = {}
    for i, ca in enumerate(cols):
        for j, cb in enumerate(cols):
            if j <= i:
                continue
            delta = round(float(abs(R_real[i, j] - R_syn[i, j])), 4)
            result[f"{ca} × {cb}"] = delta
    return result
