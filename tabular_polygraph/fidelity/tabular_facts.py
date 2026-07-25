"""
Domain-agnostic stylized facts for tabular datasets.

Evaluates how well the synthetic data preserves empirical regularities found in
common tabular data (income, housing, science, etc.):

  1. Tail Integrity: Preservation of extreme value ratios (e.g., P99/P50).
  2. Predictive Parity: Do the top predictors of the main features match?
  3. Concentration Match: Does the top 5% account for the same total sum?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tabular_polygraph.utils import numeric_columns


def tabular_stylized_facts(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict:
    """Evaluate cross-sectional stylized facts."""
    cols = columns or [
        c
        for c in numeric_columns(real)
        if c in synthetic.columns and real[c].nunique() > 10
    ]

    if not cols:
        return {
            "_summary": {
                "applicable": False,
                "note": "No suitable numeric columns found.",
            }
        }

    results = {}
    scores = []

    for col in cols:
        r = real[col].dropna().astype(float).values
        s = synthetic[col].dropna().astype(float).values
        if len(r) < 50 or len(s) < 50:
            continue

        # 1. Tail Integrity (P99 / P50 ratio match)
        r_p99, r_p50 = np.percentile(r, [99, 50])
        s_p99, s_p50 = np.percentile(s, [99, 50])
        # Use abs(median) to avoid sign-flip when median is negative
        r_denom = abs(r_p50) if abs(r_p50) > 1e-9 else 1e-9
        s_denom = abs(s_p50) if abs(s_p50) > 1e-9 else 1e-9
        r_tail = r_p99 / r_denom
        s_tail = s_p99 / s_denom
        tail_match = 1.0 - min(abs(r_tail - s_tail) / (abs(r_tail) + 1e-9), 1.0)

        # 2. Predictive ordering (Correlation with other numeric features)
        # Filter to numeric columns only to avoid TypeError on non-numeric
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(real[c])]
        r_corr = (
            real[numeric_cols]
            .corr(numeric_only=True)[col]
            .drop(col)
            .abs()
            .sort_values(ascending=False)
        )
        s_corr = (
            synthetic[numeric_cols]
            .corr(numeric_only=True)[col]
            .drop(col)
            .abs()
            .sort_values(ascending=False)
        )

        # Rank correlation of the correlations (Spearman on correlation vectors)
        common_idx = r_corr.index.intersection(s_corr.index)
        if len(common_idx) < 2:
            rank_match = 0.0
        else:
            rank_match = float(
                stats.spearmanr(
                    s_corr.loc[common_idx].values, r_corr.loc[common_idx].values
                )[0]
            )
        rank_match = max(0, rank_match)  # Clamp negative correlations

        # 3. Concentration (Lorenz-style: top 5% share of absolute values)
        # Use absolute values so the metric works for columns with negative values
        r_abs = np.abs(r)
        s_abs = np.abs(s)
        r_sorted = np.sort(r_abs)
        s_sorted = np.sort(s_abs)
        r_sum = r_sorted.sum()
        s_sum = s_sorted.sum()
        r_top5_share = (
            r_sorted[int(0.95 * len(r)) :].sum() / (r_sum + 1e-9) if r_sum > 0 else 0.0
        )
        s_top5_share = (
            s_sorted[int(0.95 * len(s)) :].sum() / (s_sum + 1e-9) if s_sum > 0 else 0.0
        )
        conc_match = 1.0 - min(
            abs(r_top5_share - s_top5_share) / (r_top5_share + 1e-9), 1.0
        )

        col_score = round((tail_match + rank_match + conc_match) / 3 * 100, 1)
        scores.append(col_score)

        results[col] = {
            "tail_integrity": round(tail_match, 3),
            "predictive_parity": round(rank_match, 3),
            "concentration_match": round(conc_match, 3),
            "score": col_score,
        }

    results["_summary"] = {
        "applicable": True,
        "mean_score": round(float(np.mean(scores)), 1) if scores else 0.0,
        "columns_tested": len(scores),
    }
    return results
