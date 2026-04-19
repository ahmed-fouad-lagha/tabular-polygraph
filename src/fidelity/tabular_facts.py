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
from src.utils import numeric_columns


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
        r_tail = r_p99 / (r_p50 + 1e-9)
        s_tail = s_p99 / (s_p50 + 1e-9)
        tail_match = 1.0 - min(abs(r_tail - s_tail) / (abs(r_tail) + 1e-9), 1.0)

        # 2. Predictive ordering (Correlation with other numeric features)
        r_corr = real[cols].corr()[col].drop(col).abs().sort_values(ascending=False)
        s_corr = (
            synthetic[cols].corr()[col].drop(col).abs().sort_values(ascending=False)
        )

        # Rank correlation of the correlations (Spearman on correlation vectors)
        rank_match = float(
            stats.spearmanr(r_corr.index.map(s_corr.get), r_corr.values)[0]
        )
        rank_match = max(0, rank_match)  # Clamp negative correlations

        # 3. Concentration (Lorenz-style: top 5% share)
        r_sorted = np.sort(r)
        s_sorted = np.sort(s)
        r_top5_share = r_sorted[int(0.95 * len(r)) :].sum() / (r_sorted.sum() + 1e-9)
        s_top5_share = s_sorted[int(0.95 * len(s)) :].sum() / (s_sorted.sum() + 1e-9)
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
