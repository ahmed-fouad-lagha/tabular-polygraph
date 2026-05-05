"""
Per-column fidelity metrics for research reporting.

Moment matching and KS fit are the primary marginal metrics used by the
current fidelity report. Scores range 0–100. Higher = more faithful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tabular_polygraph.utils import numeric_columns, to_numeric_array


def moment_matching_scores(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, float]:
    """Per-column marginal moment-matching scores for numeric columns."""
    if columns is None:
        cols = [c for c in numeric_columns(real) if c in synthetic.columns]
    else:
        cols = columns
    scores: dict[str, float] = {}
    eps = 1e-8

    for col in cols:
        if not pd.api.types.is_numeric_dtype(
            real[col]
        ) or not pd.api.types.is_numeric_dtype(synthetic[col]):
            continue

        r = to_numeric_array(real[col], fill_method="dropna")
        s = to_numeric_array(synthetic[col], fill_method="dropna")
        if len(r) < 10 or len(s) < 10:
            continue

        mean_r, std_r = float(r.mean()), float(r.std(ddof=0))
        mean_s, std_s = float(s.mean()), float(s.std(ddof=0))
        skew_r, skew_s = (
            float(stats.skew(r, nan_policy="omit")),
            float(stats.skew(s, nan_policy="omit")),
        )
        kurt_r, kurt_s = (
            float(stats.kurtosis(r, fisher=False, nan_policy="omit")),
            float(stats.kurtosis(s, fisher=False, nan_policy="omit")),
        )

        mean_err = abs(mean_s - mean_r) / (abs(mean_r) + eps)
        std_err = abs(std_s - std_r) / (std_r + eps)
        # Regularization for skew/kurtosis: when true moments are near zero,
        # use Laplace smoothing to dampen error (0.5, 1.0 are tuned defaults)
        skew_err = abs(skew_s - skew_r) / (abs(skew_r) + 0.5)
        kurt_err = abs(kurt_s - kurt_r) / (abs(kurt_r) + 1.0)

        score = 100.0 * (
            1.0 - 0.40 * mean_err - 0.35 * std_err - 0.15 * skew_err - 0.10 * kurt_err
        )
        scores[col] = round(float(max(0.0, min(100.0, score))), 2)

    return scores


def mean_moment_matching_score(scores: dict[str, float]) -> float:
    """Mean moment-matching score across numeric columns."""
    return round(float(np.mean(list(scores.values()))), 2) if scores else 0.0


def ks_distribution_scores(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, float]:
    """Per-column KS distributional fit scores for numeric columns."""
    if columns is None:
        cols = [c for c in numeric_columns(real) if c in synthetic.columns]
    else:
        cols = columns
    scores: dict[str, float] = {}

    for col in cols:
        if not pd.api.types.is_numeric_dtype(
            real[col]
        ) or not pd.api.types.is_numeric_dtype(synthetic[col]):
            continue

        r = to_numeric_array(real[col], fill_method="dropna")
        s = to_numeric_array(synthetic[col], fill_method="dropna")
        if len(r) < 10 or len(s) < 10:
            continue

        ks_stat, _ = stats.ks_2samp(r, s)
        scores[col] = round(float(max(0.0, min(100.0, (1.0 - ks_stat) * 100.0))), 2)

    return scores


def mean_ks_score(scores: dict[str, float]) -> float:
    """Mean KS score across numeric columns."""
    return round(float(np.mean(list(scores.values()))), 2) if scores else 0.0
