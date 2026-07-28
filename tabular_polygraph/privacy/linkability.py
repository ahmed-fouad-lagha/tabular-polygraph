"""
tabular_polygraph.privacy.linkability
--------------------------------
Linkability (re-identification) attack via Nearest Neighbour
Distance Ratio (NNDR).

Methodology:
  For each synthetic record, find the nearest and second-nearest
  neighbour in the real data.  If the ratio d1/d2 < threshold, the
  record is unusually close to a single real record (potential
  memorisation), and the record is considered "linkable".
  The linkability rate is the fraction of synthetic records that
  exceed this threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph._config import (
    DEFAULT_PRIVACY_LINKABILITY_BASELINE,
    DEFAULT_PRIVACY_MIN_DATA,
    DEFAULT_PRIVACY_N_ATTACKS,
    DEFAULT_PRIVACY_SEED,
)
from tabular_polygraph.utils import normalize, numeric_columns

from .common import risk_level_linkability


def _normalise(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arr = df[cols].fillna(0).values.astype(float)
    result = normalize(arr, return_params=False)
    assert isinstance(result, np.ndarray)
    return result


def linkability_risk(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    n_attacks: int = DEFAULT_PRIVACY_N_ATTACKS,
    seed: int = DEFAULT_PRIVACY_SEED,
    nn_ratio_threshold: float = 0.5,
) -> dict:
    """
    Estimate linkability risk via Nearest Neighbour Distance Ratio (NNDR).

    For each synthetic record the distance to its nearest real neighbour
    (d1) and its second-nearest real neighbour (d2) is computed.
    If d1 / d2 < *nn_ratio_threshold* the record is considered
    "linkable" — it is unusually close to a single real record,
    suggesting potential memorisation.

    Returns
    -------
    dict with linkability_rate (0–1), baseline (expected ratio by
    chance), risk_level, and lift over baseline.
    """
    rng = np.random.default_rng(seed)

    if numeric_cols is None:
        cols = [
            c for c in numeric_columns(real) if c in synthetic.columns and c != "syn_id"
        ]
    else:
        cols = numeric_cols

    if len(cols) < 2 or len(real) < DEFAULT_PRIVACY_MIN_DATA:
        return {
            "error": "Insufficient data for linkability test",
            "linkability_rate": 0.0,
        }

    real_norm = _normalise(real, cols)

    n_test = min(n_attacks, len(synthetic))
    syn_sample = synthetic.sample(n=n_test, random_state=int(seed)).reset_index(
        drop=True
    )

    linked = 0
    for i in range(n_test):
        syn_vec = _normalise(syn_sample.iloc[[i]], cols)[0]
        dists = np.sum((real_norm - syn_vec) ** 2, axis=1)
        sorted_dists = np.sort(dists)
        d1 = sorted_dists[0]
        d2 = sorted_dists[1]
        if d2 > 0 and (d1 / d2) < nn_ratio_threshold:
            linked += 1

    rate = round(linked / max(n_test, 1), 4)
    baseline = DEFAULT_PRIVACY_LINKABILITY_BASELINE  # expected by chance
    lift = round((rate - baseline) / baseline * 100, 1)

    return {
        "linkability_rate": rate,
        "baseline": baseline,
        "lift_over_baseline_pct": lift,
        "risk_level": risk_level_linkability(rate),
        "n_attacks": n_test,
        "n_linked": linked,
        "numeric_cols_used": cols,
    }
