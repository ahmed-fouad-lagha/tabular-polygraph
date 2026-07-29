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
from sklearn.neighbors import NearestNeighbors

from tabular_polygraph._config import (
    DEFAULT_PRIVACY_LINKABILITY_BASELINE,
    DEFAULT_PRIVACY_MIN_DATA,
    DEFAULT_PRIVACY_N_ATTACKS,
    DEFAULT_PRIVACY_SEED,
)
from tabular_polygraph._utils import DEFAULT_DROP_LIST, normalize, numeric_columns

from .common import risk_level_linkability


def linkability_risk(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    n_attacks: int = DEFAULT_PRIVACY_N_ATTACKS,
    seed: int = DEFAULT_PRIVACY_SEED,
    nn_ratio_threshold: float = 0.8,
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
    if numeric_cols is None:
        id_cols = DEFAULT_DROP_LIST
        cols = [
            c
            for c in numeric_columns(real)
            if c in synthetic.columns and c not in id_cols
        ]
    else:
        cols = numeric_cols

    if len(cols) < 2 or len(real) < DEFAULT_PRIVACY_MIN_DATA:
        return {
            "error": "Insufficient data for linkability test",
            "linkability_rate": 0.0,
        }

    arr_real = real[cols].fillna(0).values.astype(float)
    real_norm, mu, sigma = normalize(arr_real, return_params=True)

    n_test = min(n_attacks, len(synthetic))
    rng = np.random.default_rng(seed)
    syn_sample = synthetic.sample(n=n_test, random_state=rng).reset_index(drop=True)

    arr_syn = syn_sample[cols].fillna(0).values.astype(float)
    syn_norm = (arr_syn - mu) / sigma

    # Use NearestNeighbors for O(n log n) instead of O(n²)
    nbrs = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
    nbrs.fit(real_norm)
    dists, _ = nbrs.kneighbors(syn_norm)
    d1 = dists[:, 0]
    d2 = dists[:, 1]

    nndr = np.divide(d1, d2, out=np.full_like(d1, np.inf), where=(d2 > 0))
    linked = int(np.sum(nndr < nn_ratio_threshold))
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
