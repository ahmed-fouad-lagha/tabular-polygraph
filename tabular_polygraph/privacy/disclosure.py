"""
tabular_polygraph.privacy.disclosure
-------------------------------
Membership inference attack: can an adversary determine whether a
specific real record was used to train the generator?

Methodology (shadow model approach, simplified):
  A "member" is a real training record.
  A "non-member" is a held-out real record not seen during training.
  We score each by its distance to its nearest synthetic neighbour.
  Members are expected to be closer (memorisation signal).
  Attack advantage = AUC - 0.5  (0 = no advantage, 0.5 = perfect attack).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph._config import (
    DEFAULT_PRIVACY_BATCH,
    DEFAULT_PRIVACY_N_SAMPLE,
    DEFAULT_PRIVACY_SEED,
    DEFAULT_PRIVACY_SYN_MULTIPLIER,
)
from tabular_polygraph._utils import DEFAULT_DROP_LIST, normalize, numeric_columns

from .common import risk_level_membership


def _min_dist_to_synthetic(
    records: np.ndarray,
    synthetic: np.ndarray,
    batch_size: int = DEFAULT_PRIVACY_BATCH,
) -> np.ndarray:
    """Return minimum L2 distance from each record to the synthetic set."""
    min_dists = np.full(len(records), np.inf)
    for i in range(0, len(synthetic), batch_size):
        block = synthetic[i : i + batch_size]
        dists = np.sqrt(((records[:, None, :] - block[None, :, :]) ** 2).sum(axis=2))
        min_dists = np.minimum(min_dists, dists.min(axis=1))
    return min_dists


def _auc_from_scores(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """Compute AUC: P(member score < non-member score). Vectorized O(n log n)."""
    n_m = len(member_scores)
    n_nm = len(nonmember_scores)
    if n_m == 0 or n_nm == 0:
        return 0.5
    # Sort both arrays
    member_sorted = np.sort(member_scores)
    nonmember_sorted = np.sort(nonmember_scores)
    # Vectorized: for each non-member score, count how many member scores are smaller
    # Using searchsorted for O(n log n) instead of O(n²)
    ranks = np.searchsorted(member_sorted, nonmember_sorted, side="left")
    wins = ranks.sum()
    return round(wins / (n_m * n_nm), 4)


def membership_inference_risk(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    n_sample: int = DEFAULT_PRIVACY_N_SAMPLE,
    seed: int = DEFAULT_PRIVACY_SEED,
) -> dict:
    """
    Estimate membership inference risk.

    Parameters
    ----------
    real_train   : records used to train the generator (members)
    real_holdout : records NOT used in training (non-members)
    synthetic    : generated synthetic data
    n_sample     : number of records to sample from each group

    Returns
    -------
    dict with attack_auc, advantage (AUC - 0.5), risk_level
    """
    id_cols = DEFAULT_DROP_LIST
    cols = numeric_cols or [
        c
        for c in numeric_columns(real_train)
        if c in synthetic.columns and c not in id_cols
    ]

    if len(cols) < 2:
        return {"error": "Insufficient numeric columns", "attack_auc": 0.5}

    rng = np.random.default_rng(seed)
    sample_train = real_train.sample(n=min(n_sample, len(real_train)), random_state=rng)
    sample_holdout = real_holdout.sample(
        n=min(n_sample, len(real_holdout)), random_state=rng
    )
    sample_syn = synthetic.sample(
        n=min(n_sample * DEFAULT_PRIVACY_SYN_MULTIPLIER, len(synthetic)),
        random_state=rng,
    )

    arr_tr = sample_train[cols].fillna(0).values.astype(float)
    arr_ho = sample_holdout[cols].fillna(0).values.astype(float)
    arr_sy = sample_syn[cols].fillna(0).values.astype(float)

    # NORMALIZATION: Fit on combined data to avoid train->synthetic leakage
    # Stack train + holdout + synthetic to compute unified mean/std
    combined = np.vstack([arr_tr, arr_ho, arr_sy])
    _, mu, sigma = normalize(combined, return_params=True)
    members = (arr_tr - mu) / sigma
    nonmembers = (arr_ho - mu) / sigma
    syn_arr = (arr_sy - mu) / sigma

    # Score: distance to nearest synthetic neighbour
    m_dists = _min_dist_to_synthetic(members, syn_arr)
    nm_dists = _min_dist_to_synthetic(nonmembers, syn_arr)

    auc = _auc_from_scores(m_dists, nm_dists)
    advantage = round(auc - 0.5, 4)

    return {
        "attack_auc": auc,
        "advantage": advantage,
        "risk_level": risk_level_membership(auc),
        "n_members": len(members),
        "n_nonmembers": len(nonmembers),
        "interpretation": (
            "No meaningful memorisation detected"
            if auc < 0.52
            else "Slight memorisation signal"
            if auc < 0.60
            else "Moderate memorisation — review generator"
            if auc < 0.70
            else "High memorisation risk — consider DP noise"
        ),
    }
