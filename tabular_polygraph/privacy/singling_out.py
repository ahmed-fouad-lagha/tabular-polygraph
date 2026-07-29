"""
tabular_polygraph.privacy.singling_out
---------------------------------
Singling-out attack: can an adversary uniquely identify an individual
in the real dataset using only the synthetic data?

Methodology (simplified generalised singling-out):
  For each synthetic record, count how many real records match it
  on a random subset of quasi-identifier columns. If only 1 real record
  matches, the synthetic record has "singled out" that individual.

Optimization: Uses vectorized string comparison instead of iterrows().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph._config import (
    DEFAULT_PRIVACY_QUASI_ID_MAX,
    DEFAULT_PRIVACY_SEED,
    DEFAULT_PRIVACY_SINGLING_OUT_N_ATTACKS,
)
from tabular_polygraph.utils import DEFAULT_DROP_LIST, categorical_columns

from .common import risk_level_singling_out


def singling_out_risk(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    quasi_id_cols: list[str] | None = None,
    n_attacks: int = DEFAULT_PRIVACY_SINGLING_OUT_N_ATTACKS,
    seed: int = DEFAULT_PRIVACY_SEED,
) -> dict:
    """
    Estimate singling-out risk via random quasi-identifier subset attacks.

    Returns
    -------
    dict with singling_out_rate (0–1), n_attacks, n_singled_out, risk_level
    """
    id_cols = DEFAULT_DROP_LIST
    shared = [c for c in real.columns if c in synthetic.columns and c not in id_cols]
    if quasi_id_cols is None:
        qi_cols = [c for c in categorical_columns(real) if c in shared][
            :DEFAULT_PRIVACY_QUASI_ID_MAX
        ]
    else:
        qi_cols = quasi_id_cols

    if not qi_cols:
        return {"error": "No quasi-identifier columns found", "singling_out_rate": 0.0}

    n_singled = 0
    n_tested = min(n_attacks, len(synthetic))
    rng = np.random.default_rng(seed)
    syn_sample = (
        synthetic.sample(n=n_tested, random_state=rng)
        if len(synthetic) > n_tested
        else synthetic
    )

    # Pre-compute string columns for real data
    real_str = real[qi_cols].astype(str)
    syn_str = syn_sample[qi_cols].astype(str)

    for i in range(n_tested):
        # Pick a random subset of 2-4 quasi-identifiers
        if len(qi_cols) < 2:
            continue
        k = int(rng.integers(2, min(5, len(qi_cols) + 1)))
        cols = list(rng.choice(qi_cols, size=k, replace=False))

        # Vectorized comparison: (real[col] == syn_val).all(axis=1)
        mask = np.ones(len(real), dtype=bool)
        for col in cols:
            mask &= real_str[col].values == syn_str.iloc[i][col]

        n_matching = int(mask.sum())
        if n_matching == 1:
            n_singled += 1

    rate = round(n_singled / max(n_tested, 1), 4)
    return {
        "singling_out_rate": rate,
        "n_attacks": n_tested,
        "n_singled_out": n_singled,
        "risk_level": risk_level_singling_out(rate),
        "quasi_id_cols": qi_cols,
    }

    # Pre-convert real quasi-identifier columns to string for vectorized comparison
    real_qi = real[qi_cols].astype(str)
    syn_qi = syn_sample[qi_cols].astype(str)

    for _ in range(n_tested):
        # Pick a random subset of 2–4 quasi-identifiers
        k = int(rng.integers(2, min(5, len(qi_cols) + 1)))
        cols = list(rng.choice(qi_cols, size=k, replace=False))

        # Sample a random synthetic row
        syn_row = syn_qi.sample(n=1, random_state=rng).iloc[0]

        # Vectorized matching: find rows matching on all selected columns
        mask = np.ones(len(real), dtype=bool)
        for col in cols:
            mask &= real_qi[col].values == syn_row[col]

        n_matching = int(mask.sum())
        if n_matching == 1:
            n_singled += 1

    rate = round(n_singled / max(n_tested, 1), 4)
    return {
        "singling_out_rate": rate,
        "n_attacks": n_tested,
        "n_singled_out": n_singled,
        "risk_level": risk_level_singling_out(rate),
        "quasi_id_cols": qi_cols,
    }
