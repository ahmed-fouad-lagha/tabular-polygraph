"""
tabular_polygraph.privacy.singling_out
---------------------------------
Singling-out attack: can an adversary uniquely identify an individual
in the real dataset using only the synthetic data?

Methodology (simplified generalised singling-out):
  For each synthetic record, count how many real records match it
  on a random subset of quasi-identifier columns. If only 1 real record
  matches, the synthetic record has "singled out" that individual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph._config import (
    DEFAULT_PRIVACY_N_ATTACKS,
    DEFAULT_PRIVACY_QUASI_ID_MAX,
    DEFAULT_PRIVACY_SEED,
    DEFAULT_PRIVACY_SINGLING_OUT_N_ATTACKS,
)
from tabular_polygraph.utils import categorical_columns

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
    rng = np.random.default_rng(seed)

    shared = [c for c in real.columns if c in synthetic.columns and c != "syn_id"]
    if quasi_id_cols is None:
        qi_cols = [c for c in categorical_columns(real) if c in shared][:DEFAULT_PRIVACY_QUASI_ID_MAX]
    else:
        qi_cols = quasi_id_cols

    if not qi_cols:
        return {"error": "No quasi-identifier columns found", "singling_out_rate": 0.0}

    n_singled = 0
    n_tested = min(n_attacks, len(synthetic))
    syn_sample = (
        synthetic.sample(n=n_tested, random_state=seed)
        if len(synthetic) > n_tested
        else synthetic
    )

    for _, syn_row in syn_sample.iterrows():
        # Pick a random subset of 2–4 quasi-identifiers
        if len(qi_cols) < 2:
            n_singled += 0
            continue
        k = int(rng.integers(2, min(5, len(qi_cols) + 1)))
        cols = list(rng.choice(qi_cols, size=k, replace=False))

        mask = pd.Series([True] * len(real))
        for col in cols:
            mask = mask & (real[col].astype(str) == str(syn_row.get(col, "")))

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
