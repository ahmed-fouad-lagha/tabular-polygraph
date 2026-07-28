from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph.privacy.linkability import linkability_risk


def test_linkability_risk_smoke():
    rng = np.random.default_rng(42)
    n = 100
    real = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        }
    )
    syn = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        }
    )
    result = linkability_risk(real, syn, n_attacks=20, seed=42)
    assert "linkability_rate" in result
    assert "risk_level" in result
    assert 0 <= result["linkability_rate"] <= 1


def test_linkability_risk_few_columns():
    real = pd.DataFrame({"a": [1, 2, 3]})
    syn = pd.DataFrame({"a": [4, 5, 6]})
    result = linkability_risk(real, syn)
    assert "error" in result


def test_linkability_risk_identical_data():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        }
    )
    result = linkability_risk(df, df, n_attacks=30, seed=42)
    assert result["linkability_rate"] > 0
