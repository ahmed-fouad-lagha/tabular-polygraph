from __future__ import annotations

import pandas as pd

from tabular_polygraph.privacy.singling_out import singling_out_risk


def test_singling_out_risk_smoke():
    real = pd.DataFrame(
        {
            "cat": ["A", "B", "C", "D"] * 25,
            "region": ["X", "Y"] * 50,
            "group": ["M", "N", "O", "P"] * 25,
        }
    )
    syn = pd.DataFrame(
        {
            "cat": ["A", "B", "C", "D"] * 10,
            "region": ["X", "Y"] * 20,
            "group": ["M", "N", "O", "P"] * 10,
        }
    )
    result = singling_out_risk(real, syn, n_attacks=20, seed=42)
    assert "singling_out_rate" in result
    assert "risk_level" in result
    assert 0 <= result["singling_out_rate"] <= 1


def test_singling_out_risk_no_qi_cols():
    real = pd.DataFrame({"num": [1, 2, 3]})
    syn = pd.DataFrame({"num": [4, 5, 6]})
    result = singling_out_risk(real, syn)
    assert "error" in result


def test_singling_out_risk_explicit_qi():
    real = pd.DataFrame(
        {
            "a": ["x", "y", "z"] * 10,
            "b": ["p", "q"] * 15,
        }
    )
    syn = pd.DataFrame(
        {
            "a": ["x", "y"] * 5,
            "b": ["p"] * 10,
        }
    )
    result = singling_out_risk(
        real, syn, quasi_id_cols=["a", "b"], n_attacks=10, seed=42
    )
    assert "singling_out_rate" in result
