import pandas as pd

from tabular_polygraph.privacy.singling_out import _risk_level, singling_out_risk


def test_singling_out_risk_no_qi():
    real = pd.DataFrame({"a": [1, 2, 3]})
    syn = pd.DataFrame({"a": [1, 2, 3]})
    # categorical_columns will return empty if 'a' is int
    res = singling_out_risk(real, syn, quasi_id_cols=[])
    assert "error" in res
    assert res["singling_out_rate"] == 0.0


def test_singling_out_risk_few_qi():
    real = pd.DataFrame({"a": ["x", "y", "z"], "b": ["1", "2", "3"]})
    syn = pd.DataFrame({"a": ["x", "y", "z"], "b": ["1", "2", "3"]})
    res = singling_out_risk(real, syn, quasi_id_cols=["a"])
    assert res["singling_out_rate"] == 0.0


def test_risk_levels():
    assert _risk_level(0.0001) == "very_low"
    assert _risk_level(0.005) == "low"
    assert _risk_level(0.02) == "medium"
    assert _risk_level(0.1) == "high"
    assert _risk_level(0.2) == "very_high"
