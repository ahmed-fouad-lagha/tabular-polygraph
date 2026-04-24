import pandas as pd

from tabular_polygraph.privacy.singling_out import singling_out_risk


def test_singling_out_basic():
    # Dataset with a unique record
    real = pd.DataFrame({"age": [20, 20, 20, 99], "zip": [10001, 10001, 10001, 99999]})
    syn = pd.DataFrame({"age": [20, 20, 20, 99], "zip": [10001, 10001, 10001, 99999]})

    risk = singling_out_risk(real, syn, n_attacks=10)
    assert "singling_out_rate" in risk
    assert risk["singling_out_rate"] >= 0


def test_singling_out_no_risk():
    real = pd.DataFrame({"a": [1] * 100, "b": [2] * 100})
    syn = pd.DataFrame({"a": [1] * 100, "b": [2] * 100})
    risk = singling_out_risk(real, syn, n_attacks=10)
    assert risk["singling_out_rate"] == 0.0
