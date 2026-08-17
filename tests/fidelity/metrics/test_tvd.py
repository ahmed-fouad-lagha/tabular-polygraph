import pandas as pd

from tabular_polygraph.fidelity.metrics.tvd import TVD


def test_tvd_identical_distribution_scores_100():
    real = pd.DataFrame({"c": ["A"] * 60 + ["B"] * 40})
    syn = real.copy()

    res = TVD().compute(real, syn, ["c"])

    assert res["column_scores"]["c"] == 100.0


def test_tvd_detects_distribution_shift():
    real = pd.DataFrame({"c": ["A"] * 70 + ["B"] * 20 + ["C"] * 10})
    syn = pd.DataFrame({"c": ["A"] * 50 + ["B"] * 30 + ["C"] * 20})

    res = TVD().compute(real, syn, ["c"])

    assert res["column_scores"]["c"] == 80.0


def test_tvd_ignores_nonexistent_columns():
    real = pd.DataFrame({"a": ["x", "y", "x", "y"]})
    syn = pd.DataFrame({"b": ["x", "y", "x", "y"]})

    res = TVD().compute(real, syn, ["a"])

    assert res["column_scores"] == {}
