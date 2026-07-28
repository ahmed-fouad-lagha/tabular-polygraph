import pandas as pd

from tabular_polygraph.fidelity.metrics.ks_test import KSTest
from tabular_polygraph.fidelity.metrics.moment_matching import MomentMatching


def test_marginal_non_numeric():
    df1 = pd.DataFrame({"A": ["x", "y"] * 10})
    df2 = pd.DataFrame({"A": ["x", "y"] * 10})
    # Non-numeric columns should be skipped — the caller passes no numeric columns
    mm = MomentMatching().compute(df1, df2, [])["column_scores"]
    ks = KSTest().compute(df1, df2, [])["column_scores"]
    assert len(mm) == 0
    assert len(ks) == 0


def test_marginal_small_dataset():
    df1 = pd.DataFrame({"A": [1.0] * 5})
    df2 = pd.DataFrame({"A": [1.0] * 5})
    # Should skip small datasets (< 10 rows)
    mm = MomentMatching().compute(df1, df2, ["A"])["column_scores"]
    ks = KSTest().compute(df1, df2, ["A"])["column_scores"]
    assert len(mm) == 0
    assert len(ks) == 0


def test_marginal_constant_shift_not_perfect():
    df1 = pd.DataFrame({"A": [1.0] * 20})
    df2 = pd.DataFrame({"A": [2.0] * 20})
    mm = MomentMatching().compute(df1, df2, ["A"])["column_scores"]
    assert "A" in mm
    assert mm["A"] < 100.0


def test_fidelity_marginal_scores():
    real = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )
    syn = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
            "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 110],
        }
    )
    scores = MomentMatching().compute(real, syn, real.columns.tolist())["column_scores"]
    assert "a" in scores
    assert scores["a"] > 0
