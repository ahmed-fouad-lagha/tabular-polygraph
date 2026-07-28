import pandas as pd

from tabular_polygraph.fidelity.metrics.ks_test import KSTest


def test_ks_non_numeric():
    df1 = pd.DataFrame({"A": ["x", "y"] * 10})
    df2 = pd.DataFrame({"A": ["x", "y"] * 10})
    ks = KSTest().compute(df1, df2, [])["column_scores"]
    assert len(ks) == 0


def test_ks_small_dataset():
    df1 = pd.DataFrame({"A": [1.0] * 5})
    df2 = pd.DataFrame({"A": [1.0] * 5})
    ks = KSTest().compute(df1, df2, ["A"])["column_scores"]
    assert len(ks) == 0
