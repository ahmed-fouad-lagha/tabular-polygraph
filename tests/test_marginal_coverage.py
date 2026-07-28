import pandas as pd

from tabular_polygraph.fidelity.marginal import (
    ks_distribution_scores,
    moment_matching_scores,
)


def test_marginal_non_numeric():
    df1 = pd.DataFrame({"A": ["x", "y"] * 10})
    df2 = pd.DataFrame({"A": ["x", "y"] * 10})
    # Should skip non-numeric columns
    mm = moment_matching_scores(df1, df2, columns=["A"])
    ks = ks_distribution_scores(df1, df2, columns=["A"])
    assert len(mm) == 0
    assert len(ks) == 0


def test_marginal_small_dataset():
    df1 = pd.DataFrame({"A": [1.0] * 5})
    df2 = pd.DataFrame({"A": [1.0] * 5})
    # Should skip small datasets (< 10 rows)
    mm = moment_matching_scores(df1, df2, columns=["A"])
    ks = ks_distribution_scores(df1, df2, columns=["A"])
    assert len(mm) == 0
    assert len(ks) == 0


def test_marginal_constant_shift_not_perfect():
    df1 = pd.DataFrame({"A": [1.0] * 20})
    df2 = pd.DataFrame({"A": [2.0] * 20})
    mm = moment_matching_scores(df1, df2, columns=["A"])
    assert "A" in mm
    assert mm["A"] < 100.0


def test_fidelity_marginal_scores():
    from tabular_polygraph.fidelity.marginal import moment_matching_scores

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
    scores = moment_matching_scores(real, syn)
    assert "a" in scores
    assert scores["a"] > 0
