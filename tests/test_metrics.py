import pandas as pd


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
