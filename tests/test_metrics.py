import numpy as np
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


def test_privacy_membership_risk():
    from tabular_polygraph.privacy.disclosure import membership_inference_risk

    train = pd.DataFrame(
        {"a": np.random.normal(0, 1, 100), "b": np.random.normal(0, 1, 100)}
    )
    holdout = pd.DataFrame(
        {"a": np.random.normal(0, 1, 100), "b": np.random.normal(0, 1, 100)}
    )
    syn = pd.DataFrame(
        {"a": np.random.normal(0, 1, 100), "b": np.random.normal(0, 1, 100)}
    )
    risk = membership_inference_risk(train, holdout, syn, n_sample=20)
    assert "attack_auc" in risk


def test_privacy_linkability_risk():
    from tabular_polygraph.privacy.linkability import linkability_risk

    real = pd.DataFrame(
        {"a": np.random.normal(0, 1, 100), "b": np.random.normal(0, 1, 100)}
    )
    syn = pd.DataFrame(
        {"a": np.random.normal(0, 1, 100), "b": np.random.normal(0, 1, 100)}
    )
    risk = linkability_risk(real, syn, n_attacks=20)
    assert "linkability_rate" in risk
