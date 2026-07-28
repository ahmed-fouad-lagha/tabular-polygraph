from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph.privacy.disclosure import (
    _auc_from_scores,
    _min_dist_to_synthetic,
    membership_inference_risk,
)


def test_min_dist_to_synthetic():
    records = np.array([[0.0, 0.0], [10.0, 10.0]])
    synthetic = np.array([[0.1, 0.1], [0.2, 0.2]])
    dists = _min_dist_to_synthetic(records, synthetic)
    assert dists[0] < dists[1]


def test_auc_from_scores():
    members = np.array([0.1, 0.2, 0.3])
    nonmembers = np.array([5.0, 6.0, 7.0])
    auc = _auc_from_scores(members, nonmembers)
    assert auc > 0.9


def test_auc_tied_no_error():
    scores = np.array([1.0, 2.0, 3.0])
    result = _auc_from_scores(scores, scores)
    assert 0 <= result <= 1


def test_auc_empty_lists():
    assert _auc_from_scores(np.array([]), np.array([])) == 0.5
    assert _auc_from_scores(np.array([1.0]), np.array([])) == 0.5


def test_membership_inference_risk_smoke():
    rng = np.random.default_rng(42)
    n = 100
    real = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
        }
    )
    syn = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
        }
    )
    result = membership_inference_risk(
        real_train=real.iloc[:70],
        real_holdout=real.iloc[70:],
        synthetic=syn,
        n_sample=20,
        seed=42,
    )
    assert "attack_auc" in result
    assert "risk_level" in result
    assert 0 <= result["attack_auc"] <= 1


def test_membership_inference_few_columns_errors():
    real = pd.DataFrame({"a": [1, 2, 3]})
    holdout = pd.DataFrame({"a": [4, 5, 6]})
    syn = pd.DataFrame({"a": [7, 8, 9]})
    result = membership_inference_risk(real, holdout, syn, n_sample=5)
    assert "error" in result
