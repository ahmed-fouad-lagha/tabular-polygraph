from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabular_polygraph.fidelity.downstream import tstr_score


def test_tstr_score_classification():
    np.random.seed(42)
    n = 100
    real = pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.choice(["A", "B"], size=n),
        }
    )
    syn = pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.choice(["A", "B"], size=n),
        }
    )

    res = tstr_score(real, syn, target_col="target", task="classification")
    assert "error" not in res
    assert res["task"] == "classification"
    assert "tstr_score" in res
    assert "trr_score" in res
    assert "ratio" in res


def test_tstr_score_regression():
    np.random.seed(42)
    n = 100
    real = pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.randn(n),
        }
    )
    syn = pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.randn(n),
        }
    )

    res = tstr_score(real, syn, target_col="target", task="regression")
    assert "error" not in res
    assert res["task"] == "regression"
    assert "tstr_score" in res
    assert "trr_score" in res


def test_tstr_score_missing_target():
    real = pd.DataFrame({"x": [1, 2, 3]})
    syn = pd.DataFrame({"x": [1, 2, 3]})
    res = tstr_score(real, syn, target_col="nonexistent")
    assert "error" in res


def test_tstr_score_dropna_warning():
    np.random.seed(42)
    n = 20  # small size (< 50)
    real = pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], size=n),
        }
    )
    syn = pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], size=n),
        }
    )
    with pytest.warns(UserWarning, match="is < 50"):
        tstr_score(real, syn, target_col="target")
