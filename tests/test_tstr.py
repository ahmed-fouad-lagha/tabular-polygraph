from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.metrics.downstream import Downstream


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

    metric = Downstream(target_col="target")
    assert metric.validate(real, syn) is None
    res = metric.compute(real, syn, real.columns.tolist())
    assert "error" not in res
    assert res["task"] == "class"
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

    metric = Downstream(target_col="target")
    assert metric.validate(real, syn) is None
    res = metric.compute(real, syn, real.columns.tolist())
    assert "error" not in res
    assert res["task"] == "reg"
    assert "tstr_score" in res
    assert "trr_score" in res


def test_tstr_score_missing_target():
    real = pd.DataFrame({"x": [1, 2, 3]})
    syn = pd.DataFrame({"x": [1, 2, 3]})
    metric = Downstream(target_col="nonexistent")
    err = metric.validate(real, syn)
    assert err is not None


def test_tstr_score_small_data_returns_error():
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
    metric = Downstream(target_col="target")
    assert metric.validate(real, syn) is None
    res = metric.compute(real, syn, real.columns.tolist())
    assert "error" in res
