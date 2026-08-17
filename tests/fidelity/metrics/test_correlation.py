import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.metrics.correlation import Correlation


def test_correlation_identical_data_scores_100():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    real = pd.DataFrame({"a": x, "b": x + rng.normal(scale=0.05, size=200)})

    res = Correlation().compute(real, real.copy(), ["a", "b"])

    assert res["correlation_distance_score"] == 100.0


def test_correlation_detects_inverted_relationship():
    rng = np.random.default_rng(4)
    x = rng.normal(size=250)
    real = pd.DataFrame({"a": x, "b": x})
    syn = pd.DataFrame({"a": x, "b": -x})

    res = Correlation().compute(real, syn, ["a", "b"])

    assert res["correlation_distance_score"] == 0.0


def test_correlation_handles_mixed_types():
    real = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat": ["A", "A", "B", "B", "C", "C"],
        }
    )
    syn = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat": ["A", "B", "A", "C", "B", "C"],
        }
    )

    res = Correlation().compute(real, syn, ["num", "cat"])

    assert 0.0 <= res["correlation_distance_score"] <= 100.0
    assert isinstance(res["pairwise_deltas"], dict)
