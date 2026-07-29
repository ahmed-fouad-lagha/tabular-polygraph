from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.metrics.stylized_facts import StylizedFacts


def test_tabular_stylized_facts_basic():
    np.random.seed(42)
    n = 100
    real = pd.DataFrame(
        {
            "a": np.random.exponential(scale=2.0, size=n),
            "b": np.random.normal(loc=10, scale=3, size=n),
        }
    )
    syn = pd.DataFrame(
        {
            "a": np.random.exponential(scale=2.1, size=n),
            "b": np.random.normal(loc=9.8, scale=3.1, size=n),
        }
    )

    res = StylizedFacts().compute(real, syn, ["a", "b"])
    assert res["columns_tested"] >= 1
    assert res["mean_score"] is not None
    assert 0 <= res["mean_score"] <= 100


def test_tabular_stylized_facts_no_numeric():
    real = pd.DataFrame({"cat": ["A", "B"] * 20})
    syn = pd.DataFrame({"cat": ["A", "B"] * 20})
    res = StylizedFacts().compute(real, syn, [])
    assert not res["applicable"]
