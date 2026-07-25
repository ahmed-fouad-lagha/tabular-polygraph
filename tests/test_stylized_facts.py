from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.tabular_facts import tabular_stylized_facts


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

    res = tabular_stylized_facts(real, syn, columns=["a", "b"])
    assert "_summary" in res
    assert res["_summary"]["columns_tested"] >= 1
    assert "mean_score" in res["_summary"]
    assert 0 <= res["_summary"]["mean_score"] <= 100


def test_tabular_stylized_facts_no_numeric():
    real = pd.DataFrame({"cat": ["A", "B"] * 20})
    syn = pd.DataFrame({"cat": ["A", "B"] * 20})
    res = tabular_stylized_facts(real, syn, columns=[])
    assert res["_summary"]["applicable"] is False
