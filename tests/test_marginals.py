from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph.generators.gaussian_copula import (
    _CategoricalMarginal,
    _NumericMarginal,
)


def test_numeric_marginal():
    np.random.seed(42)
    s = pd.Series(np.random.randn(100))
    m = _NumericMarginal().fit(s)

    u = m.to_uniform(s)
    assert len(u) == 100
    assert np.all((u >= 0) & (u <= 1))

    rec = m.from_uniform(u)
    assert len(rec) == 100


def test_categorical_marginal():
    s = pd.Series(["cat", "dog", "mouse", "cat", "dog"])
    m = _CategoricalMarginal().fit(s)

    u = m.to_uniform(s)
    assert len(u) == 5
    assert np.all((u >= 0) & (u <= 1))

    rec = m.from_uniform(u)
    assert len(rec) == 5
    assert set(rec).issubset({"cat", "dog", "mouse"})


def test_categorical_marginal_nan_handling():
    s = pd.Series(["cat", "dog", None, "mouse"])
    m = _CategoricalMarginal().fit(s)

    u = m.to_uniform(s)
    assert len(u) == 4
    # Check that NaN does not crash and gets mapped to 0.5 uniform value
    assert u[2] == 0.5
