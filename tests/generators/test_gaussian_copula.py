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
    assert u[2] == 0.5


class TestGaussianCopula:
    def test_reproducible_with_seed_on_custom_data(self):
        from tabular_polygraph.generators import GaussianCopulaGenerator

        df = pd.DataFrame(
            {
                "loan_amount": [100000 + i * 1000 for i in range(120)],
                "interest_rate": [3.5 + i * 0.01 for i in range(120)],
                "credit_score": [640 + (i % 60) for i in range(120)],
                "segment": (["A", "B", "C"] * 40),
            }
        )
        gen = GaussianCopulaGenerator()
        gen.fit(df)
        syn1 = gen.generate(50, seed=99)
        syn2 = gen.generate(50, seed=99)
        pd.testing.assert_frame_equal(
            syn1.drop(columns=["syn_id"]), syn2.drop(columns=["syn_id"])
        )

    def test_all_cross_sectional_datasets(self, all_seeds):
        from tabular_polygraph.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(all_seeds["census_acs"])
        out = gen.generate(100, seed=42)
        assert len(out) == 100
        assert "syn_id" in out.columns

    def test_different_seeds_differ(self):
        from tabular_polygraph.generators import GaussianCopulaGenerator

        df = pd.DataFrame(
            {
                "loan_amount": [100000 + i * 1000 for i in range(200)],
                "interest_rate": [3.0 + i * 0.01 for i in range(200)],
                "credit_score": [620 + (i % 80) for i in range(200)],
                "segment": (["A", "B", "C", "D"] * 50),
            }
        )
        gen = GaussianCopulaGenerator()
        gen.fit(df)
        out1 = gen.generate(120, seed=1)
        out2 = gen.generate(120, seed=2)
        assert not out1["loan_amount"].equals(out2["loan_amount"])

    def test_correlation_matrix_property(self):
        from tabular_polygraph.generators import GaussianCopulaGenerator

        df = pd.DataFrame(
            {
                "a": list(range(150)),
                "b": [i * 1.5 + 10 for i in range(150)],
                "c": [200 - i for i in range(150)],
            }
        )
        gen = GaussianCopulaGenerator()
        gen.fit(df)
        corr = gen.correlation_matrix
        assert corr is not None
        assert corr.shape[0] == corr.shape[1]
        assert np.allclose(np.diag(corr.values), 1.0, atol=0.01)
