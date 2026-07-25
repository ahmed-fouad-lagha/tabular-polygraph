import numpy as np
import pandas as pd
import pytest


class TestBaseGenerator:
    def test_abstract_cannot_instantiate(self):
        from tabular_polygraph.generators.base import BaseGenerator

        with pytest.raises(TypeError):
            BaseGenerator()

    def test_require_fitted_guard(self):
        from tabular_polygraph.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            gen.generate(10)

    def test_fit_sample_fluent(self):
        from tabular_polygraph.generators import GaussianCopulaGenerator

        df = pd.DataFrame(
            {
                "x": list(range(120)),
                "y": [i * 0.5 for i in range(120)],
                "grp": (["A", "B", "C"] * 40),
            }
        )
        out = GaussianCopulaGenerator().fit_sample(df, 50, seed=1)
        assert len(out) == 50

    def test_base_aliases(self):
        from tabular_polygraph.generators import GaussianCopulaGenerator

        df = pd.DataFrame({"a": [1, 2, 3] * 50})
        gen = GaussianCopulaGenerator()
        gen.fit(df)
        assert len(gen.sample(10)) == 10
        assert len(gen.fit_sample(df, 10)) == 10

    def test_repr_before_after_fit(self):
        from tabular_polygraph.generators import GaussianCopulaGenerator

        df = pd.DataFrame(
            {
                "x": list(range(100)),
                "y": [100 + i for i in range(100)],
                "grp": (["A", "B"] * 50),
            }
        )
        gen = GaussianCopulaGenerator()
        assert "not fitted" in repr(gen)
        gen.fit(df)
        assert "fitted on" in repr(gen)


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
        pd.testing.assert_frame_equal(syn1, syn2)

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


class TestAdvancedGenerators:
    def test_vine_copula_smoke(self):
        try:
            import pyvinecopulib  # noqa: F401
        except ImportError:
            pytest.skip("pyvinecopulib not installed")

        from tabular_polygraph.generators import VineCopulaGenerator

        df = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),
                "c": np.random.choice(["X", "Y"], 100),
            }
        )
        gen = VineCopulaGenerator()
        gen.fit(df)
        syn = gen.generate(10)
        assert len(syn) == 10
        assert all(col in syn.columns for col in df.columns)

    def test_tvae_smoke(self):
        try:
            from sdv.single_table import TVAESynthesizer  # noqa: F401
        except ImportError:
            pytest.skip("sdv not installed")

        from tabular_polygraph.generators import TVAEGenerator

        df = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),
                "c": np.random.choice(["X", "Y"], 100),
            }
        )
        gen = TVAEGenerator(epochs=10)
        gen.fit(df)
        syn = gen.generate(10)
        assert len(syn) == 10
        assert all(col in syn.columns for col in df.columns)
