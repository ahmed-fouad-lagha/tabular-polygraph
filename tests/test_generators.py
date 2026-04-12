import numpy as np
import pandas as pd
import pytest


class TestBaseGenerator:
    def test_abstract_cannot_instantiate(self):
        from src.generators.base import BaseGenerator

        with pytest.raises(TypeError):
            BaseGenerator()

    def test_require_fitted_guard(self):
        from src.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            gen.generate(10)

    def test_fit_sample_fluent(self):
        from src.generators import GaussianCopulaGenerator

        df = pd.DataFrame(
            {
                "x": [i for i in range(120)],
                "y": [i * 0.5 for i in range(120)],
                "grp": (["A", "B", "C"] * 40),
            }
        )
        out = GaussianCopulaGenerator().fit_sample(df, 50, seed=1)
        assert len(out) == 50

    def test_repr_before_after_fit(self):
        from src.generators import GaussianCopulaGenerator

        df = pd.DataFrame(
            {
                "x": [i for i in range(100)],
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
        from src.generators import GaussianCopulaGenerator

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
        from src.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(all_seeds["census_acs"])
        out = gen.generate(100, seed=42)
        assert len(out) == 100
        assert "syn_id" in out.columns

    def test_different_seeds_differ(self):
        from src.generators import GaussianCopulaGenerator

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
        from src.generators import GaussianCopulaGenerator

        df = pd.DataFrame(
            {
                "a": [i for i in range(150)],
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


class TestVARGenerator:
    def test_basic_generation(self, syn_macro):
        assert len(syn_macro) == 300
        assert "syn_id" in syn_macro.columns

    def test_correct_column_count(self, fred_macro, syn_macro):
        expected = len(fred_macro.columns)
        assert len(syn_macro.columns) == expected

    def test_numeric_columns_in_range(self, fred_macro, syn_macro):
        for col in ["cpi_yoy", "unemployment_rate", "fed_funds_rate"]:
            if col in syn_macro.columns:
                r_min = fred_macro[col].min() * 4 - fred_macro[col].abs().max()
                r_max = fred_macro[col].max() * 4
                assert syn_macro[col].between(r_min, r_max).all(), (
                    f"{col} values outside plausible range"
                )

    def test_bls_var_generation(self, all_seeds):
        from src.generators.time_series import VARGenerator

        gen = VARGenerator(lags=2, time_col="quarter")
        gen.fit(all_seeds["bls"])
        out = gen.generate(100, seed=1)
        assert len(out) == 100
