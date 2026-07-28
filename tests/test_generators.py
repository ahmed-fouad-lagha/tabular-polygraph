from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabular_polygraph.generators.base import BaseGenerator


class DummyGenerator(BaseGenerator):
    """Minimal concrete generator for testing BaseGenerator internals."""

    def _init(self, **kwargs):
        pass

    def fit(self, data: pd.DataFrame) -> DummyGenerator:
        self._record_schema(data)
        self._fitted = True
        return self

    def _generate(
        self, n: int, filters: dict | None = None, seed: int | None = None
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "age": [20 + i for i in range(n)],
                "score": [1.23456 for _ in range(n)],
                "state": ["CA" if i % 2 == 0 else "NY" for i in range(n)],
            }
        )
        df = self._cast_types(df)
        if filters:
            df = self._apply_filters(df, filters)
        return self._add_syn_id(df.head(n))


class TestBaseGenerator:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseGenerator()

    def test_require_fitted_guard(self):
        gen = DummyGenerator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            gen.generate(5)

    def test_syn_id_sequential(self):
        df = pd.DataFrame(
            {
                "age": [25, 30, 35],
                "score": [1.1, 2.2, 3.3],
                "state": ["CA", "NY", "CA"],
            }
        )

        gen = DummyGenerator()
        gen.fit(df)
        syn1 = gen.generate(3)
        assert "syn_id" in syn1.columns
        assert len(syn1) == 3
        assert syn1["syn_id"].iloc[0] == "SYN-0"

        syn2 = gen.generate(2)
        assert syn2["syn_id"].iloc[0] == "SYN-3"

    def test_fit_generate_fluent(self):
        df = pd.DataFrame({"a": [1, 2, 3] * 50})
        gen = DummyGenerator()
        out = gen.fit_generate(df, 10)
        assert len(out) == 10

    def test_filters(self):
        df = pd.DataFrame(
            {
                "debt_to_income": [10.0, 20.0, 30.0, 40.0],
                "state": ["CA", "NY", "CA", "TX"],
            }
        )

        gen = DummyGenerator()
        gen.fit(df)
        gen._columns = ["debt_to_income", "state"]

        filtered = gen._apply_filters(df, {"debt_to_income_min": 25.0})
        assert len(filtered) == 2
        assert (filtered["debt_to_income"] >= 25.0).all()

    def test_repr_before_after_fit(self):
        gen = DummyGenerator()
        assert "not fitted" in repr(gen)
        gen.fit(pd.DataFrame({"a": [1]}))
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


class TestCTGAN:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 200).astype(float),
                "income": np.random.uniform(20000, 150000, 200),
                "category": np.random.choice(["A", "B", "C"], 200),
                "label": np.random.choice([0, 1], 200).astype(float),
            }
        )

    def test_discovery_logic(self, sample_data):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        pytest.importorskip("ctgan")
        gen = CTGANGenerator(epochs=1)
        gen.fit(sample_data)
        assert gen._fitted
        assert len(gen._columns) == 4

    def test_generate_basic(self, sample_data):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        pytest.importorskip("ctgan")
        gen = CTGANGenerator(epochs=1)
        gen.fit(sample_data)
        syn = gen.generate(10)
        assert len(syn) == 10
        assert list(syn.columns) == ["syn_id"] + list(sample_data.columns)
        assert syn["age"].dtype == sample_data["age"].dtype

    @pytest.mark.xfail(
        reason="SDV 1.x handles RNG state internally for consecutive sampling"
    )
    def test_reproducibility(self, sample_data):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        pytest.importorskip("ctgan")
        gen = CTGANGenerator(epochs=1)
        gen.fit(sample_data)
        syn1 = gen.generate(5, seed=42)
        syn2 = gen.generate(5, seed=42)
        pd.testing.assert_frame_equal(syn1, syn2)

    def test_filters(self, sample_data):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        pytest.importorskip("ctgan")
        gen = CTGANGenerator(epochs=1)
        gen.fit(sample_data)
        syn = gen.generate(5, filters={"category": "A"})
        assert len(syn) == 5
        assert (syn["category"] == "A").all()

    def test_require_fitted_guard(self):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        gen = CTGANGenerator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            gen.generate(10)

    def test_manual_discrete_override(self, sample_data):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        pytest.importorskip("ctgan")
        gen = CTGANGenerator(epochs=1, discrete_columns=["category"])
        gen.fit(sample_data)
        assert gen._fitted

    def test_custom_params(self, sample_data):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        pytest.importorskip("ctgan")
        gen = CTGANGenerator(
            epochs=2,
            batch_size=100,
            generator_lr=1e-3,
            discriminator_steps=5,
            discrete_threshold=5,
        )
        gen.fit(sample_data)
        assert gen._epochs == 2
        assert gen._batch_size == 100

    def test_missing_dependency(self, monkeypatch):
        import builtins

        from tabular_polygraph.generators.ctgan import CTGANGenerator

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sdv.single_table":
                raise ImportError("Mocked missing SDV")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        gen = CTGANGenerator()
        with pytest.raises(ImportError, match="SDV is not installed"):
            gen._require_sdv()


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
