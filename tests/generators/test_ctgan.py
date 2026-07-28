import builtins

import numpy as np
import pandas as pd
import pytest


def _sample_data():
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 200).astype(float),
            "income": np.random.uniform(20000, 150000, 200),
            "category": np.random.choice(["A", "B", "C"], 200),
            "label": np.random.choice([0, 1], 200).astype(float),
        }
    )


class TestCTGAN:
    def test_discovery_logic(self):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        pytest.importorskip("ctgan")
        gen = CTGANGenerator(epochs=1)
        gen.fit(_sample_data())
        assert gen._fitted
        assert len(gen._columns) == 4

    def test_generate_basic(self):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        pytest.importorskip("ctgan")
        gen = CTGANGenerator(epochs=1)
        gen.fit(_sample_data())
        syn = gen.generate(10)
        assert len(syn) == 10
        assert list(syn.columns) == ["syn_id"] + list(_sample_data().columns)
        assert syn["age"].dtype == _sample_data()["age"].dtype

    @pytest.mark.xfail(
        reason="SDV 1.x handles RNG state internally for consecutive sampling"
    )
    def test_reproducibility(self):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        pytest.importorskip("ctgan")
        gen = CTGANGenerator(epochs=1)
        gen.fit(_sample_data())
        syn1 = gen.generate(5, seed=42)
        syn2 = gen.generate(5, seed=42)
        pd.testing.assert_frame_equal(syn1, syn2)

    def test_filters(self):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        pytest.importorskip("ctgan")
        gen = CTGANGenerator(epochs=1)
        gen.fit(_sample_data())
        syn = gen.generate(5, filters={"category": "A"})
        assert len(syn) == 5
        assert (syn["category"] == "A").all()

    def test_require_fitted_guard(self):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        gen = CTGANGenerator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            gen.generate(10)

    def test_manual_discrete_override(self):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        pytest.importorskip("ctgan")
        gen = CTGANGenerator(epochs=1)
        gen.fit(_sample_data())
        assert gen._fitted

    def test_custom_params(self):
        from tabular_polygraph.generators.ctgan import CTGANGenerator

        pytest.importorskip("ctgan")
        gen = CTGANGenerator(
            epochs=2,
            batch_size=100,
        )
        gen.fit(_sample_data())
        assert gen._epochs == 2
        assert gen._batch_size == 100

    def test_missing_dependency(self, monkeypatch):
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
