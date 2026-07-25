import numpy as np
import pandas as pd

from tabular_polygraph.generators import GaussianCopulaGenerator


class TestEdgeCases:
    def test_single_column_numeric(self):
        df = pd.DataFrame({"x": list(range(100))})
        gen = GaussianCopulaGenerator()
        gen.fit(df)
        syn = gen.generate(20, seed=42)
        assert len(syn) == 20
        assert "x" in syn.columns

    def test_two_columns_perfect_correlation(self):
        df = pd.DataFrame({"a": list(range(100)), "b": list(range(100))})
        gen = GaussianCopulaGenerator()
        gen.fit(df)
        syn = gen.generate(50, seed=42)
        corr = syn["a"].corr(syn["b"])
        assert corr > 0.8

    def test_many_constant_columns(self):
        df = pd.DataFrame(
            {
                "const1": [1] * 100,
                "const2": ["A"] * 100,
                "real": np.random.randn(100),
            }
        )
        gen = GaussianCopulaGenerator()
        gen.fit(df)
        syn = gen.generate(30, seed=42)
        assert (syn["const1"] == 1).all()
        assert (syn["const2"] == "A").all()

    def test_generate_different_sizes(self):
        df = pd.DataFrame(
            {"a": np.random.randn(200), "b": np.random.choice(["X", "Y"], 200)}
        )
        gen = GaussianCopulaGenerator()
        gen.fit(df)
        for n in [1, 10, 50, 100]:
            syn = gen.generate(n, seed=42)
            assert len(syn) == n

    def test_fidelity_report_small_data(self):
        from tabular_polygraph.fidelity import fidelity_report

        real = pd.DataFrame(
            {
                "cat1": ["A", "B"] * 20,
                "cat2": ["X", "Y"] * 20,
            }
        )
        syn = real.sample(frac=1.0, random_state=0).reset_index(drop=True)
        report = fidelity_report(real, syn)
        assert "summary" in report

    def test_empty_filters_passthrough(self):
        df = pd.DataFrame(
            {"a": np.random.randn(100), "b": np.random.choice(["X", "Y"], 100)}
        )
        gen = GaussianCopulaGenerator()
        gen.fit(df)
        syn = gen.generate(50, filters=None, seed=42)
        assert len(syn) == 50
