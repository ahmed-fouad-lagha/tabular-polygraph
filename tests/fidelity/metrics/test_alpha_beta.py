import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.metrics.alpha_beta import AlphaBeta, _encode


class TestAlphaPrecisionBetaRecall:
    def test_identical_data_high_scores(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 2, 200)})
        result = AlphaBeta().compute(df, df, df.columns.tolist())
        assert result["alpha_precision"] > 0.9
        assert result["beta_recall"] > 0.9

    def test_random_vs_structured_low_precision(self):
        rng = np.random.default_rng(42)
        real = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(5, 2, 300)})
        syn = pd.DataFrame({"a": rng.normal(10, 5, 300), "b": rng.normal(-5, 10, 300)})
        result = AlphaBeta().compute(real, syn, real.columns.tolist())
        assert result["alpha_precision"] < 0.8

    def test_too_few_rows_raises(self):
        real = pd.DataFrame({"a": [1, 2, 3]})
        syn = pd.DataFrame({"a": [1, 2]})
        err = AlphaBeta().validate(real, syn)
        assert err is not None
        assert "Too few real rows" in err

    def test_all_scores_in_range(self):
        rng = np.random.default_rng(7)
        real = pd.DataFrame(
            {
                "x": rng.uniform(0, 10, 150),
                "y": rng.choice(["p", "q", "r"], 150),
            }
        )
        syn = real.sample(frac=1.0, random_state=0).reset_index(drop=True)
        result = AlphaBeta().compute(real, syn, real.columns.tolist())
        assert 0 <= result["alpha_precision"] <= 1
        assert 0 <= result["beta_recall"] <= 1
        assert 0 <= result["authenticity"] <= 1

    def test_encode_handles_mixed_types(self):
        real = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        syn = pd.DataFrame({"a": [4, 5, 6], "b": ["x", "y", "z"]})
        X, X_s = _encode(real, syn)
        assert X.shape == (3, 4)
        assert X_s.shape == (3, 4)
        assert X.dtype == float
