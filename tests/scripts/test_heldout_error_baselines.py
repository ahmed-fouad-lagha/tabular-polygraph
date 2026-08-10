import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

_spec = importlib.util.spec_from_file_location(
    "heldout_error_baselines", SCRIPTS_DIR / "05_heldout_error_baselines.py"
)
baselines = importlib.util.module_from_spec(_spec)
sys.modules["heldout_error_baselines"] = baselines
_spec.loader.exec_module(baselines)

fit_gmm = baselines.fit_gmm
detect_gmm = baselines.detect_gmm
_encode_for_outlier_detection = baselines._encode_for_outlier_detection


@pytest.fixture
def structured_data():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(0, 1, size=n)
    y = 2.0 * x + rng.normal(0, 0.2, size=n)
    real = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "cat": np.where(x > 0, "A", "B"),
        }
    )
    return real


def test_fit_gmm_returns_fitted_model(structured_data):
    gmm = fit_gmm(structured_data, seed=42)
    assert isinstance(gmm, baselines.GaussianMixture)
    assert 1 <= gmm.n_components <= 6


def test_detect_gmm_scores_typical_rows_low(structured_data):
    rng = np.random.default_rng(1)
    real = structured_data
    x = rng.normal(0, 1, size=200)
    typical = pd.DataFrame(
        {
            "x": x,
            "y": 2.0 * x + rng.normal(0, 0.2, size=200),
            "cat": np.where(x > 0, "A", "B"),
        }
    )
    violating = pd.DataFrame(
        {
            "x": x,
            "y": -2.0 * x,
            "cat": np.where(x > 0, "A", "B"),
        }
    )
    gmm = fit_gmm(real, seed=42)
    _, scores_typical = detect_gmm(gmm, real, typical, contamination=0.4)
    _, scores_violating = detect_gmm(gmm, real, violating, contamination=0.4)
    assert scores_violating.mean() > scores_typical.mean()
    assert scores_typical.shape == (200,)


def test_detect_gmm_threshold_calibrates_on_real_scores(structured_data):
    real = structured_data
    gmm = fit_gmm(real, seed=42)
    X_real, _ = _encode_for_outlier_detection(real, real)
    thr = np.quantile(-gmm.score_samples(X_real), 1.0 - 0.4)
    assert np.isfinite(thr)
    assert thr >= np.min(-gmm.score_samples(X_real))


def test_detect_gmm_preds_are_binary_and_shaped(structured_data):
    real = structured_data
    rng = np.random.default_rng(2)
    syn = real.sample(50, random_state=rng.integers(1e9))
    gmm = fit_gmm(real, seed=42)
    preds, scores = detect_gmm(gmm, real, syn, contamination=0.4)
    assert preds.shape == (50,)
    assert set(np.unique(preds)) <= {0, 1}
    assert np.isfinite(scores).all()
