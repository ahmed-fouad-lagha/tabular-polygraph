"""Golden-value tests: metrics compared against independently computed
references. These pin the metric definitions so bugs like the JCD
sign-blindness or the inverted authenticity cannot silently return to the
manuscript numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.metrics.alpha_beta import AlphaBeta
from tabular_polygraph.fidelity.metrics.correlation import Correlation
from tabular_polygraph.fidelity.metrics.ks import KSTest
from tabular_polygraph.fidelity.metrics.moment_matching import MomentMatching
from tabular_polygraph.fidelity.metrics.tvd import TVD


def test_ks_identical_scores_one():
    x = np.linspace(0, 1, 100)
    res = KSTest().compute(pd.DataFrame({"a": x}), pd.DataFrame({"a": x}), ["a"])
    assert res["column_scores"]["a"] == 1.0


def test_ks_disjoint_scores_zero():
    res = KSTest().compute(
        pd.DataFrame({"a": np.zeros(50)}),
        pd.DataFrame({"a": np.ones(50)}),
        ["a"],
    )
    assert res["column_scores"]["a"] == 0.0


def test_ks_shifted_golden_value():
    r = np.linspace(0, 1, 100)
    s = np.linspace(0.1, 1.1, 100)
    res = KSTest().compute(pd.DataFrame({"a": r}), pd.DataFrame({"a": s}), ["a"])
    assert res["column_scores"]["a"] == 0.9


def test_tvd_hand_computed_golden_value():
    real = pd.DataFrame({"c": ["A"] * 70 + ["B"] * 20 + ["C"] * 10})
    syn = pd.DataFrame({"c": ["A"] * 50 + ["B"] * 30 + ["C"] * 20})
    res = TVD().compute(real, syn, ["c"])
    assert res["column_scores"]["c"] == 80.0


def test_tvd_identical_scores_one_hundred():
    real = pd.DataFrame({"c": ["A"] * 60 + ["B"] * 40})
    res = TVD().compute(real, real.copy(), ["c"])
    assert res["column_scores"]["c"] == 100.0


def test_tvd_disjoint_support_scores_zero():
    res = TVD().compute(
        pd.DataFrame({"c": ["A"] * 20}),
        pd.DataFrame({"c": ["B"] * 20}),
        ["c"],
    )
    assert res["column_scores"]["c"] == 0.0


def test_jcd_identical_scores_one_hundred():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    real = pd.DataFrame({"a": x, "b": x + rng.normal(0, 0.01, 200)})
    res = Correlation().compute(real, real.copy(), ["a", "b"])
    assert res["correlation_distance_score"] == 100.0


def test_jcd_perfectly_inverted_is_penalized():
    """Regression test for the sign-blindness bug: abs() on the numeric block
    made a perfectly inverted correlation score 100.0."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    real = pd.DataFrame({"a": x, "b": x})
    syn = pd.DataFrame({"a": x, "b": -x})
    res = Correlation().compute(real, syn, ["a", "b"])
    assert res["correlation_distance_score"] == 0.0


def test_jcd_numeric_column_read_as_text():
    """Regression test for per-frame dtype inference: a numeric column read
    as strings from CSV must not collapse the synthetic matrix to all ones."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    real = pd.DataFrame({"a": x, "b": x + rng.normal(0, 0.1, 200)})
    syn = pd.DataFrame(
        {"a": x.astype(str), "b": (x + rng.normal(0, 0.1, 200)).astype(str)}
    )
    res = Correlation().compute(real, syn, ["a", "b"])
    assert res["correlation_distance_score"] > 90.0


def test_moment_matching_identical_is_location_invariant():
    """Identical mean-centered data must score ~100; the old |mean| divisor
    scored identical N(0,1) data at 59."""
    x = np.random.default_rng(0).normal(size=5000)
    res = MomentMatching().compute(
        pd.DataFrame({"c": x}), pd.DataFrame({"c": x}), ["c"]
    )
    assert res["column_scores"]["c"] == 100.0
    res_shift = MomentMatching().compute(
        pd.DataFrame({"c": x + 1000}), pd.DataFrame({"c": x + 1000}), ["c"]
    )
    assert res_shift["column_scores"]["c"] == 100.0


def _alpha_beta_case(real: pd.DataFrame, syn: pd.DataFrame) -> dict:
    return AlphaBeta(random_state=0).compute(real, syn, real.columns.tolist())


def test_alpha_beta_identical():
    rng = np.random.default_rng(3)
    n = 1500
    real = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "g": rng.integers(0, 4, size=n).astype(str),
        }
    )
    res = _alpha_beta_case(real, real.copy())
    assert res["alpha_precision"] > 0.99
    assert res["beta_recall"] > 0.99
    assert res["authenticity"] > 0.99


def test_alpha_beta_authenticity_discriminates_mode_collapse():
    """Regression test for the inverted-authenticity bug: authenticity must
    fall sharply when synthetic data collapses onto a corner of the real
    support."""
    rng = np.random.default_rng(3)
    n = 1500
    real = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "g": rng.integers(0, 4, size=n).astype(str),
        }
    )
    collapsed = pd.DataFrame(
        {
            "a": np.abs(rng.normal(size=n)),
            "b": np.abs(rng.normal(size=n)),
            "g": np.zeros(n, dtype=int).astype(str),
        }
    )
    res = _alpha_beta_case(real, collapsed)
    assert res["beta_recall"] < 0.7
    assert res["authenticity"] < 0.2
