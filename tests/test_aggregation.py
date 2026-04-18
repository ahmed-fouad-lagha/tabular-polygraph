import numpy as np
from src.fidelity.report import _summary_section
import pandas as pd


def test_weighted_geometric_mean_sensitivity():
    """
    Verify that a zero score in LCV results in a very low overall score,
    unlike an arithmetic mean.
    """
    df = pd.DataFrame({"a": [1]})  # dummy

    # Case 1: Perfect stats, Zero Logic
    # New geometric (30% logic): exp(0.7 * log(101) + 0.3 * log(1)) - 1 = exp(3.23) - 1 approx 24.
    res = _summary_section(
        real=df,
        synthetic=df,
        mm_score=100.0,
        ks_score=100.0,
        corr_score=100.0,
        privacy_score=100.0,
        exact_copies=0,
        t0=0,
        logical_validity=0.0,
    )

    holistic_score = res["holistic_integrity"]
    assert holistic_score < 40.0  # Should be significantly lower than 80.0
    assert "holistic_integrity" in res


def test_perfect_scores_return_100():
    df = pd.DataFrame({"a": [1]})
    res = _summary_section(df, df, 100.0, 100.0, 100.0, 100.0, 0, 0, 100.0)
    assert np.isclose(res["holistic_integrity"], 100.0)


def test_missing_lcv_defaults_to_vacuous_consistency():
    """
    If logical_validity is not provided (e.g. no categorical columns),
    it should treat the logic score as 100% (vacuously consistent).
    """
    df = pd.DataFrame({"a": [1]})
    res = _summary_section(df, df, 90.0, 90.0, 90.0, 100.0, 0, 0, None)

    # Should be 90.0 since all components (including the default 100% logic) are >= 90
    assert res["holistic_integrity"] >= 90.0
