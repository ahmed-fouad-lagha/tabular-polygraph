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
    # Old arithmetic (20% logic): 0.8 * 100 + 0.2 * 0 = 80.0
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

    low_score = res["overall_fidelity"]
    assert low_score < 40.0  # Should be significantly lower than 80.0
    assert res["is_holistic"] is True


def test_holistic_vs_marginal_labeling():
    df = pd.DataFrame({"a": [1]})

    # With LCV
    res_h = _summary_section(df, df, 90, 90, 90, 100, 0, 0, 90)
    assert res_h["is_holistic"] is True

    # Without LCV
    res_m = _summary_section(df, df, 90, 90, 90, 100, 0, 0, None)
    assert res_m["is_holistic"] is False
    assert "overall_fidelity" in res_m


def test_perfect_scores_return_100():
    df = pd.DataFrame({"a": [1]})
    res = _summary_section(df, df, 100.0, 100.0, 100.0, 100.0, 0, 0, 100.0)
    assert np.isclose(res["overall_fidelity"], 100.0)
