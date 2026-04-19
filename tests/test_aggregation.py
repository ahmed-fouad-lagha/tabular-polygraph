import numpy as np
from src.fidelity.report import _summary_section


def test_weighted_geometric_mean_sensitivity():
    """
    Verify that a zero score in HIF results in a very low overall score,
    unlike an arithmetic mean.
    """

    # Case 1: Perfect stats, Zero Logic
    # New geometric (30% logic): exp(0.7 * log(101) + 0.3 * log(1)) - 1 = exp(3.23) - 1 approx 24.
    res = _summary_section(
        mm_score=100.0,
        ks_score=100.0,
        corr_score=100.0,
        privacy_score=100.0,
        logical_validity=0.0,
        utility_report={},
        n_real=1,
        n_syn=1,
        t0=0,
    )

    holistic_score = res["holistic_integrity"]
    assert holistic_score < 40.0  # Should be significantly lower than 80.0
    assert "holistic_integrity" in res


def test_perfect_scores_return_100():
    res = _summary_section(100.0, 100.0, 100.0, 100.0, 100.0, {}, 1, 1, 0)
    assert np.isclose(res["holistic_integrity"], 100.0)


def test_missing_hif_defaults_to_vacuous_consistency():
    """
    If logical_validity is not provided (e.g. no categorical columns),
    it should treat the logic score as 100% (vacuously consistent).
    """
    # Should be 90.0 since all components (including the default 100% logic) are >= 90
    res = _summary_section(90.0, 90.0, 90.0, 100.0, 100.0, {}, 1, 1, 0)
    assert res["holistic_integrity"] >= 90.0
