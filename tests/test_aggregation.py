from tabular_polygraph.fidelity.report import _summary_section


def test_summary_section_basic():
    res = _summary_section(
        mm_score=100.0,
        ks_score=100.0,
        corr_score=100.0,
        logical_validity=0.0,
        coverage={},
        utility_report={},
        n_real=1,
        n_syn=1,
        t0=0,
    )

    assert res["moment_matching_score"] == 100.0
    assert res["ks_score"] == 100.0
    assert res["joint_score"] == 100.0
    assert res["logic_score"] == 0.0
    assert "rows_real" in res


def test_perfect_scores():
    res = _summary_section(100.0, 100.0, 100.0, 100.0, {}, {}, 1, 1, 0)
    assert res["moment_matching_score"] == 100.0
    assert res["logic_score"] == 100.0


def test_missing_hif_defaults_to_none():
    res = _summary_section(90.0, 90.0, 90.0, None, {}, {}, 1, 1, 0)
    assert res["logic_score"] is None


def test_coverage_scores_in_summary():
    res = _summary_section(
        mm_score=95.0,
        ks_score=90.0,
        corr_score=85.0,
        logical_validity=80.0,
        coverage={"alpha_precision": 0.92, "beta_recall": 0.33, "authenticity": 0.64},
        utility_report={},
        n_real=100,
        n_syn=100,
        t0=0,
    )
    assert res["alpha_precision"] == 0.92
    assert res["beta_recall"] == 0.33
    assert res["authenticity"] == 0.64
