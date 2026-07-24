from tabular_polygraph.fidelity.report import _summary_section


def test_summary_section_basic():
    res = _summary_section(
        mm_score=100.0,
        ks_score=100.0,
        corr_score=100.0,
        logical_validity=0.0,
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
    res = _summary_section(100.0, 100.0, 100.0, 100.0, {}, 1, 1, 0)
    assert res["moment_matching_score"] == 100.0
    assert res["logic_score"] == 100.0


def test_missing_hif_defaults_to_none():
    res = _summary_section(90.0, 90.0, 90.0, None, {}, 1, 1, 0)
    assert res["logic_score"] is None
