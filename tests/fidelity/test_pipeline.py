from tabular_polygraph._types.report import (
    CoverageScore,
    DownstreamScore,
    FidelityReport,
    JointScore,
    LogicalScore,
    PerColumnScore,
    StylizedFactsScore,
)
from tabular_polygraph.fidelity.pipeline import _build_summary


def _make_report(
    mm_score: float = 0.0,
    ks_score: float = 0.0,
    jt_score: float = 0.0,
    lv: float | None = None,
    coverage: dict | None = None,
    sf_score: float | None = None,
    ds_ratio: float | None = None,
) -> FidelityReport:
    cov = CoverageScore(
        alpha_precision=coverage.get("alpha_precision") if coverage else None,
        beta_recall=coverage.get("beta_recall") if coverage else None,
        authenticity=coverage.get("authenticity") if coverage else None,
    )
    ds = DownstreamScore(ratio=ds_ratio) if ds_ratio is not None else None
    sf = StylizedFactsScore(mean_score=sf_score) if sf_score is not None else None
    lg = LogicalScore(hif_score_pct=lv) if lv is not None else None
    return FidelityReport(
        moment_matching=PerColumnScore(columns={}, mean=mm_score),
        distribution_fit=PerColumnScore(columns={}, mean=ks_score),
        joint=JointScore(correlation_distance_score=jt_score),
        coverage=cov,
        stylized_facts=sf,
        downstream=ds,
        logical=lg,
    )


def test_summary_section_basic():
    report = _make_report(mm_score=100.0, ks_score=100.0, jt_score=100.0, lv=0.0)
    s = _build_summary(report)

    assert s.moment_matching_score == 100.0
    assert s.ks_score == 100.0
    assert s.joint_score == 100.0
    assert s.logic_score == 0.0
    assert s.stylized_facts_score is None
    assert s.tstr_ratio is None


def test_perfect_scores():
    report = _make_report(
        mm_score=100.0,
        ks_score=100.0,
        jt_score=100.0,
        lv=100.0,
        sf_score=90.0,
        ds_ratio=0.85,
    )
    s = _build_summary(report)
    assert s.moment_matching_score == 100.0
    assert s.logic_score == 100.0
    assert s.stylized_facts_score == 90.0
    assert s.tstr_ratio == 0.85


def test_missing_hif_defaults_to_none():
    report = _make_report(mm_score=90.0, ks_score=90.0, jt_score=90.0)
    s = _build_summary(report)
    assert s.logic_score is None


def test_coverage_scores_in_summary():
    report = _make_report(
        mm_score=95.0,
        ks_score=90.0,
        jt_score=85.0,
        lv=80.0,
        coverage={"alpha_precision": 0.92, "beta_recall": 0.33, "authenticity": 0.64},
        sf_score=72.5,
        ds_ratio=0.91,
    )
    s = _build_summary(report)
    assert s.alpha_precision == 0.92
    assert s.beta_recall == 0.33
    assert s.authenticity == 0.64
    assert s.stylized_facts_score == 72.5
    assert s.tstr_ratio == 0.91
