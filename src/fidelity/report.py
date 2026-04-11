"""
Assembles a complete fidelity report by running all available metrics.
Returns a structured dict suitable for JSON serialisation or CLI display.
"""

from __future__ import annotations
import time
import pandas as pd
from src.utils import numeric_columns

from .marginal import (
    moment_matching_scores,
    mean_moment_matching_score,
    ks_distribution_scores,
    mean_ks_score,
)
from .joint import correlation_distance_score, pairwise_correlation_report
from .stylized_facts import stylized_facts_score
from .causality import causality_score


def _shared_columns(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str] | None
) -> list[str]:
    return columns or [
        c for c in real.columns if c in synthetic.columns and c != "syn_id"
    ]


def _stylized_facts_section(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    num_cols: list[str],
    dataset_type: str,
) -> dict:
    if dataset_type in ("time_series", "panel"):
        return stylized_facts_score(real, synthetic, num_cols)

    return {
        "_summary": {
            "mean_score": None,
            "columns_tested": 0,
            "applicable": False,
            "note": "Stylized facts are not evaluated for cross-sectional data.",
        }
    }


def _temporal_section(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    cols: list[str],
    dataset_type: str,
    include_temporal: bool | None,
) -> dict | None:
    do_temporal = (
        include_temporal
        if include_temporal is not None
        else dataset_type in ("time_series", "panel")
    )
    if not do_temporal:
        return None

    from .temporal.stationarity import stationarity_score
    from .temporal.cointegration import cointegration_score
    from .temporal.breaks import breaks_score

    return {
        "stationarity": stationarity_score(real, synthetic, cols),
        "cointegration": cointegration_score(real, synthetic),
        "breaks": breaks_score(real, synthetic, cols),
        "causality": causality_score(real, synthetic),
    }


def _downstream_section(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str | None,
    include_downstream: bool,
) -> dict | None:
    if not (include_downstream and target_col and target_col in real.columns):
        return None

    from .downstream import tstr_score

    return tstr_score(real, synthetic, target_col=target_col)


def _logical_section(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    cols: list[str],
    include_logical: bool,
    rule_min_confidence: float,
    rule_min_support: float,
    rule_max_rules: int,
    rule_min_lift: float,
    rule_max_antecedents: int,
) -> tuple[dict | None, float | None]:
    logical_validity = None
    if not include_logical:
        return None, logical_validity

    try:
        from .logical import lcv_score, rule_violation_score

        cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(real[c])]
        if not cat_cols:
            return None, logical_validity

        lcv_result = lcv_score(
            real, synthetic, columns=cat_cols, epochs=20, verbose=False
        )
        rule_result = rule_violation_score(
            real,
            synthetic,
            columns=cat_cols,
            min_confidence=rule_min_confidence,
            min_support=rule_min_support,
            max_rules=rule_max_rules,
            min_lift=rule_min_lift,
            max_antecedents=rule_max_antecedents,
        )
        logical_validity = round(float(lcv_result["lcv_score"] * 100.0), 2)
        return {
            "lcv_score_pct": round(float(lcv_result["lcv_score"] * 100.0), 2),
            "lcv_violation_rate_pct": round(
                float(lcv_result["violation_rate"] * 100.0), 2
            ),
            "mean_penalty_pct": round(float(lcv_result["mean_penalty"] * 100.0), 2),
            "num_lcv_violations": lcv_result["num_violations"],
            "columns_used": lcv_result["columns_used"],
            "rule_violation_rate_pct": round(
                float(rule_result["rule_violation_rate"] * 100.0), 2
            ),
            "num_rule_violations": rule_result["num_rule_violations"],
            "num_rules_mined": rule_result["num_rules_mined"],
            "rows_with_rule_violations": rule_result["rows_with_rule_violations"],
            "rows_evaluated": rule_result["rows_evaluated"],
            "example_rules": rule_result.get("example_rules", []),
            "top_violated_rules": rule_result.get("top_violated_rules", []),
            "violation_examples": rule_result.get("violation_examples", []),
        }, logical_validity
    except Exception as e:
        if "torch" in str(e).lower():
            return {
                "error": "PyTorch not installed. Install with: pip install torch"
            }, logical_validity
        return {"error": str(e)}, logical_validity


def _summary_section(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    mm_score: float,
    ks_score: float,
    corr_score: float,
    privacy_score: float,
    exact_copies: int,
    t0: float,
    logical_validity: float | None,
) -> dict:
    summary_dict = {
        "overall_fidelity": round(
            float(0.45 * mm_score + 0.30 * ks_score + 0.25 * corr_score), 2
        ),
        "moment_matching_score": mm_score,
        "ks_score": ks_score,
        "joint_score": corr_score,
        "privacy_score": privacy_score,
        "exact_copies": exact_copies,
        "rows_real": len(real),
        "rows_synthetic": len(synthetic),
        "elapsed_seconds": round(time.time() - t0, 3),
    }

    if logical_validity is not None:
        summary_dict["logical_validity"] = logical_validity

    return summary_dict


def fidelity_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    dataset_type: str = "cross_sectional",  # cross_sectional | time_series | panel
    target_col: str | None = None,  # for TSTR downstream score
    include_temporal: bool | None = None,  # auto-detect from dataset_type
    include_downstream: bool = True,
    include_logical: bool = True,  # LCV logical validation
    columns: list[str] | None = None,
    rule_min_confidence: float = 0.95,
    rule_min_support: float = 0.005,
    rule_max_rules: int = 25,
    rule_min_lift: float = 1.0,
    rule_max_antecedents: int = 1,
) -> dict:
    """
    Run all applicable fidelity metrics and return a structured report.

    Parameters
    ----------
    real, synthetic  : DataFrames to compare
    dataset_type     : drives which temporal tests are included
    target_col       : if given, runs TSTR downstream evaluation
    include_temporal : override temporal test inclusion
    include_downstream : run TSTR if target_col is provided
    include_logical  : run LCV for logical constraint validation
    columns          : restrict to these columns (default: all shared)

    Returns
    -------
    Nested dict with sections: moment_matching, distribution_fit, joint, temporal (optional),
    stylized_facts, downstream (optional), logical (optional), summary.
    """
    t0 = time.time()

    cols = _shared_columns(real, synthetic, columns)
    syn = synthetic.drop(columns=["syn_id"], errors="ignore")

    report: dict = {"dataset_type": dataset_type, "columns_evaluated": cols}

    # ── Joint ─────────────────────────────────────────────────────────────────
    num_cols = [c for c in cols if c in numeric_columns(real)]
    corr_score = correlation_distance_score(real, syn, num_cols)
    report["joint"] = {
        "correlation_distance_score": corr_score,
        "pairwise_deltas": pairwise_correlation_report(real, syn, num_cols),
    }

    # ── Moment matching + KS fit ──────────────────────────────────────────────
    mm_scores = moment_matching_scores(real, syn, num_cols)
    ks_scores = ks_distribution_scores(real, syn, num_cols)
    report["moment_matching"] = {
        "column_scores": mm_scores,
        "mean_score": mean_moment_matching_score(mm_scores),
    }
    report["distribution_fit"] = {
        "column_scores": ks_scores,
        "mean_score": mean_ks_score(ks_scores),
    }

    # ── Stylized facts ────────────────────────────────────────────────────────
    report["stylized_facts"] = _stylized_facts_section(
        real, syn, num_cols, dataset_type
    )

    # ── Temporal (time series / panel only) ───────────────────────────────────
    temporal_report = _temporal_section(real, syn, cols, dataset_type, include_temporal)
    if temporal_report is not None:
        report["temporal"] = temporal_report

    # ── Downstream ────────────────────────────────────────────────────────────
    downstream_report = _downstream_section(real, syn, target_col, include_downstream)
    if downstream_report is not None:
        report["downstream"] = downstream_report

    # ── Privacy (basic — full audit is in privacy/audit.py) ──────────────────
    real_hashes = set(real[cols].astype(str).apply("|".join, axis=1))
    syn_hashes = (
        syn[[c for c in cols if c in syn.columns]].astype(str).apply("|".join, axis=1)
    )
    exact_copies = int(syn_hashes.isin(real_hashes).sum())
    report["privacy_basic"] = {
        "exact_copies": exact_copies,
        "privacy_score": round((1 - exact_copies / max(len(syn), 1)) * 100, 2),
    }

    # ── Logical Constraint Validation (LCV) ────────────────────────────
    logical_report, logical_validity = _logical_section(
        real,
        syn,
        cols,
        include_logical,
        rule_min_confidence,
        rule_min_support,
        rule_max_rules,
        rule_min_lift,
        rule_max_antecedents,
    )
    if logical_report is not None:
        report["logical"] = logical_report

    # ── Summary ───────────────────────────────────────────────────────────────
    mm_score = report["moment_matching"]["mean_score"]
    ks_score = report["distribution_fit"]["mean_score"]
    report["summary"] = _summary_section(
        real,
        syn,
        mm_score,
        ks_score,
        corr_score,
        report["privacy_basic"]["privacy_score"],
        exact_copies,
        t0,
        logical_validity,
    )

    return report


def format_report(report: dict, width: int = 60) -> str:
    """Return a human-readable string summary of a fidelity report."""
    lines = []
    s = report.get("summary", {})

    lines.append("=" * width)
    lines.append("  FIDELITY REPORT")
    lines.append("=" * width)
    lines.append(f"  Dataset type    : {report.get('dataset_type', '—')}")
    lines.append(
        f"  Rows (real/syn) : {s.get('rows_real', '?')} / {s.get('rows_synthetic', '?')}"
    )
    lines.append("")
    lines.append(f"  Overall fidelity: {s.get('overall_fidelity', '—')}%")
    lines.append(f"  Moment matching : {s.get('moment_matching_score', '—')}%")
    lines.append(f"  KS distribution : {s.get('ks_score', '—')}%")
    lines.append(f"  Joint score     : {s.get('joint_score', '—')}%")
    lines.append(f"  Privacy score   : {s.get('privacy_score', '—')}%")
    logical_validity = s.get("logical_validity")
    if logical_validity is None:
        lines.append("  Logical validity: —")
    else:
        lines.append(f"  Logical validity: {logical_validity}%")
    lines.append(f"  Exact copies    : {s.get('exact_copies', '—')}")
    lines.append("")

    sf_summary = report.get("stylized_facts", {}).get("_summary", {})
    lines.append("  Stylized facts:")
    if sf_summary.get("applicable", True):
        lines.append(f"    Mean score  : {sf_summary.get('mean_score', '—')}%")
        lines.append(f"    Columns     : {sf_summary.get('columns_tested', '—')}")
    else:
        lines.append(f"    {sf_summary.get('note', 'Not evaluated.')}")
    lines.append("")

    mm_cols = report.get("moment_matching", {}).get("column_scores", {})
    if mm_cols:
        lines.append("")
        lines.append("  Per-column moment matching scores:")
        for col, sc in mm_cols.items():
            bar = "█" * int(sc / 5) + "░" * (20 - int(sc / 5))
            lines.append(f"    {col:<26} {bar}  {sc}%")

    ks_cols = report.get("distribution_fit", {}).get("column_scores", {})
    if ks_cols:
        lines.append("")
        lines.append("  Per-column KS distribution scores:")
        for col, sc in ks_cols.items():
            bar = "█" * int(sc / 5) + "░" * (20 - int(sc / 5))
            lines.append(f"    {col:<26} {bar}  {sc}%")

    if "downstream" in report:
        d = report["downstream"]
        lines.append("")
        lines.append("  Downstream (TSTR):")
        lines.append(f"    Target  : {d.get('target_col')}")
        lines.append(
            f"    Metric  : {d.get('metric')} | TSTR {d.get('tstr_score')} | TRR {d.get('trr_score')}"
        )
        lines.append(
            f"    Ratio   : {d.get('ratio')}  — {d.get('interpretation', '')[:50]}"
        )

    if "logical" in report:
        lg = report["logical"]
        lines.append("")
        lines.append("  Logical (Rules + LCV):")
        lines.append(f"    LCV score      : {lg.get('lcv_score_pct', '—')}%")
        lines.append(
            f"    LCV violation rate : {lg.get('lcv_violation_rate_pct', '—')}%"
        )
        lines.append(
            f"    Rule violation rate  : {lg.get('rule_violation_rate_pct', '—')}%"
        )
        lines.append(f"    Mean penalty         : {lg.get('mean_penalty_pct', '—')}%")
        lines.append(
            f"    Rule violations      : {lg.get('num_rule_violations', '—')} (rules mined: {lg.get('num_rules_mined', '—')})"
        )

        top_rules = lg.get("top_violated_rules", [])
        if top_rules:
            lines.append("    Top violated rules:")
            for rule in top_rules[:5]:
                ant = rule.get("antecedent_repr")
                if not ant:
                    ant_feat = rule.get("antecedent_feature")
                    ant_val = rule.get("antecedent_value")
                    ant = (
                        f"{ant_feat}={ant_val}"
                        if ant_feat is not None
                        else "(unknown antecedent)"
                    )
                lines.append(
                    "      "
                    + f"IF {ant} "
                    + f"THEN {rule.get('consequent_feature')}={rule.get('consequent_value')} "
                    + f"| conf={rule.get('confidence')} lift={rule.get('lift', '—')} support={rule.get('support')} "
                    + f"violations={rule.get('violation_count')}"
                )

        examples = lg.get("violation_examples", [])
        if examples:
            lines.append("    Example violating rows:")
            for ex in examples[:5]:
                lines.append(
                    "      "
                    + f"row={ex.get('row_index')} | {ex.get('antecedent')} | "
                    + f"expected {ex.get('expected')} | actual {ex.get('actual')}"
                )

    lines.append("")
    lines.append(f"  Computed in {s.get('elapsed_seconds', '?')}s")
    lines.append("=" * width)
    return "\n".join(lines)
