"""
Assembles a complete fidelity report by running all available metrics.
Returns a structured dict suitable for JSON serialisation or CLI display.
"""

import logging
import time
from typing import Any

import pandas as pd

from tabular_polygraph.utils import DEFAULT_DROP_LIST, numeric_columns

from .alpha_beta import alpha_precision_beta_recall
from .downstream import tstr_score
from .joint import correlation_distance_score, pairwise_correlation_report
from .marginal import (
    ks_distribution_scores,
    mean_ks_score,
    mean_moment_matching_score,
    moment_matching_scores,
)
from .tabular_facts import tabular_stylized_facts


def _shared_columns(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str] | None
) -> list[str]:
    # DEFAULT_DROP_LIST: Exclude non-statistical identifiers automatically.
    # Case-insensitive match ensures 'SYN_ID', 'Tract_ID', etc. are dropped.
    drop_lower = {s.lower() for s in DEFAULT_DROP_LIST}
    return columns or [
        c
        for c in real.columns
        if c in synthetic.columns and c.lower() not in drop_lower
    ]


def _downstream_section(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str | None,
    include_downstream: bool,
    seed: int = 42,
) -> dict | None:
    if not (include_downstream and target_col and target_col in real.columns):
        return None

    result = tstr_score(real, synthetic, target_col=target_col, seed=seed)
    if "error" in result:
        return {"status": "skipped", "reason": result["error"]}
    return result


def _logical_section(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    cols: list[str],
    rule_min_confidence: float,
    rule_min_support: float,
    rule_max_rules: int,
    rule_min_lift: float,
    rule_max_antecedents: int,
    hif_epochs: int = 10,
    hif_hubs: int = 5,
    hif_depth: int = 12,
    random_state: int = 42,
    verbose: bool = False,
    progress_callback: Any | None = None,
) -> tuple[dict, float | None]:
    logical_validity: float | None = None

    try:
        from .logical import hif_score

        hif_result = hif_score(
            real,
            synthetic,
            columns=cols,
            verbose=verbose,
            hif_epochs=hif_epochs,
            hif_hubs=hif_hubs,
            hif_depth=hif_depth,
            rule_min_confidence=rule_min_confidence,
            rule_min_support=rule_min_support,
            rule_max_rules=rule_max_rules,
            rule_min_lift=rule_min_lift,
            rule_max_antecedents=rule_max_antecedents,
            random_state=random_state,
            progress_callback=progress_callback,
        )
        logical_validity = round(float(hif_result["hif_score"] * 100.0), 2)
        return {
            "hif_score_pct": logical_validity,
            "hif_violation_rate_pct": round(
                float(hif_result["violation_rate"] * 100.0), 2
            ),
            "mean_penalty_pct": round(float(hif_result["mean_penalty"] * 100.0), 2),
            "num_hif_violations": hif_result["num_violations"],
            "violation_threshold": hif_result.get("violation_threshold"),
            "nic_violation_rate_pct": round(
                float(hif_result.get("nic_violation_rate", 0) * 100.0), 2
            ),
            "columns_used": hif_result["columns_used"],
            "rule_violation_rate_pct": round(
                float(hif_result.get("rule_violation_rate", 0) * 100.0), 2
            ),
            "num_rule_violations": hif_result.get("num_rule_violations", 0),
            "num_rules_mined": hif_result.get("num_rules_mined", 0),
            "top_violated_rules": hif_result.get("top_violated_rules", []),
            "violation_examples": hif_result.get("violation_examples", []),
        }, logical_validity
    except Exception as e:
        if "torch" in str(e).lower():
            return {
                "error": "PyTorch not installed. Install with: pip install torch"
            }, None
        return {"error": str(e)}, None


def _summary_section(
    mm_score: float,
    ks_score: float,
    corr_score: float,
    logical_validity: float | None,
    coverage: dict,
    stylized_mean: float | None,
    tstr_ratio: float | None,
    n_real: int,
    n_syn: int,
    t0: float,
) -> dict:
    """Collect individual metric scores into a summary dict."""

    return {
        "moment_matching_score": mm_score,
        "ks_score": ks_score,
        "joint_score": corr_score,
        "alpha_precision": coverage.get("alpha_precision"),
        "beta_recall": coverage.get("beta_recall"),
        "authenticity": coverage.get("authenticity"),
        "logic_score": round(float(logical_validity), 2)
        if logical_validity is not None
        else None,
        "stylized_facts_score": stylized_mean,
        "tstr_ratio": tstr_ratio,
        "rows_real": n_real,
        "rows_synthetic": n_syn,
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def fidelity_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    dataset_type: str = "cross_sectional",
    target_col: str | None = None,  # for TSTR downstream score
    include_downstream: bool = True,
    columns: list[str] | None = None,
    rule_min_confidence: float = 0.95,
    rule_min_support: float = 0.005,
    rule_max_rules: int = 25,
    rule_min_lift: float = 1.0,
    rule_max_antecedents: int = 2,
    hif_epochs: int = 10,
    hif_hubs: int = 5,
    hif_depth: int = 12,
    random_state: int = 42,
    verbose: bool = False,
    progress_callback: Any | None = None,
) -> dict:
    """
    Run all applicable fidelity metrics and return a structured report.

    Parameters
    ----------
    real, synthetic  : DataFrames to compare
    dataset_type     : label stored in report output
    target_col       : if given, runs TSTR downstream evaluation
    include_downstream : run TSTR if target_col is provided
    columns          : restrict to these columns (default: all shared)

    Returns
    -------
    Nested dict with sections: moment_matching, distribution_fit, joint, stylized_facts, downstream (optional),
    logical, summary.
    """
    t0 = time.time()

    cols = _shared_columns(real, synthetic, columns)

    # Autonomous Alignment: Ensure both datasets operate in the exact same feature space.
    real = real[cols].copy()
    syn = synthetic[cols].copy()

    report: dict = {"dataset_type": dataset_type, "columns_evaluated": cols}

    # ── Joint ─────────────────────────────────────────────────────────────────
    num_cols = [c for c in cols if c in numeric_columns(real)]
    corr_score = correlation_distance_score(real, syn, num_cols)
    report["joint"] = {
        "correlation_distance_score": corr_score,
        "pairwise_deltas": pairwise_correlation_report(real, syn, num_cols),
    }

    # ── Alpha-precision / Beta-recall (geometric coverage) ───────────────────
    try:
        n_min = min(len(real), len(syn), 10000)
        if n_min >= 10:
            ab_result = alpha_precision_beta_recall(
                real.sample(n_min, random_state=random_state).reset_index(drop=True),
                syn.sample(n_min, random_state=random_state + 1).reset_index(drop=True),
            )
        else:
            ab_result = {
                "alpha_precision": None,
                "beta_recall": None,
                "authenticity": None,
            }
    except Exception as e:
        logging.warning(f"Alpha-precision/Beta-recall computation failed: {e}")
        ab_result = {"alpha_precision": None, "beta_recall": None, "authenticity": None}
    report["coverage"] = ab_result

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
    report["stylized_facts"] = tabular_stylized_facts(real, syn, num_cols)

    # ── Downstream ────────────────────────────────────────────────────────────
    downstream_report = _downstream_section(
        real, syn, target_col, include_downstream, seed=random_state
    )
    if downstream_report is not None:
        report["downstream"] = downstream_report

    logical_report, logical_validity = _logical_section(
        real,
        syn,
        cols,
        rule_min_confidence,
        rule_min_support,
        rule_max_rules,
        rule_min_lift,
        rule_max_antecedents,
        hif_epochs=hif_epochs,
        hif_hubs=hif_hubs,
        hif_depth=hif_depth,
        random_state=random_state,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    report["logical"] = logical_report

    # ── Summary ───────────────────────────────────────────────────────────────
    mm_score = report["moment_matching"]["mean_score"]
    ks_score = report["distribution_fit"]["mean_score"]

    stylized_mean = (
        report.get("stylized_facts", {}).get("_summary", {}).get("mean_score")
    )
    tstr_ratio = report.get("downstream", {}).get("ratio")

    report["summary"] = _summary_section(
        mm_score=mm_score,
        ks_score=ks_score,
        corr_score=corr_score,
        logical_validity=logical_validity,
        coverage=report["coverage"],
        stylized_mean=stylized_mean,
        tstr_ratio=tstr_ratio,
        n_real=len(real),
        n_syn=len(syn),
        t0=t0,
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
    lines.append("-" * width)
    lines.append(f"  Moment matching : {s.get('moment_matching_score', 0):>6.2f}%")
    lines.append(f"  KS distribution : {s.get('ks_score', 0):>6.2f}%")
    lines.append(f"  Joint distance  : {s.get('joint_score', 0):>6.2f}%")
    ap = s.get("alpha_precision")
    br = s.get("beta_recall")
    au = s.get("authenticity")
    if ap is not None:
        lines.append(f"  Alpha-precision : {ap:>6.3f}")
        lines.append(f"  Beta-recall     : {br:>6.3f}")
        lines.append(f"  Authenticity    : {au:>6.3f}")
    l_score = s.get("logic_score")
    if l_score is not None:
        lines.append(f"  HIF (structural): {l_score:>6.2f}%")
    lines.append("-" * width)
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
        if d.get("status") == "skipped":
            lines.append(f"    Skipped: {d.get('reason')}")
        else:
            lines.append(f"    Target  : {d.get('target_col')}")
            lines.append(
                f"    Metric  : {d.get('metric')} | TSTR {d.get('tstr_score')} | TRR {d.get('trr_score')}"
            )
            lines.append(f"    Ratio   : {d.get('ratio')}")

    lg = report.get("logical", {})
    lines.append("")
    lines.append("  Logical:")

    if "error" in lg:
        lines.append(f"    Status         : ERROR ({lg['error']})")
    else:
        lines.append(
            f"    Unified violation rate : {lg.get('hif_violation_rate_pct', '—')}%"
        )
        lines.append(
            f"    NIC (Continuous) rate  : {lg.get('nic_violation_rate_pct', '—')}%"
        )
        lines.append(
            f"    Rule violation rate    : {lg.get('rule_violation_rate_pct', '—')}%"
        )
        lines.append(f"    Mean penalty           : {lg.get('mean_penalty_pct', '—')}%")
        lines.append(
            f"    Noise floor threshold  : {lg.get('violation_threshold', '—')}"
        )
        lines.append(
            f"    Violations found       : {lg.get('num_hif_violations', '—')} (rules mined: {lg.get('num_rules_mined', '—')})"
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
