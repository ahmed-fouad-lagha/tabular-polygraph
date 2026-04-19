"""
Assembles a complete fidelity report by running all available metrics.
Returns a structured dict suitable for JSON serialisation or CLI display.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from src.utils import numeric_columns, DEFAULT_DROP_LIST

from .marginal import (
    moment_matching_scores,
    mean_moment_matching_score,
    ks_distribution_scores,
    mean_ks_score,
)
from .joint import correlation_distance_score, pairwise_correlation_report
from .stylized_facts import stylized_facts_score
from .tabular_facts import tabular_stylized_facts
from .causality import causality_score
from .downstream import tstr_score


def _shared_columns(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str] | None
) -> list[str]:
    # DEFAULT_DROP_LIST: Exclude non-statistical identifiers automatically
    return columns or [
        c
        for c in real.columns
        if c in synthetic.columns and c.lower() not in DEFAULT_DROP_LIST
    ]


def _stylized_facts_section(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    num_cols: list[str],
    dataset_type: str,
) -> dict:
    if dataset_type in ("time_series", "panel"):
        return stylized_facts_score(real, synthetic, num_cols)

    # For cross-sectional data, use the new tabular facts engine
    return tabular_stylized_facts(real, synthetic, num_cols)


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
    seed: int = 42,
) -> dict | None:
    if not (include_downstream and target_col and target_col in real.columns):
        return None

    return tstr_score(real, synthetic, target_col=target_col, seed=seed)


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
) -> tuple[dict, float]:
    logical_validity = 100.0

    try:
        from .logical import hif_score, rule_violation_score

        hif_result = hif_score(
            real,
            synthetic,
            columns=cols,
            verbose=True,
            hif_epochs=hif_epochs,
            hif_hubs=hif_hubs,
            hif_depth=hif_depth,
            random_state=random_state,
        )
        rule_result = rule_violation_score(
            real,
            synthetic,
            columns=cols,
            min_confidence=rule_min_confidence,
            min_support=rule_min_support,
            max_rules=rule_max_rules,
            min_lift=rule_min_lift,
            max_antecedents=rule_max_antecedents,
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
    mm_score: float,
    ks_score: float,
    corr_score: float,
    privacy_score: float,
    logical_validity: float,
    utility_report: dict,
    n_real: int,
    n_syn: int,
    t0: float,
) -> dict:
    """
    Compute 4-Pillar Hybrid Integrity scores.
    """
    eps = 1.0

    # 1. Fidelity Pillar (Stats)
    fidelity_score = (
        np.exp(
            (np.log(mm_score + eps) + np.log(ks_score + eps) + np.log(corr_score + eps))
            / 3
        )
        - eps
    )
    fidelity_score = round(float(max(0.0, min(100.0, fidelity_score))), 2)

    # 2. Logic Pillar (Integrity)
    logic_score = round(float(logical_validity), 2)

    # 3. Utility Pillar (Downstream)
    u_scores = []
    if "downstream" in utility_report and "ratio" in utility_report["downstream"]:
        # Perfect ratio is 1.0
        ratio = utility_report["downstream"]["ratio"]
        tstr_val = max(0.0, (1.0 - abs(1.0 - ratio)) * 100)
        u_scores.append(tstr_val)

    if (
        "stylized_facts" in utility_report
        and "_summary" in utility_report["stylized_facts"]
    ):
        u_scores.append(
            utility_report["stylized_facts"]["_summary"].get("mean_score", 0)
        )

    utility_score = round(float(np.mean(u_scores)), 2) if u_scores else None

    # 4. Privacy Pillar
    p_score = round(float(privacy_score), 2)

    # Hybrid Aggregate
    # If utility is missing, we average the 3 available pillars
    pillars = {"fidelity": fidelity_score, "logic": logic_score, "privacy": p_score}
    if utility_score is not None:
        pillars["utility"] = utility_score

    log_sum = sum(np.log(val + eps) for val in pillars.values())
    overall = np.exp(log_sum / len(pillars)) - eps
    overall = round(float(max(0.0, min(100.0, overall))), 2)

    return {
        "hybrid_integrity": overall,
        "pillars": {
            "fidelity": fidelity_score,
            "logic": logic_score,
            "utility": utility_score,
            "privacy": p_score,
        },
        "moment_matching_score": mm_score,
        "ks_score": ks_score,
        "joint_score": corr_score,
        "rows_real": n_real,
        "rows_synthetic": n_syn,
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def fidelity_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    dataset_type: str = "cross_sectional",  # cross_sectional | time_series | panel
    target_col: str | None = None,  # for TSTR downstream score
    include_temporal: bool | None = None,  # auto-detect from dataset_type
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
    include_logical  : run HIF for logical constraint validation
    columns          : restrict to these columns (default: all shared)

    Returns
    -------
    Nested dict with sections: moment_matching, distribution_fit, joint, temporal (optional),
    stylized_facts, downstream (optional), logical (optional), summary.
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
    downstream_report = _downstream_section(
        real, syn, target_col, include_downstream, seed=random_state
    )
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
    )
    report["logical"] = logical_report

    # ── Summary ───────────────────────────────────────────────────────────────
    mm_score = report["moment_matching"]["mean_score"]
    ks_score = report["distribution_fit"]["mean_score"]

    report["summary"] = _summary_section(
        mm_score=mm_score,
        ks_score=ks_score,
        corr_score=corr_score,
        privacy_score=report["privacy_basic"]["privacy_score"],
        logical_validity=logical_validity,
        utility_report=report,  # Contains stylized_facts and downstream
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
    # 4-Pillar Scoreboard
    lines.append("-" * width)
    lines.append(f"{'  PILLAR':<25} | {'SCORE':<10}")
    lines.append("-" * width)
    pillars = s.get("pillars", {})
    lines.append(f"  1. Fidelity (Stats)     | {pillars.get('fidelity', 0):>6.2f}%")
    lines.append(f"  2. Logic (Integrity)    | {pillars.get('logic', 0):>6.2f}%")

    u_score = pillars.get("utility")
    u_str = f"{u_score:>6.2f}%" if u_score is not None else "N/A"
    lines.append(f"  3. Utility (Tasks)      | {u_str}")

    lines.append(f"  4. Privacy (Audit)      | {pillars.get('privacy', 0):>6.2f}%")
    lines.append(f"  HYBRID INTEGRITY (HIF)  | {s.get('hybrid_integrity', 0):>6.2f}%")
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
        lines.append(f"    Target  : {d.get('target_col')}")
        lines.append(
            f"    Metric  : {d.get('metric')} | TSTR {d.get('tstr_score')} | TRR {d.get('trr_score')}"
        )
        lines.append(
            f"    Ratio   : {d.get('ratio')}  — {d.get('interpretation', '')[:50]}"
        )

    lg = report.get("logical", {})
    lines.append("")
    lines.append("  Logical:")

    if "error" in lg:
        lines.append(f"    Status         : ERROR ({lg['error']})")
    elif "info" in lg:
        lines.append(f"    Status         : {lg['info']}")
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
