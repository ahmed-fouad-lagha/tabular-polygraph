"""HIF orchestration entry point."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .binning import (  # noqa: F401
    RULE_QUANTIZATION_BINS,
)
from .binning import (
    apply_binning as _apply_binning,
)
from .binning import (
    canonicalize_code_columns as _canonicalize_code_columns,
)
from .binning import (
    fit_binning as _fit_binning,
)
from .nic import (  # noqa: F401
    NIC_COLLAPSE_PENALTY,
    NIC_COLLAPSE_THRESHOLD,
    NIC_GAMMA_PERCENTILE,
    NIC_Z_PERCENTILE,
    NeighborInvariantContinuity,
)
from .rules import (  # noqa: F401
    MAX_RULE_CANDIDATES,
    mine_implication_rules,
    rule_violation_score,
)
from .sentinel import (  # noqa: F401
    LSE_MIN_SAMPLES_LEAF,
    LogicalSentinelEnsemble,
    ManifoldEncoder,
)

logger = logging.getLogger(__name__)

__all__ = [
    "hif_score",
    "ManifoldEncoder",
    "LogicalSentinelEnsemble",
    "NeighborInvariantContinuity",
    "mine_implication_rules",
    "rule_violation_score",
    "MAX_RULE_CANDIDATES",
    "LSE_MIN_SAMPLES_LEAF",
    "NIC_COLLAPSE_THRESHOLD",
    "NIC_COLLAPSE_PENALTY",
    "NIC_Z_PERCENTILE",
    "NIC_GAMMA_PERCENTILE",
    "RULE_QUANTIZATION_BINS",
]


def hif_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
    hif_epochs: int = 10,
    hif_hubs: int = 5,
    hif_depth: int = 12,
    confidence_percentile: float = 5.0,
    violation_threshold: float = 0.5,
    rule_min_confidence: float = 0.95,
    rule_min_support: float = 0.005,
    rule_max_rules: int = 25,
    rule_min_lift: float = 1.0,
    rule_max_antecedents: int = 2,
    random_state: int = 42,
    verbose: bool = False,
    ablation_mode: str = "full",
    aggregation: str = "geometric",
    component_floor: float = 1e-4,
    progress_callback: Any | None = None,
) -> dict:
    """
    Hybrid Integrity Framework (HIF) Entry Point.
    Orchestrates the Tabular Polygraph via Logical Sentinel Ensemble (LSE)
    and Neighbor-Invariant Continuity (NIC).

    Parameters
    ----------
    hif_epochs : int
        Multiplier for RF tree count per sentinel (NOT training epochs):
        ``n_trees = max(10, hif_epochs * 10)``.
    """
    seed_val = int(random_state) if random_state is not None else 42
    from tabular_polygraph.utils import set_seed

    set_seed(seed_val)

    if columns is None:
        columns = real.columns.intersection(synthetic.columns).tolist()

    valid_cols, skipped_cols = [], []
    for col in columns:
        if pd.api.types.is_numeric_dtype(real[col]):
            skipped_cols.append(col)
        else:
            valid_cols.append(col)

    bin_edges = _fit_binning(real[columns], columns)
    all_f_real = _apply_binning(real[columns], columns, bin_edges)
    all_f_syn = _apply_binning(synthetic[columns], columns, bin_edges)
    real_f = all_f_real
    synthetic_f = all_f_syn

    encoder = ManifoldEncoder()
    encoder.fit(real_f)
    x_real_cat = encoder.transform(real_f)
    x_syn_cat = encoder.transform(synthetic_f)

    if callable(progress_callback):
        progress_callback(1, 3, "Auditing Sentinels...")

    oracle = LogicalSentinelEnsemble(
        top_n_hubs=hif_hubs,
        max_depth=hif_depth,
        random_state=random_state,
        confidence_percentile=confidence_percentile,
    )
    if verbose:
        logger.debug("Auditing Sentinel Logical Consistency...")
    oracle.fit(
        real_f,
        hif_epochs=hif_epochs,
        verbose=verbose,
        x_precomputed=x_real_cat,
        potential_hubs=columns,
    )
    _, cat_penalties, meta = oracle.audit(synthetic_f, x_precomputed=x_syn_cat)

    nic_violation_rate = 0.0
    nic_penalties = np.zeros(len(synthetic))

    nic_targets = [c for c in skipped_cols if c not in oracle.hubs]

    if callable(progress_callback):
        progress_callback(2, 3, "Auditing Continuity (NIC)...")

    if nic_targets:
        if verbose:
            logger.debug("Training Neighbor-Invariant Continuity Auditor...")
        nic_auditor = NeighborInvariantContinuity(random_state=random_state)
        cat_context_real = real_f[oracle.hubs] if oracle.hubs else real_f
        cat_context_syn = synthetic_f[oracle.hubs] if oracle.hubs else synthetic_f

        nic_auditor.fit(
            cat_context_real,
            real[nic_targets],
            x_precomputed=None,
            verbose=verbose,
        )
        _, nic_penalties_raw = nic_auditor.score(
            cat_context_syn, synthetic[nic_targets], x_precomputed=None
        )
        nic_penalties = nic_penalties_raw
        nic_violation_rate = (nic_penalties > 0.5).mean()
        if verbose:
            logger.debug("NIC Auditor training complete.")

    if callable(progress_callback):
        progress_callback(3, 3, "Mining Implication Rules...")

    if verbose:
        logger.debug("Mining and checking Implication Rules...")
    real_f, synthetic_f = _canonicalize_code_columns(real_f, synthetic_f, columns)
    rule_result = rule_violation_score(
        real_f,
        synthetic_f,
        columns=columns,
        min_confidence=rule_min_confidence,
        min_support=rule_min_support,
        max_rules=rule_max_rules,
        min_lift=rule_min_lift,
        max_antecedents=rule_max_antecedents,
        random_state=random_state,
        pre_binned=True,
    )
    rule_penalties = np.zeros(len(synthetic))
    if rule_result.get("num_rows_with_violations", 0) > 0:
        rule_penalties = rule_result.get("row_violation_mask", np.zeros(len(synthetic)))

    if verbose:
        logger.debug(f"Rule mining complete ({rule_result['num_rules_mined']} rules).")

    eps = component_floor

    if ablation_mode == "lse_only":
        active_components = [np.clip(1.0 - cat_penalties, eps, 1.0)]
    elif ablation_mode == "nic_only":
        if nic_targets:
            active_components = [np.clip(1.0 - nic_penalties, eps, 1.0)]
        else:
            active_components = [np.ones(len(synthetic))]
    elif ablation_mode == "rules_only":
        active_components = [np.clip(1.0 - rule_penalties, eps, 1.0)]
    elif ablation_mode == "lse_nic":
        active_components = [np.clip(1.0 - cat_penalties, eps, 1.0)]
        if nic_targets:
            active_components.append(np.clip(1.0 - nic_penalties, eps, 1.0))
    else:
        active_components = [np.clip(1.0 - cat_penalties, eps, 1.0)]
        if nic_targets:
            active_components.append(np.clip(1.0 - nic_penalties, eps, 1.0))
        active_components.append(np.clip(1.0 - rule_penalties, eps, 1.0))

    if aggregation == "arithmetic":
        row_validity = np.asarray(sum(active_components) / len(active_components))
    else:
        log_sum = sum(np.log(c) for c in active_components)
        row_validity = np.asarray(np.exp(log_sum / len(active_components)))

    row_penalties = np.asarray(1.0 - row_validity)
    hif_score_val = row_validity.mean()

    num_violations = (row_penalties > violation_threshold).sum()
    violation_rate = float(num_violations / len(row_penalties))
    cat_violation_rate = (cat_penalties > violation_threshold).mean()

    return {
        "hif_score": round(float(hif_score_val), 4),
        "row_penalties": row_penalties,
        "violation_rate": round(violation_rate, 4),
        "mean_penalty": round(float(row_penalties.mean()), 4),
        "num_violations": int(num_violations),
        "violation_threshold": violation_threshold,
        "lse_violation_rate": round(float(cat_violation_rate), 4),
        "nic_violation_rate": round(float(nic_violation_rate), 4),
        "rule_violation_rate": round(float(rule_result["rule_violation_rate"]), 4),
        "num_rule_violations": int(rule_result.get("num_rows_with_violations", 0)),
        "num_rules_mined": int(rule_result["num_rules_mined"]),
        "total_rule_hits": int(rule_result.get("total_rule_hits", 0)),
        "top_violated_rules": rule_result["top_violated_rules"],
        "violation_examples": rule_result["violation_examples"],
        "columns_used": valid_cols + skipped_cols,
        "traces": meta.get("traces", []),
        "ablation_mode": ablation_mode,
        "aggregation": aggregation,
    }
