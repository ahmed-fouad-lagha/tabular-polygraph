"""HIF orchestration entry point — backward-compat hif_score wrapper."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from tabular_polygraph._config import HIFConfig

from .auditor import HIFAuditor

logger = logging.getLogger(__name__)


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
    config = HIFConfig(
        epochs=hif_epochs,
        hubs=hif_hubs,
        depth=hif_depth,
        confidence_percentile=confidence_percentile,
        violation_threshold=violation_threshold,
        rule_min_confidence=rule_min_confidence,
        rule_min_support=rule_min_support,
        rule_max_rules=rule_max_rules,
        rule_min_lift=rule_min_lift,
        rule_max_antecedents=rule_max_antecedents,
        ablation_mode=ablation_mode,
        aggregation=aggregation,
        component_floor=component_floor,
        verbose=verbose,
    )

    auditor = HIFAuditor(config)
    auditor.fit(real, columns=columns)
    return auditor.score(synthetic, progress_callback=progress_callback)
