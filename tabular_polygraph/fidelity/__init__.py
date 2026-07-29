from __future__ import annotations

from typing import Any

import pandas as pd

from tabular_polygraph._config import FidelityConfig

from .formatters import ConsoleFormatter
from .pipeline import FidelityPipeline


def fidelity_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    dataset_type: str = "cross_sectional",
    target_col: str | None = None,
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
    include_privacy: bool = False,
    privacy_n_attacks: int = 300,
) -> dict:
    from tabular_polygraph._config import HIFConfig

    config = FidelityConfig(
        columns=columns,
        target_col=target_col,
        include_downstream=include_downstream,
        random_state=random_state,
        dataset_type=dataset_type,
        verbose=verbose,
        progress_callback=progress_callback,
        hif=HIFConfig(
            epochs=hif_epochs,
            hubs=hif_hubs,
            depth=hif_depth,
            rule_min_confidence=rule_min_confidence,
            rule_min_support=rule_min_support,
            rule_max_rules=rule_max_rules,
            rule_min_lift=rule_min_lift,
            rule_max_antecedents=rule_max_antecedents,
            random_state=random_state,
        ),
    )

    pipeline = FidelityPipeline(config)
    report = pipeline.run(real, synthetic)
    report_dict = report.to_dict()

    if include_privacy:
        from tabular_polygraph.privacy import privacy_audit

        try:
            privacy_report = privacy_audit(
                real,
                synthetic,
                real_holdout=None,
                holdout_frac=0.2,
                quasi_id_cols=None,
                numeric_cols=None,
                n_attacks=privacy_n_attacks,
                seed=random_state,
            )
            report_dict["privacy"] = privacy_report
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Privacy audit failed: {e}")

    return report_dict


def format_report(report: dict, width: int = 60) -> str:
    return ConsoleFormatter(width=width).format(report)


__all__ = [
    "fidelity_report",
    "format_report",
]
