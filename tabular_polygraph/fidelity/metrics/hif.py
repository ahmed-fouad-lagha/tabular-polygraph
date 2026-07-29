from __future__ import annotations

import pandas as pd

from tabular_polygraph._config import HIFConfig, RulesConfig
from tabular_polygraph._types import Metric

from . import register


@register
class HIFMetric(Metric):
    name = "hif"

    def __init__(self, config: HIFConfig | None = None):
        self._config = config or HIFConfig()

    def required_column_types(self) -> set[str]:
        return {"all"}

    def fit(self, real: pd.DataFrame, columns: list[str]) -> None:
        pass

    def compute(
        self, real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]
    ) -> dict:
        from tabular_polygraph.fidelity.hif import hif_score as _hif_score

        rules = self._config.rules if hasattr(self._config, "rules") else RulesConfig()
        result = _hif_score(
            real,
            synthetic,
            columns=columns,
            hif_epochs=self._config.epochs,
            hif_hubs=self._config.hubs,
            hif_depth=self._config.depth,
            rule_min_confidence=rules.min_confidence,
            rule_min_support=rules.min_support,
            rule_max_rules=rules.max_rules,
            rule_min_lift=rules.min_lift,
            rule_max_antecedents=rules.max_antecedents,
            random_state=self._config.random_state,
            verbose=False,
            progress_callback=self._config.progress_callback,
        )

        logical_validity = round(float(result["hif_score"] * 100.0), 2)

        return {
            "hif_score_pct": logical_validity,
            "hif_violation_rate_pct": round(float(result["violation_rate"] * 100.0), 2),
            "mean_penalty_pct": round(float(result["mean_penalty"] * 100.0), 2),
            "num_hif_violations": int(result["num_violations"]),
            "violation_threshold": result.get("violation_threshold"),
            "nic_violation_rate_pct": round(
                float(result.get("nic_violation_rate", 0) * 100.0), 2
            ),
            "lse_violation_rate_pct": round(
                float(result.get("lse_violation_rate", 0) * 100.0), 2
            ),
            "rule_violation_rate_pct": round(
                float(result.get("rule_violation_rate", 0) * 100.0), 2
            ),
            "num_rule_violations": int(result.get("num_rule_violations", 0)),
            "num_rules_mined": int(result["num_rules_mined"]),
            "columns_used": result["columns_used"],
            "top_violated_rules": result.get("top_violated_rules", []),
            "violation_examples": result.get("violation_examples", []),
            "logical_validity": logical_validity,
        }
