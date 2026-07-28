"""HIFAuditor — stateful wrapper for the Hybrid Integrity Framework."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from tabular_polygraph._config import HIFConfig

from .binning import apply_binning as _apply_binning
from .binning import canonicalize_code_columns as _canonicalize_code_columns
from .binning import fit_binning as _fit_binning
from .nic import NeighborInvariantContinuity
from .rules import rule_violation_score
from .sentinel import LogicalSentinelEnsemble, ManifoldEncoder

logger = logging.getLogger(__name__)


class HIFAuditor:
    """Stateful HIF auditor with separate fit/score lifecycle.

    Fits a manifold encoder, LSE oracle, NIC auditor, and implication rules
    on *real* data, then scores *synthetic* data against the learned manifold.
    """

    def __init__(self, config: HIFConfig | None = None) -> None:
        self.config = config or HIFConfig()
        self.encoder: ManifoldEncoder | None = None
        self.oracle: LogicalSentinelEnsemble | None = None
        self.nic_auditor: NeighborInvariantContinuity | None = None
        self._bin_edges: dict[str, np.ndarray | None] = {}
        self._columns: list[str] = []
        self._valid_cols: list[str] = []
        self._skipped_cols: list[str] = []
        self._real_f: pd.DataFrame | None = None
        self._is_fitted: bool = False

    def fit(
        self,
        real: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> None:
        cfg = self.config
        if columns is None:
            columns = real.columns.tolist()

        self._columns = columns
        self._valid_cols, self._skipped_cols = [], []
        for col in columns:
            if pd.api.types.is_numeric_dtype(real[col]):
                self._skipped_cols.append(col)
            else:
                self._valid_cols.append(col)

        self._bin_edges = _fit_binning(real[columns], columns)

        real_f = _apply_binning(real[columns], columns, self._bin_edges)
        self._real_f = real_f
        self.encoder = ManifoldEncoder()
        self.encoder.fit(real_f)
        x_real_cat = self.encoder.transform(real_f)

        self.oracle = LogicalSentinelEnsemble(
            top_n_hubs=cfg.hubs,
            max_depth=cfg.depth,
            random_state=cfg.random_state,
            confidence_percentile=cfg.confidence_percentile,
        )
        self.oracle.fit(
            real_f,
            hif_epochs=cfg.epochs,
            verbose=cfg.verbose,
            x_precomputed=x_real_cat,
            potential_hubs=columns,
        )

        hubs = self.oracle.hubs if self.oracle is not None else []
        nic_targets = [c for c in self._skipped_cols if c not in hubs]
        if nic_targets:
            cat_context_real = real_f[hubs] if hubs else real_f
            self.nic_auditor = NeighborInvariantContinuity(
                config=cfg,
                random_state=cfg.random_state,
            )
            self.nic_auditor.fit(
                cat_context_real,
                real[nic_targets],
                x_precomputed=None,
                verbose=cfg.verbose,
            )

        self._is_fitted = True

    def score(
        self,
        synthetic: pd.DataFrame,
        progress_callback: Any | None = None,
    ) -> dict:
        if not self._is_fitted:
            raise RuntimeError("HIFAuditor must be fitted before score().")

        cfg = self.config
        columns = self._columns
        synthetic_f = _apply_binning(synthetic[columns], columns, self._bin_edges)

        if callable(progress_callback):
            progress_callback(1, 3, "Auditing Sentinels...")

        if self.encoder is None or self.oracle is None:
            raise RuntimeError(
                "HIFAuditor is not fully fitted — encoder or oracle is None"
            )
        x_syn_cat = self.encoder.transform(synthetic_f)
        _, cat_penalties, meta = self.oracle.audit(synthetic_f, x_precomputed=x_syn_cat)

        nic_violation_rate = 0.0
        nic_penalties = np.zeros(len(synthetic))

        hubs = self.oracle.hubs if self.oracle is not None else []
        nic_targets = [c for c in self._skipped_cols if c not in hubs]
        if callable(progress_callback):
            progress_callback(2, 3, "Auditing Continuity (NIC)...")

        if nic_targets and self.nic_auditor is not None and self.oracle is not None:
            cat_context_syn = synthetic_f[hubs] if hubs else synthetic_f
            _, nic_penalties_raw = self.nic_auditor.score(
                cat_context_syn, synthetic[nic_targets], x_precomputed=None
            )
            nic_penalties = nic_penalties_raw
            nic_violation_rate = (nic_penalties > 0.5).mean()

        if callable(progress_callback):
            progress_callback(3, 3, "Mining Implication Rules...")

        real_f, synthetic_f_norm = _canonicalize_code_columns(
            self._real_f, synthetic_f, columns
        )
        rule_result = rule_violation_score(
            real_f,
            synthetic_f_norm,
            columns=columns,
            min_confidence=cfg.rule_min_confidence,
            min_support=cfg.rule_min_support,
            max_rules=cfg.rule_max_rules,
            min_lift=cfg.rule_min_lift,
            max_antecedents=cfg.rule_max_antecedents,
            random_state=cfg.random_state,
            pre_binned=True,
        )
        rule_penalties = np.zeros(len(synthetic))
        if rule_result.get("num_rows_with_violations", 0) > 0:
            rule_penalties = rule_result.get(
                "row_violation_mask", np.zeros(len(synthetic))
            )

        eps = cfg.component_floor

        if cfg.ablation_mode == "lse_only":
            active_components = [np.clip(1.0 - cat_penalties, eps, 1.0)]
        elif cfg.ablation_mode == "nic_only":
            if nic_targets:
                active_components = [np.clip(1.0 - nic_penalties, eps, 1.0)]
            else:
                active_components = [np.ones(len(synthetic))]
        elif cfg.ablation_mode == "rules_only":
            active_components = [np.clip(1.0 - rule_penalties, eps, 1.0)]
        elif cfg.ablation_mode == "lse_nic":
            active_components = [np.clip(1.0 - cat_penalties, eps, 1.0)]
            if nic_targets:
                active_components.append(np.clip(1.0 - nic_penalties, eps, 1.0))
        else:
            active_components = [np.clip(1.0 - cat_penalties, eps, 1.0)]
            if nic_targets:
                active_components.append(np.clip(1.0 - nic_penalties, eps, 1.0))
            active_components.append(np.clip(1.0 - rule_penalties, eps, 1.0))

        if cfg.aggregation == "arithmetic":
            row_validity = np.asarray(sum(active_components) / len(active_components))
        else:
            log_sum = sum(np.log(c) for c in active_components)
            row_validity = np.asarray(np.exp(log_sum / len(active_components)))

        row_penalties = np.asarray(1.0 - row_validity)
        hif_score_val = row_validity.mean()

        num_violations = (row_penalties > cfg.violation_threshold).sum()
        violation_rate = float(num_violations / len(row_penalties))
        cat_violation_rate = (cat_penalties > cfg.violation_threshold).mean()

        return {
            "hif_score": round(float(hif_score_val), 4),
            "row_penalties": row_penalties,
            "violation_rate": round(violation_rate, 4),
            "mean_penalty": round(float(row_penalties.mean()), 4),
            "num_violations": int(num_violations),
            "violation_threshold": cfg.violation_threshold,
            "lse_violation_rate": round(float(cat_violation_rate), 4),
            "nic_violation_rate": round(float(nic_violation_rate), 4),
            "rule_violation_rate": round(float(rule_result["rule_violation_rate"]), 4),
            "num_rule_violations": int(rule_result.get("num_rows_with_violations", 0)),
            "num_rules_mined": int(rule_result["num_rules_mined"]),
            "total_rule_hits": int(rule_result.get("total_rule_hits", 0)),
            "top_violated_rules": rule_result["top_violated_rules"],
            "violation_examples": rule_result["violation_examples"],
            "columns_used": self._valid_cols + self._skipped_cols,
            "traces": meta.get("traces", []),
            "ablation_mode": cfg.ablation_mode,
            "aggregation": cfg.aggregation,
        }

    def run(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict:
        """Convenience: fit on real data, then score synthetic."""
        self.fit(real, columns=columns)
        return self.score(synthetic)
