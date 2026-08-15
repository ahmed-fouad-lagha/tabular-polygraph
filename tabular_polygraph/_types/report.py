from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class PerColumnScore:
    columns: dict[str, float]
    mean: float


@dataclass
class CoverageScore:
    alpha_precision: float | None = None
    beta_recall: float | None = None
    authenticity: float | None = None


@dataclass
class JointScore:
    correlation_distance_score: float = 0.0
    pairwise_deltas: dict[str, float] = field(default_factory=dict)


@dataclass
class StylizedFactsScore:
    per_column: dict[str, dict[str, Any]] = field(default_factory=dict)
    mean_score: float | None = None
    columns_tested: int = 0
    applicable: bool = True
    note: str | None = None


@dataclass
class DownstreamScore:
    target_col: str = ""
    metric: str = ""
    task: str = ""
    tstr_score: float = 0.0
    trr_score: float = 0.0
    ratio: float = 0.0
    status: str | None = None
    reason: str | None = None


@dataclass
class LogicalScore:
    hif_score_pct: float = 0.0
    hif_violation_rate_pct: float = 0.0
    mean_penalty_pct: float = 0.0
    num_hif_violations: int = 0
    violation_threshold: float | None = None
    nic_violation_rate_pct: float = 0.0
    lse_violation_rate_pct: float = 0.0
    rule_violation_rate_pct: float = 0.0
    num_rule_violations: int = 0
    num_rules_mined: int = 0
    columns_used: list[str] = field(default_factory=list)
    top_violated_rules: list[dict[str, Any]] = field(default_factory=list)
    violation_examples: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class Summary:
    moment_matching_score: float = 0.0
    ks_score: float = 0.0
    tvd_score: float = 0.0
    joint_score: float = 0.0
    alpha_precision: float | None = None
    beta_recall: float | None = None
    authenticity: float | None = None
    logic_score: float | None = None
    stylized_facts_score: float | None = None
    tstr_ratio: float | None = None
    rows_real: int = 0
    rows_synthetic: int = 0
    elapsed_seconds: float = 0.0
    failed_metrics: list[str] = field(default_factory=list)


@dataclass
class FidelityReport:
    dataset_type: str = "cross_sectional"
    columns_evaluated: list[str] = field(default_factory=list)

    moment_matching: PerColumnScore | None = None
    distribution_fit: PerColumnScore | None = None
    categorical_tvd: PerColumnScore | None = None
    joint: JointScore | None = None
    coverage: CoverageScore | None = None
    stylized_facts: StylizedFactsScore | None = None
    downstream: DownstreamScore | None = None
    logical: LogicalScore | None = None
    summary: Summary = field(default_factory=Summary)

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for field_name, field_value in asdict(self).items():
            result[field_name] = _json_safe(field_value)
        return result


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return _json_safe(dict(obj._asdict()))  # type: ignore[attr-defined]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
