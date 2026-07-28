"""Hybrid Integrity Framework public API."""

from .binning import (
    RULE_QUANTIZATION_BINS,
    apply_binning,
    canonicalize_code_columns,
    fit_binning,
)
from .nic import (
    NIC_COLLAPSE_PENALTY,
    NIC_COLLAPSE_THRESHOLD,
    NIC_GAMMA_PERCENTILE,
    NIC_Z_PERCENTILE,
    NeighborInvariantContinuity,
)
from .orchestrator import hif_score
from .rules import (
    MAX_RULE_CANDIDATES,
    mine_implication_rules,
    rule_violation_score,
)
from .sentinel import (
    LSE_MIN_SAMPLES_LEAF,
    LogicalSentinelEnsemble,
    ManifoldEncoder,
)

__all__ = [
    "hif_score",
    "ManifoldEncoder",
    "LogicalSentinelEnsemble",
    "NeighborInvariantContinuity",
    "mine_implication_rules",
    "rule_violation_score",
    "apply_binning",
    "canonicalize_code_columns",
    "fit_binning",
    "MAX_RULE_CANDIDATES",
    "LSE_MIN_SAMPLES_LEAF",
    "NIC_COLLAPSE_THRESHOLD",
    "NIC_COLLAPSE_PENALTY",
    "NIC_Z_PERCENTILE",
    "NIC_GAMMA_PERCENTILE",
    "RULE_QUANTIZATION_BINS",
]
