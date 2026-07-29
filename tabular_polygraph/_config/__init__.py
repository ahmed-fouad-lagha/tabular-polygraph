from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "HIFConfig",
    "FidelityConfig",
    "GeneratorConfig",
]

# ── Fidelity metric defaults ─────────────────────────────────────────────────

# MomentMatching: weights for (mean, var, skew, kurtosis) — sum to 1.0
# Chosen to prioritize first two moments per statistical convention
DEFAULT_MOMENT_WEIGHTS: tuple[float, float, float, float] = (0.40, 0.35, 0.15, 0.10)
DEFAULT_MOMENT_EPS: float = 1e-8  # Numerical stability for division
DEFAULT_MOMENT_MIN_SAMPLES: int = 10  # Minimum rows for reliable moments

DEFAULT_KS_MIN_SAMPLES: int = 10  # Minimum for KS test asymptotic validity

DEFAULT_TVD_MIN_SAMPLES: int = 1  # TVD needs only 1 sample per category

DEFAULT_JOINT_MIN_COLS: int = 2  # Minimum 2 columns for correlation matrix

# AlphaBeta: 30 steps balances resolution vs runtime (O(n_steps * n_rows))
DEFAULT_ALPHA_BETA_N_STEPS: int = 30
DEFAULT_ALPHA_BETA_MAX_ROWS: int = 10000  # Cap for memory/performance
DEFAULT_ALPHA_BETA_MIN_ROWS: int = 10  # Minimum for 2D density estimation

# StylizedFacts: 50 samples minimum for tail/concentration estimates
DEFAULT_STYLIZED_MIN_SAMPLES: int = 50
DEFAULT_STYLIZED_TAIL_PERCENTILES: tuple[int, int] = (99, 50)  # P99 vs median
DEFAULT_STYLIZED_CONCENTRATION_TOP: float = 0.05  # Top 5% concentration

# Downstream TSTR: 30% test split is standard ML practice
DEFAULT_DOWNSTREAM_TEST_FRAC: float = 0.3
DEFAULT_DOWNSTREAM_N_ESTIMATORS: int = 100  # RF trees, balances accuracy/speed
DEFAULT_DOWNSTREAM_CLASS_THRESHOLD: int = 10  # Max classes for classification

# ── NIC defaults ─────────────────────────────────────────────────────────────

DEFAULT_NIC_LATENT_DIM_CAP: int = 32  # Max SVD components (prevents overfit)
DEFAULT_NIC_Z_PERCENTILE: int = 95  # Residual threshold: 95th percentile
DEFAULT_NIC_GAMMA_PERCENTILE: int = 98  # Gamma scaling: 98th percentile
DEFAULT_NIC_COLLAPSE_THRESHOLD: float = 0.1  # std(pred) < 0.1 → collapsed
DEFAULT_NIC_COLLAPSE_PENALTY: float = 0.6  # Penalty for collapsed regressors
DEFAULT_NIC_MAX_ITER: int = 100  # HGBR max iterations
DEFAULT_NIC_MAX_DEPTH: int = 5  # Tree depth (shallow = more robust)
DEFAULT_NIC_LEARNING_RATE: float = 0.1  # Shrinkage
DEFAULT_NIC_L2_REGULARIZATION: float = 1.0  # L2 penalty

# ── LSE defaults ─────────────────────────────────────────────────────────────

DEFAULT_LSE_N_ESTIMATORS_DISCOVERY: int = 25  # Fast discovery phase
DEFAULT_LSE_MAX_DEPTH_DISCOVERY: int = 8  # Deeper for discovery
DEFAULT_LSE_MAX_FEATURES_DISCOVERY: str = "sqrt"  # Feature subsampling
DEFAULT_LSE_MAX_FEATURES_TRAIN: str = "log2"  # More aggressive for final
DEFAULT_LSE_CONFIDENCE_FLOOR: float = 0.01  # Min prob for OOB calibration
DEFAULT_LSE_MIN_SAMPLES_LEAF: int = 5  # Prevents overfit on small groups
DEFAULT_LSE_CV_SPLITS: int = 5  # Standard 5-fold CV
DEFAULT_LSE_HIGH_CARDINALITY_CAP: int = 50  # Skip hub candidates > 50 levels

# ── Rule-mining defaults ─────────────────────────────────────────────────────

DEFAULT_RULE_QUANTIZATION_BINS: int = 10  # Quantiles for numeric→categorical
DEFAULT_RULE_MAX_CANDIDATES: int = 10000  # Max rule candidates before pruning
DEFAULT_RULE_MAX_EXAMPLES: int = 20  # Max violation examples per rule
DEFAULT_RULE_MIN_CONFIDENCE: float = 0.95  # Association rule confidence
DEFAULT_RULE_MIN_SUPPORT: float = 0.005  # Minimum support (0.5%)
DEFAULT_RULE_MAX_RULES: int = 25  # Max rules to keep
DEFAULT_RULE_MIN_LIFT: float = 1.0  # Lift > 1.0 = positive association
DEFAULT_RULE_MAX_ANTECEDENTS: int = 2  # Max conditions in rule IF-part

# ── HIF runner defaults ──────────────────────────────────────────────────────

DEFAULT_HIF_EPOCHS: int = 10  # LSE trees = max(10, epochs * 10)
DEFAULT_HIF_HUBS: int = 5  # Number of manifold hubs
DEFAULT_HIF_DEPTH: int = 12  # Max tree depth
DEFAULT_HIF_CONFIDENCE_PERCENTILE: float = 5.0  # OOB prob floor percentile
DEFAULT_HIF_VIOLATION_THRESHOLD: float = 0.5  # Penalty > 0.5 = violation
DEFAULT_HIF_COMPONENT_FLOOR: float = 1e-4  # Geometric mean floor
DEFAULT_HIF_ABLATION_MODE: str = (
    "full"  # full | lse_only | nic_only | rules_only | lse_nic
)
DEFAULT_HIF_AGGREGATION: str = "geometric"  # geometric | arithmetic mean

# ── Fidelity runner defaults ─────────────────────────────────────────────────

DEFAULT_FIDELITY_RANDOM_STATE: int = 42  # Reproducibility seed
DEFAULT_FIDELITY_DATASET_TYPE: str = "cross_sectional"
DEFAULT_FIDELITY_PARALLEL: bool = False  # ThreadPoolExecutor for metrics
DEFAULT_FIDELITY_MAX_WORKERS: int = 4  # CPU cores
DEFAULT_FIDELITY_INCLUDE_DOWNSTREAM: bool = True  # TSTR by default
DEFAULT_FIDELITY_VERBOSE: bool = False

# ── Privacy audit defaults ───────────────────────────────────────────────────

DEFAULT_PRIVACY_HOLDOUT_FRAC: float = 0.2  # 20% holdout for MI attacks
DEFAULT_PRIVACY_N_ATTACKS: int = 300  # Shadow model attacks per test
DEFAULT_PRIVACY_N_SAMPLE: int = 200  # Samples per attack
DEFAULT_PRIVACY_SYN_MULTIPLIER: int = 5  # Synthetic samples = 5x attacks
DEFAULT_PRIVACY_BATCH: int = 100  # Batch size for distance computation
DEFAULT_PRIVACY_SEED: int = 42  # Reproducibility
DEFAULT_PRIVACY_QUASI_ID_MAX: int = 8  # Max quasi-identifier columns
DEFAULT_PRIVACY_MIN_DATA: int = 20  # Minimum rows for attack
DEFAULT_PRIVACY_SINGLING_OUT_N_ATTACKS: int = 500  # SO attacks
DEFAULT_PRIVACY_LINKABILITY_BASELINE: float = 0.5  # Expected NNDR by chance


# ── Config dataclasses ───────────────────────────────────────────────────────


@dataclass
class NICConfig:
    latent_dim_cap: int = DEFAULT_NIC_LATENT_DIM_CAP
    z_percentile: int = DEFAULT_NIC_Z_PERCENTILE
    gamma_percentile: int = DEFAULT_NIC_GAMMA_PERCENTILE
    collapse_threshold: float = DEFAULT_NIC_COLLAPSE_THRESHOLD
    collapse_penalty: float = DEFAULT_NIC_COLLAPSE_PENALTY
    max_iter: int = DEFAULT_NIC_MAX_ITER
    max_depth: int = DEFAULT_NIC_MAX_DEPTH
    learning_rate: float = DEFAULT_NIC_LEARNING_RATE
    l2_regularization: float = DEFAULT_NIC_L2_REGULARIZATION


@dataclass
class LSEConfig:
    n_estimators_discovery: int = DEFAULT_LSE_N_ESTIMATORS_DISCOVERY
    max_depth_discovery: int = DEFAULT_LSE_MAX_DEPTH_DISCOVERY
    max_features_discovery: str = DEFAULT_LSE_MAX_FEATURES_DISCOVERY
    max_features_train: str = DEFAULT_LSE_MAX_FEATURES_TRAIN
    confidence_floor: float = DEFAULT_LSE_CONFIDENCE_FLOOR
    min_samples_leaf: int = DEFAULT_LSE_MIN_SAMPLES_LEAF
    cv_splits: int = DEFAULT_LSE_CV_SPLITS
    high_cardinality_cap: int = DEFAULT_LSE_HIGH_CARDINALITY_CAP


@dataclass
class RulesConfig:
    quantization_bins: int = DEFAULT_RULE_QUANTIZATION_BINS
    max_candidates: int = DEFAULT_RULE_MAX_CANDIDATES
    max_examples: int = DEFAULT_RULE_MAX_EXAMPLES
    min_confidence: float = DEFAULT_RULE_MIN_CONFIDENCE
    min_support: float = DEFAULT_RULE_MIN_SUPPORT
    max_rules: int = DEFAULT_RULE_MAX_RULES
    min_lift: float = DEFAULT_RULE_MIN_LIFT
    max_antecedents: int = DEFAULT_RULE_MAX_ANTECEDENTS


@dataclass
class HIFConfig:
    epochs: int = DEFAULT_HIF_EPOCHS
    hubs: int = DEFAULT_HIF_HUBS
    depth: int = DEFAULT_HIF_DEPTH
    confidence_percentile: float = DEFAULT_HIF_CONFIDENCE_PERCENTILE
    violation_threshold: float = DEFAULT_HIF_VIOLATION_THRESHOLD
    component_floor: float = DEFAULT_HIF_COMPONENT_FLOOR
    ablation_mode: str = DEFAULT_HIF_ABLATION_MODE
    aggregation: str = DEFAULT_HIF_AGGREGATION
    verbose: bool = False
    progress_callback: Any | None = None
    random_state: int = DEFAULT_FIDELITY_RANDOM_STATE

    nic: NICConfig = field(default_factory=NICConfig)
    lse: LSEConfig = field(default_factory=LSEConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)


@dataclass
class FidelityConfig:
    columns: list[str] | None = None
    target_col: str | None = None
    include_downstream: bool = DEFAULT_FIDELITY_INCLUDE_DOWNSTREAM
    random_state: int = DEFAULT_FIDELITY_RANDOM_STATE
    dataset_type: str = DEFAULT_FIDELITY_DATASET_TYPE
    verbose: bool = DEFAULT_FIDELITY_VERBOSE
    progress_callback: Any | None = None

    parallel: bool = DEFAULT_FIDELITY_PARALLEL
    max_workers: int = DEFAULT_FIDELITY_MAX_WORKERS

    hif: HIFConfig = field(default_factory=HIFConfig)


@dataclass
class GeneratorConfig:
    random_state: int = DEFAULT_FIDELITY_RANDOM_STATE
