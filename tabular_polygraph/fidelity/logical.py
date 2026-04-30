"""
HIF: Hybrid Integrity Framework (The Tabular Polygraph).

A neurosymbolic logical constraint validator for synthetic tabular data.
Trains a Logical Sentinel Ensemble (LSE) and Neighbor-Invariant Continuity (NIC)
auditors on ground-truth manifolds to detect semantic hallucinations via
multiplicative manifold integrity penalties.
"""

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import check_random_state

# --- HIF AUDIT CONSTANTS ---
MAX_RULE_CANDIDATES = 10000
LSE_MIN_SAMPLES_LEAF = 5
LSE_MIN_SUPPORT = 0.005
NIC_COLLAPSE_THRESHOLD = 0.5
NIC_COLLAPSE_PENALTY = 0.6
NIC_Z_PERCENTILE = 95
NIC_GAMMA_PERCENTILE = 98
RULE_QUANTIZATION_BINS = 10
# ---------------------------

_ANTE_JOIN = " & "


def _adaptive_binning(
    df: pd.DataFrame, columns: list[str], n_bins: int = RULE_QUANTIZATION_BINS
) -> pd.DataFrame:
    """Discretize continuous numeric columns into quantile-based bins for logical analysis."""
    df_binned = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 1:
                df_binned[col] = "bin_0"
                continue
            try:
                # Robust binning: Try qcut, fallback to cut, fallback to unique values as strings
                if df[col].nunique() <= n_bins:
                    df_binned[col] = df[col].astype(str)
                    continue
                bins = pd.qcut(df[col], q=n_bins, labels=False, duplicates="drop")
                # SAFE CONVERSION: Ensure we don't int(None) or int(NaN)
                df_binned[col] = bins.apply(
                    lambda x: f"bin_{int(x)}" if pd.notna(x) else x
                )
            except Exception:
                try:
                    df_binned[col] = pd.cut(df[col], bins=n_bins, labels=False).apply(
                        lambda x: f"bin_{int(x)}" if pd.notna(x) else x
                    )
                except Exception:
                    df_binned[col] = df[col].astype(str)
    return df_binned


def _canonicalize_code_columns(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize digit-like categorical codes so numeric CSV reads keep leading zeros."""
    real_norm = real.copy()
    synthetic_norm = synthetic.copy()

    for column in columns:
        real_values = real_norm[column].dropna().astype(str)
        if real_values.empty:
            continue

        if real_values.str.fullmatch(r"\d+").all():
            width = int(real_values.str.len().max())

            def _pad(value, pad_width=width):
                if pd.isna(value):
                    return value
                text = str(value)
                return text.zfill(pad_width) if text.isdigit() else text

            real_norm[column] = real_norm[column].map(_pad)
            synthetic_norm[column] = synthetic_norm[column].map(_pad)
        else:
            real_norm[column] = real_norm[column].astype(str)
            synthetic_norm[column] = synthetic_norm[column].astype(str)

    return real_norm, synthetic_norm


class ManifoldEncoder:
    """Stateful Categorical-to-Ordinal projection with feature mapping."""

    def __init__(self) -> None:
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names: List[str] = []
        self.feature_map: Dict[str, List[str]] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """Fit encoder on reference manifold and build feature map."""
        if df.empty:
            return
        self.encoder.fit(df)
        self.feature_names = list(self.encoder.get_feature_names_out())

        # Build O(1) lookup map for Sentinel predictors
        self.feature_map = {}
        for original_col in df.columns:
            self.feature_map[original_col] = [
                name
                for name in self.feature_names
                if name.startswith(f"{original_col}_")
            ]
        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project data into the reference subspace."""
        if not self.is_fitted:
            return pd.DataFrame(index=df.index)
        encoded = self.encoder.transform(df)
        return pd.DataFrame(encoded, columns=self.feature_names, index=df.index)


class LogicalSentinelEnsemble:
    """
    Neuro-Symbolic Integrity Oracle (LSE).
    Learns 'Manifold Laws' using Random Forest Sentinels on high-dependency hubs.
    """

    def __init__(
        self, top_n_hubs: int = 5, max_depth: int = 12, random_state: int = 42
    ) -> None:
        self.top_n_hubs = top_n_hubs
        self.max_depth = max_depth
        self.random_state = random_state
        self.sentinels: Dict[str, RandomForestClassifier] = {}
        self.hubs: List[str] = []
        self.confidence_floors: Dict[str, float] = {}
        self.encoder = ManifoldEncoder()
        self.is_trained: bool = False

    def _discover_hubs(
        self,
        df: pd.DataFrame,
        x_encoded: pd.DataFrame,
        potential_hubs: List[str] | None = None,
    ) -> List[str]:
        """Discover 'Manifold Hubs' using predictive synergy (captures higher-order interactions)."""
        cols = potential_hubs if potential_hubs is not None else df.columns
        scores = {}

        for hub_col in cols:
            # Use all available columns as context, not just potential hubs
            other_cols = [c for c in df.columns if c != hub_col]
            hub_features = []
            for col in other_cols:
                hub_features.extend(self.encoder.feature_map.get(col, []))

            if not hub_features:
                continue

            X = x_encoded[hub_features]
            y = df[hub_col].astype(str)

            if len(y.unique()) < 2:
                continue

            # Use a fast RF to measure how "constrained" this feature is by the rest of the manifold.
            clf = RandomForestClassifier(
                n_estimators=25,
                max_depth=8,
                min_samples_leaf=LSE_MIN_SAMPLES_LEAF,
                random_state=self.random_state,
                max_features="sqrt",
            )
            clf.fit(X, y)
            scores[hub_col] = float(clf.score(X, y))

        sorted_hubs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [h[0] for h in sorted_hubs[: self.top_n_hubs]]

    def fit(
        self,
        df: pd.DataFrame,
        hif_epochs: int = 10,
        verbose: bool = True,
        x_precomputed: pd.DataFrame | None = None,
        potential_hubs: List[str] | None = None,
    ):
        """Train Sentinels using stateful manifold projection."""
        if len(df) < 50:
            return

        if x_precomputed is not None:
            x_encoded = x_precomputed
            # feature_map must be built manually if precomputed
            self.encoder.feature_names = x_encoded.columns.tolist()
            self.encoder.is_fitted = True
            for original_col in df.columns:
                self.encoder.feature_map[original_col] = [
                    name
                    for name in self.encoder.feature_names
                    if name.startswith(f"{original_col}_")
                ]
        else:
            self.encoder.fit(df)
            x_encoded = self.encoder.transform(df)

        if verbose:
            print(
                "  [HIF Hubs] Discovering Synergistic Manifold Hubs...",
                end="",
                flush=True,
            )
        self.hubs = self._discover_hubs(df, x_encoded, potential_hubs=potential_hubs)

        if verbose:
            print(f" Done. Selected: {self.hubs}")

        for hub_col in self.hubs:
            # SUBSPACE: Use feature_map instead of startswith scan
            other_cols = [c for c in df.columns if c != hub_col]
            hub_features = []
            for col in other_cols:
                hub_features.extend(self.encoder.feature_map.get(col, []))

            X = x_encoded[hub_features]
            y = df[hub_col]

            if verbose:
                print(
                    f"  [HIF Sentinels] Training Sentinel for Hub '{hub_col}' ({X.shape[1]} features)...",
                    end="",
                    flush=True,
                )

            n_trees = max(10, hif_epochs * 10)
            clf = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=self.max_depth,
                min_samples_leaf=LSE_MIN_SAMPLES_LEAF,
                random_state=self.random_state,
                max_features="log2",
                oob_score=True,  # Enable OOB for honest confidence calibration
            )
            clf.fit(X, y)
            self.sentinels[hub_col] = clf
            if verbose:
                print("Done.")

            # FIX: Use Out-Of-Bag (OOB) predictions for confidence floor calibration.
            # In-sample predictions overfit on large datasets — the RF predicts its own
            # training data with near-perfect confidence, pushing the percentile floor
            # to near-zero values that make the sentinel blind during audit.
            # OOB predictions are honest cross-validated estimates that reflect the
            # true generalization confidence of each sentinel.
            if hasattr(clf, "oob_decision_function_"):
                oob_probs = clf.oob_decision_function_
            else:
                # Fallback: use in-sample if OOB not available (e.g., very small data)
                oob_probs = clf.predict_proba(X)

            classes = clf.classes_
            y_str = y.astype(str).values
            probs_true = np.zeros(len(y))
            for idx, cls in enumerate(classes):
                mask = y_str == str(cls)
                if mask.any():
                    probs_true[mask] = oob_probs[mask, idx]

            # Use 5th percentile of OOB confidence as the floor.
            # With OOB, this reflects realistic generalization uncertainty
            # rather than memorization artifacts.
            self.confidence_floors[hub_col] = max(
                float(np.percentile(probs_true, 5.0)), 0.01
            )

        self.is_trained = True

    def audit(
        self, df: pd.DataFrame, x_precomputed: pd.DataFrame | None = None
    ) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        """Audit synthetic rows for 'Logical Ruptures' using reference manifold."""
        if not self.is_trained:
            return 1.0, np.zeros(len(df)), {}

        row_penalties = np.zeros(len(df))
        traces = []

        if x_precomputed is not None:
            # Ensure we only use features corresponding to our hubs/context
            x_encoded = x_precomputed
        else:
            x_encoded = self.encoder.transform(df)

        for hub_col in self.hubs:
            clf = self.sentinels[hub_col]
            # Ensure feature alignment with training state
            X = x_encoded.reindex(columns=clf.feature_names_in_, fill_value=0)

            probs = clf.predict_proba(X)
            classes = clf.classes_
            observed_values = df[hub_col].astype(str).values

            probs_observed = np.zeros(len(df))
            for class_idx, class_val in enumerate(classes):
                mask = observed_values == str(class_val)
                if mask.any():
                    probs_observed[mask] = probs[mask, class_idx]

            floor = self.confidence_floors[hub_col]

            # CALIBRATION: Nonlinear error response scaled by floor
            # If probs_observed < floor, penalty starts.
            # Full penalty if probs_observed is 10x smaller than floor or 0.
            if floor > 1e-6:
                penalty = np.clip((floor - probs_observed) / floor, 0, 1)
            else:
                penalty = np.zeros(len(df))

            # ATOMIC AGGREGATION: Multiplicative manifold penalty
            # This ensures that ruptures in multiple hubs compound the penalty
            row_penalties = 1.0 - (1.0 - row_penalties) * (1.0 - penalty)

            ruptures = penalty > 0.5
            if ruptures.any():
                traces.append(
                    {
                        "column": hub_col,
                        "violations": int(ruptures.sum()),
                        "mean_prob": float(probs_observed[ruptures].mean()),
                    }
                )

        hif_score_val = 1.0 - row_penalties.mean()
        avg_floor = (
            float(np.mean(list(self.confidence_floors.values())))
            if self.confidence_floors
            else 1.0
        )
        return (
            float(hif_score_val),
            row_penalties,
            {
                "traces": traces,
                "confidence_floors": self.confidence_floors,
                "avg_floor": avg_floor,
            },
        )


class NeighborInvariantContinuity:
    """
    Neighbor-Invariant Continuity (NIC).
    Audits continuous features against categorical manifold using non-linear reconstruction.
    """

    def __init__(self, random_state: int = 42):
        self.regressors: Dict[str, HistGradientBoostingRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.z_thresholds: Dict[str, float] = {}
        self.gamma_scalings: Dict[str, float] = {}
        self.marginal_references: Dict[str, np.ndarray] = {}
        self.training_prediction_vars: Dict[str, float] = {}
        self.pca: PCA | None = None
        self.latent_scaler = StandardScaler(with_mean=False)
        self.encoder = ManifoldEncoder()
        self.random_state = random_state

    def fit(
        self,
        categorical_df: pd.DataFrame,
        continuous_df: pd.DataFrame,
        x_precomputed: pd.DataFrame | None = None,
        verbose: bool = True,
    ):
        """Fit non-linear regressors on the training manifold."""
        valid_cols = [
            c for c in continuous_df.columns if continuous_df[c].nunique() > 1
        ]
        if not valid_cols:
            return

        active_df = continuous_df[valid_cols]

        if x_precomputed is not None:
            x_encoded = x_precomputed
        else:
            self.encoder.fit(categorical_df)
            x_encoded = self.encoder.transform(categorical_df)

        n_feat = x_encoded.shape[1]
        n_samples = x_encoded.shape[0]
        if n_feat < 1 or n_samples < 2:
            return

        # Spectral projection to reduce categorical manifold sparsity
        n_comp = min(n_samples, n_feat, 100)
        if verbose:
            print(
                f"  [HIF NIC] Spectral Embedding ({n_feat} -> {n_comp} target)...",
                end="",
                flush=True,
            )

        self.pca = PCA(
            n_components=n_comp,
            svd_solver="randomized",
            random_state=self.random_state,
        )
        # HARDENING: Scale categorical manifold to balance rare levels before PCA
        x_scaled = self.latent_scaler.fit_transform(x_encoded)
        latent = self.pca.fit_transform(x_scaled)
        if verbose:
            print(f"Done ({self.pca.n_components_} components).")

        self.regressors = {}
        for col in active_df.columns:
            if verbose:
                print(f"  [HIF NIC] Regressing variable '{col}'...", end="", flush=True)

            # SAFE DROP: Filter both X (latent) and y to remove NaNs in this specific target
            y_raw = active_df[col].values
            valid_mask = ~np.isnan(y_raw)
            if not valid_mask.any():
                if verbose:
                    print("Skipped (all NaN).")
                continue

            y_valid = y_raw[valid_mask]
            # Store sorted reference for Marginal Alignment
            self.marginal_references[col] = np.sort(y_valid)
            latent_valid = latent[valid_mask]

            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y_valid.reshape(-1, 1)).flatten()

            # Upgrade to non-linear booster for complex manifold laws
            reg = HistGradientBoostingRegressor(
                max_iter=100,
                max_depth=5,
                random_state=self.random_state,
                l2_regularization=1.0,
            )
            reg.fit(latent_valid, y_scaled)
            if verbose:
                print("Done.")

            y_pred = reg.predict(latent_valid)
            self.training_prediction_vars[col] = float(np.var(y_pred))
            residuals = np.abs(y_scaled - y_pred)

            self.regressors[col] = reg
            self.scalers[col] = scaler

            # HARDENING: Robust Hybrid MAD-Z Thresholding
            mad = float(median_abs_deviation(residuals))
            med = float(np.median(residuals))
            p95 = float(np.percentile(residuals, 95))

            # Use 95th percentile but ensure it's at least med + 2*MAD for robustness
            self.z_thresholds[col] = max(p95, med + 2.0 * mad)
            # Dynamic gamma factor based on natural prediction noise
            # Balances sensitivity to corruption while suppressing false positives on base synthetic data
            self.gamma_scalings[col] = max(self.z_thresholds[col], 2.0 * mad, 0.5)

    def score(
        self,
        categorical_df: pd.DataFrame,
        continuous_df: pd.DataFrame,
        x_precomputed: pd.DataFrame | None = None,
    ) -> Tuple[float, np.ndarray]:
        """Score continuous features for manifold continuity violations."""
        if self.pca is None or not self.regressors:
            return 1.0, np.zeros(len(continuous_df))

        if x_precomputed is not None:
            x_aligned = x_precomputed
        else:
            x_aligned = self.encoder.transform(categorical_df)

        # HARDENING: Apply latent scaling and projection
        x_scaled = self.latent_scaler.transform(x_aligned)
        latent = self.pca.transform(x_scaled)
        row_penalties = np.zeros(len(continuous_df))

        for col in continuous_df.columns:
            if col not in self.regressors:
                continue

            y = continuous_df[col].values
            # Handle NaNs during scoring: validity = 1 (penalty=0) for NaN values?
            # Or should we flag them as violations? Paper says HIF is non-compensatory.
            # We'll treat NaN as "neutral" (penalty 0) to avoid double-counting if Rules already catch them.
            valid_mask = ~np.isnan(y)
            if not valid_mask.any():
                continue

            y_valid = y[valid_mask]
            latent_valid = latent[valid_mask]

            # HARDENING: Marginal Alignment (Quantile Mapping)
            # This prevents the Marginal Paradox where corruption with real samples
            # appears to 'improve' integrity by fixing the generator's marginal errors.
            # We map synthetic marginals to the real marginals used during training.
            try:
                # Use a robust percentile-based mapping
                y_sorted_real = self.marginal_references[col]
                ranks = np.argsort(np.argsort(y_valid))
                # Create a target distribution matching the size of y_valid
                indices = np.linspace(0, len(y_sorted_real) - 1, len(y_valid)).astype(
                    int
                )
                y_aligned = y_sorted_real[indices][ranks]
            except (KeyError, AttributeError, ValueError):
                y_aligned = y_valid

            y_scaled = self.scalers[col].transform(y_aligned.reshape(-1, 1)).flatten()
            y_pred = self.regressors[col].predict(latent_valid)
            residuals = np.abs(y_scaled - y_pred)

            threshold = self.z_thresholds[col]
            gamma = self.gamma_scalings[col]
            col_penalty = np.zeros(len(y))
            if threshold > 0:
                # Nonlinear penalty scaling (Hardened Response)
                col_penalty[valid_mask] = np.clip((residuals - threshold) / gamma, 0, 1)

            row_penalties = np.maximum(row_penalties, col_penalty)

        return float(1.0 - row_penalties.mean()), row_penalties


def hif_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
    hif_epochs: int = 10,
    hif_hubs: int = 5,
    hif_depth: int = 12,
    rule_min_confidence: float = 0.95,
    rule_min_support: float = 0.005,
    rule_max_rules: int = 25,
    rule_min_lift: float = 1.0,
    rule_max_antecedents: int = 2,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Hybrid Integrity Framework (HIF) Entry Point.
    Orchestrates the Tabular Polygraph via Logical Sentinel Ensemble (LSE)
    and Neighbor-Invariant Continuity (NIC).
    """
    seed_val = int(random_state) if random_state is not None else 42
    np.random.seed(seed_val)
    random.seed(seed_val)

    if columns is None:
        columns = real.columns.intersection(synthetic.columns).tolist()

    valid_cols, skipped_cols = [], []
    for col in columns:
        if pd.api.types.is_numeric_dtype(real[col]):
            skipped_cols.append(col)
        else:
            valid_cols.append(col)

    if not valid_cols:
        return {
            "hif_score": 1.0,
            "row_penalties": np.zeros(len(synthetic)),
            "violation_rate": 0.0,
            "mean_penalty": 0.0,
            "num_violations": 0,
            "columns_used": [],
        }

    # Pre-processing: Exhaustive Adaptive Binning (Categorical + Numeric for context)
    all_f_real, all_f_syn = _canonicalize_code_columns(
        _adaptive_binning(real[columns], columns),
        _adaptive_binning(synthetic[columns], columns),
        columns,
    )
    # The Sentinel will target valid_cols (categorical hubs) but can use all columns as context
    real_f = all_f_real
    synthetic_f = all_f_syn

    # UNIFIED STATEFUL ENCODING: Project into categorical manifold
    encoder = ManifoldEncoder()
    encoder.fit(real_f)
    x_real_cat = encoder.transform(real_f)
    x_syn_cat = encoder.transform(synthetic_f)

    # 1. Categorical Layer: Manifold Sentinels
    oracle = LogicalSentinelEnsemble(
        top_n_hubs=hif_hubs, max_depth=hif_depth, random_state=random_state
    )
    if verbose:
        print(
            "  [HIF Audit] Auditing Sentinel Logical Consistency...", end="", flush=True
        )
    oracle.fit(
        real_f,
        hif_epochs=hif_epochs,
        verbose=verbose,
        x_precomputed=x_real_cat,
        potential_hubs=valid_cols,
    )
    _, cat_penalties, meta = oracle.audit(synthetic_f, x_precomputed=x_syn_cat)
    # 2. Continuous Layer: Neighbor-Invariant Continuity (NIC)
    nic_violation_rate = 0.0
    nic_penalties = np.zeros(len(synthetic))
    if skipped_cols:
        if verbose:
            print(
                "  [HIF NIC] Training Neighbor-Invariant Continuity Auditor...",
                end="",
                flush=True,
            )
        if not oracle.hubs:
            if verbose:
                print(" Done (No predictive hubs found).")
            # Early exit: if no hubs are found, the manifold is likely trivial
            # return zero penalties for NIC
            nic_penalties = np.zeros(len(synthetic))
            nic_violation_rate = 0.0
        else:
            # MCC HARDENING: Ensure the NIC manifold only sees the hubs relevant to the audit
            # and re-calculates the encoding to match the active feature subspace.
            nic_auditor = NeighborInvariantContinuity(random_state=random_state)
            nic_auditor.fit(
                real_f[oracle.hubs],
                real[skipped_cols],
                x_precomputed=None,  # Force local subspace encoding
                verbose=verbose,
            )
            _, nic_penalties_raw = nic_auditor.score(
                synthetic_f[oracle.hubs], synthetic[skipped_cols], x_precomputed=None
            )
            # HARDENING: Manifold Coherence Coupling (MCC)
            # Continuous continuity cannot exist if the underlying categorical manifold
            # has ruptured. We anchor NIC penalties to the LSE baseline to prevent
            # 'Marginal Paradox' drops in high-noise regimes.
            nic_penalties = np.maximum(nic_penalties_raw, cat_penalties)
            nic_violation_rate = (nic_penalties > 0.5).mean()
            if verbose:
                print("Done.")

    # 3. Structural Layer: Logical Rules (Hard Constraints)
    if verbose:
        print(
            "  [HIF Rules] Mining and checking Implication Rules...", end="", flush=True
        )
    # Pass pre-binned data to avoid redundant processing
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
    # Convert rule violations to row-level binary penalties via the pre-computed mask
    rule_penalties = np.zeros(len(synthetic))
    if rule_result.get("num_rows_with_violations", 0) > 0:
        rule_penalties = rule_result.get("row_violation_mask", np.zeros(len(synthetic)))

    if verbose:
        print(f"Done ({rule_result['num_rules_mined']} rules).")

    # GEOMETRIC AGGREGATION: Integrative validities
    eps = 1e-6
    # Combine all active auditors: Sentinels (Cat), NIC (Cont), and Rules (Structural)
    active_components = [np.clip(1.0 - cat_penalties, eps, 1.0)]
    if skipped_cols:
        active_components.append(np.clip(1.0 - nic_penalties, eps, 1.0))

    # Layer 3 (Rules) as a Kill Switch:
    # If a rule is violated (penalty=1.0), validity becomes eps (near-zero)
    active_components.append(np.clip(1.0 - rule_penalties, eps, 1.0))

    # Calculate geometric mean across active auditors
    log_sum = sum(np.log(c) for c in active_components)
    row_validity = np.exp(log_sum / len(active_components))

    row_penalties = 1.0 - row_validity
    hif_score_val = row_validity.mean()

    num_violations = (row_penalties > 0.5).sum()
    violation_rate = float(num_violations / len(row_penalties))
    cat_violation_rate = (cat_penalties > 0.5).mean()

    return {
        "hif_score": round(float(hif_score_val), 4),
        "row_penalties": row_penalties,
        "violation_rate": round(violation_rate, 4),
        "mean_penalty": round(float(row_penalties.mean()), 4),
        "num_violations": int(num_violations),
        "violation_threshold": 0.5,
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
    }


def mine_implication_rules(
    real: pd.DataFrame,
    columns: list[str],
    min_confidence: float = 0.95,
    min_support: float = 0.005,
    max_rules: int = 25,
    min_lift: float = 1.0,
    max_antecedents: int = 2,
    random_state: int | None = None,
) -> list[dict[str, Any]]:
    n_rows = len(real)
    if n_rows == 0:
        return []
    min_support_count = max(1, int(np.ceil(min_support * n_rows)))
    rules: list[dict[str, Any]] = []
    cat = pd.DataFrame(index=real.index)
    for col in columns:
        col_data = real[col]
        # Skip binning if the data is already discrete/categorical
        if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(
            col_data
        ):
            cat[col] = col_data.astype(str)
            continue

        n_unique = col_data.nunique()
        if pd.api.types.is_numeric_dtype(col_data) and n_unique > 50:
            try:
                quantized = pd.qcut(
                    col_data, RULE_QUANTIZATION_BINS, labels=None, duplicates="drop"
                )
                cat[col] = quantized.astype(str)
            except Exception:
                cat[col] = col_data.astype(str)
        else:
            cat[col] = col_data.astype(str)
    frequent_items: dict[tuple[str, str], int] = {}
    for col in columns:
        counts = cat[col].value_counts()
        for val, count in counts.items():
            if count >= min_support_count:
                frequent_items[(col, str(val))] = int(count)
    item_masks: dict[tuple[str, str], np.ndarray] = {}
    for col, val in frequent_items.keys():
        item_masks[(col, val)] = cat[col].values == val
    frequent_sets_by_size: dict[int, list[tuple[tuple[str, str], ...]]] = {
        1: [(item,) for item in frequent_items.keys()]
    }
    support_counts: dict[tuple[tuple[str, str], ...], int] = {
        (item,): count for item, count in frequent_items.items()
    }
    max_k = max_antecedents + 1
    for k in range(2, max_k + 1):
        prev_frequent = frequent_sets_by_size.get(k - 1, [])
        if not prev_frequent:
            break
        candidates_set = set()
        prefix_map: dict[tuple[tuple[str, str], ...], list[tuple[str, str]]] = {}
        for itemset in prev_frequent:
            prefix = itemset[:-1]
            last_item = itemset[-1]
            if prefix not in prefix_map:
                prefix_map[prefix] = []
            prefix_map[prefix].append(last_item)

        for prefix, items in prefix_map.items():
            # FURTHER OPTIMIZATION: Group items by feature to avoid cross-product of same feature
            feature_groups: dict[str, list[tuple[str, str]]] = {}
            for item in items:
                feat = item[0]
                if feat not in feature_groups:
                    feature_groups[feat] = []
                feature_groups[feat].append(item)

            feat_list = list(feature_groups.keys())
            for i in range(len(feat_list)):
                for j in range(i + 1, len(feat_list)):
                    # Cross-join only between DIFFERENT features
                    for item_a in feature_groups[feat_list[i]]:
                        for item_b in feature_groups[feat_list[j]]:
                            cand = tuple(sorted(list(prefix) + [item_a, item_b]))
                            candidates_set.add(cand)
        # SORT CANDIDATES for determinism before potential sampling
        candidates = sorted(candidates_set)
        if not candidates:
            break

        # PRUNING: Limit candidate explosion for high-cardinality datasets
        if len(candidates) > MAX_RULE_CANDIDATES:
            rng = check_random_state(random_state)
            # Use deterministic local sampling instead of global random.seed
            indices = rng.choice(len(candidates), MAX_RULE_CANDIDATES, replace=False)
            candidates = [candidates[i] for i in sorted(indices)]

        current_frequent = []
        for cand in candidates:
            # Use precomputed masks for O(1) intersection
            mask = item_masks[cand[0]]
            for i in range(1, len(cand)):
                mask = mask & item_masks[cand[i]]

            count = int(mask.sum())
            if count >= min_support_count:
                support_counts[cand] = count
                current_frequent.append(cand)

                # Rule generation
                for i in range(len(cand)):
                    consequent_item = cand[i]
                    antecedent_items = tuple(cand[:i] + cand[i + 1 :])

                    # Safety check for missing antecedents in support_counts
                    ant_count = support_counts.get(antecedent_items)
                    if ant_count is None or ant_count == 0:
                        continue

                    confidence = count / ant_count
                    if confidence >= min_confidence:
                        consequent_support = support_counts[(consequent_item,)] / n_rows
                        lift = confidence / consequent_support
                        if lift >= min_lift:
                            antecedents = [
                                {"feature": f, "value": v} for f, v in antecedent_items
                            ]
                            rules.append(
                                {
                                    "antecedents": antecedents,
                                    "antecedent_repr": _ANTE_JOIN.join(
                                        f"{a['feature']}={a['value']}"
                                        for a in antecedents
                                    ),
                                    "consequent_feature": consequent_item[0],
                                    "consequent_value": consequent_item[1],
                                    "support": round(count / n_rows, 4),
                                    "confidence": round(confidence, 4),
                                    "lift": round(lift, 4),
                                    "support_count": count,
                                    "antecedent_count": support_counts[
                                        antecedent_items
                                    ],
                                    "antecedent_feature": antecedents[0]["feature"]
                                    if len(antecedents) == 1
                                    else None,
                                }
                            )
        if not current_frequent:
            break
        frequent_sets_by_size[k] = current_frequent
    # Use lexicographical tie-breakers for total determinism
    rules.sort(
        key=lambda x: (
            x["confidence"],
            x["lift"],
            x["support"],
            x["antecedent_repr"],
            x["consequent_feature"],
            x["consequent_value"],
        ),
        reverse=True,
    )
    return rules[:max_rules]


def rule_violation_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str],
    min_confidence: float = 0.95,
    min_support: float = 0.005,
    max_rules: int = 25,
    min_lift: float = 1.0,
    max_antecedents: int = 2,
    max_violation_examples: int = 20,
    random_state: int | None = None,
    pre_binned: bool = False,
) -> dict[str, Any]:
    rule_diagnostics: List[Dict[str, Any]] = []
    violation_examples: List[Dict[str, Any]] = []

    if not columns or max_rules < 1:
        return {
            "rule_violation_rate": 0.0,
            "num_rule_violations": 0,
            "num_rules_mined": 0,
            "rows_with_rule_violations": 0,
            "rows_evaluated": len(synthetic),
            "top_violated_rules": [],
            "violation_examples": [],
        }

    if pre_binned:
        real_f, syn_f = real, synthetic
    else:
        real_f, syn_f = _canonicalize_code_columns(
            _adaptive_binning(real, columns),
            _adaptive_binning(synthetic, columns),
            columns,
        )
    rules = mine_implication_rules(
        real_f,
        columns=columns,
        min_confidence=min_confidence,
        min_support=min_support,
        max_rules=max_rules,
        min_lift=min_lift,
        max_antecedents=max_antecedents,
        random_state=random_state,
    )

    if not rules:
        return {
            "rule_violation_rate": 0.0,
            "num_rule_violations": 0,
            "num_rules_mined": 0,
            "rows_with_rule_violations": 0,
            "rows_evaluated": len(syn_f),
            "top_violated_rules": [],
            "violation_examples": [],
        }

    row_violation_mask = np.zeros(len(syn_f), dtype=bool)
    total_violations = 0

    for rule in rules:
        ants = rule["antecedents"]
        ant_mask = pd.Series(True, index=syn_f.index)
        for ant in ants:
            ant_mask &= syn_f[ant["feature"]].astype(str).eq(str(ant["value"]))

        if not ant_mask.any():
            continue

        violates = ant_mask & (
            ~syn_f[rule["consequent_feature"]].astype(str).eq(rule["consequent_value"])
        )
        row_violation_mask |= violates.to_numpy()
        v_count = int(violates.sum())
        total_violations += v_count
        if v_count > 0:
            rule_diagnostics.append({**rule, "violation_count": v_count})
            for ridx in syn_f.index[violates][:3]:
                if len(violation_examples) >= max_violation_examples:
                    break
                violation_examples.append(
                    {
                        "row_index": str(ridx),
                        "antecedent": rule["antecedent_repr"],
                        "expected": f"{rule['consequent_feature']}={rule['consequent_value']}",
                        "actual": f"{rule['consequent_feature']}={syn_f.loc[ridx, rule['consequent_feature']]}",
                    }
                )

    rule_diagnostics.sort(key=lambda d: d["violation_count"], reverse=True)
    return {
        "rule_violation_rate": round(row_violation_mask.sum() / len(syn_f), 4),
        "total_rule_hits": int(
            total_violations
        ),  # Sum of all violations across all rules
        "num_rules_mined": int(len(rules)),
        "num_rows_with_violations": int(
            row_violation_mask.sum()
        ),  # Unique rows affected
        "rows_evaluated": int(len(syn_f)),
        "row_violation_mask": row_violation_mask.astype(float),
        "top_violated_rules": rule_diagnostics[:10],
        "violation_examples": violation_examples,
    }
