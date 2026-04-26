"""
HIF: Hybrid Integrity Framework (The Tabular Polygraph).

A neurosymbolic logical constraint validator for synthetic tabular data.
Trains a Logical Sentinel Ensemble (LSE) and Neighbor-Invariant Continuity (NIC)
auditors on ground-truth manifolds to detect semantic hallucinations via the
Continuous Semantic Severity Penalty (CSSP).
"""

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_ANTE_JOIN = " & "


def _adaptive_binning(
    df: pd.DataFrame, columns: list[str], n_bins: int = 5
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

    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names: List[str] = []
        self.feature_map: Dict[str, List[str]] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
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
    ):
        self.top_n_hubs = top_n_hubs
        self.max_depth = max_depth
        self.random_state = random_state
        self.sentinels: Dict[str, RandomForestClassifier] = {}
        self.hubs: List[str] = []
        self.confidence_floors: Dict[str, float] = {}
        self.encoder = ManifoldEncoder()
        self.is_trained: bool = False

    def _calculate_dependency_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discover 'Manifold Hubs' using Symmetric Mutual Information."""
        cols = df.columns
        n = len(cols)
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                if df[cols[i]].nunique() <= 1 or df[cols[j]].nunique() <= 1:
                    mi = 0.0
                else:
                    mi = normalized_mutual_info_score(
                        df[cols[i]], df[cols[j]], average_method="arithmetic"
                    )
                matrix[i, j] = mi
                matrix[j, i] = mi
        return pd.DataFrame(matrix, index=cols, columns=cols)

    def _select_diverse_hubs(self, mi_matrix: pd.DataFrame) -> List[str]:
        """Greedy selection of hubs based on dependency sum and redundancy filter."""
        hub_scores = mi_matrix.sum(axis=1).sort_values(ascending=False)
        potential_hubs = hub_scores.index.tolist()

        selected: List[str] = []
        redundancy_threshold = 0.8

        for candidate in potential_hubs:
            if len(selected) >= self.top_n_hubs:
                break

            # Check redundancy with already selected hubs
            is_redundant = False
            for active in selected:
                correlation = mi_matrix.loc[candidate, active]
                if correlation > redundancy_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                selected.append(candidate)

        return selected

    def fit(
        self,
        df: pd.DataFrame,
        hif_epochs: int = 10,
        verbose: bool = True,
        x_precomputed: pd.DataFrame | None = None,
    ):
        """Train Sentinels using stateful manifold projection."""
        if len(df) < 50:
            return

        mi_matrix = self._calculate_dependency_matrix(df)
        self.hubs = self._select_diverse_hubs(mi_matrix)

        if verbose:
            print(
                f"  [HIF Hubs] Selected {len(self.hubs)} Diverse Manifold Hubs: {self.hubs}"
            )

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
                random_state=self.random_state,
                min_samples_leaf=3,
                max_features="log2",
            )
            clf.fit(X, y)
            self.sentinels[hub_col] = clf
            if verbose:
                print("Done.")

            probs = clf.predict_proba(X)
            classes = clf.classes_
            y_str = y.astype(str).values
            probs_true = np.zeros(len(y))
            for idx, cls in enumerate(classes):
                probs_true[y_str == str(cls)] = probs[y_str == str(cls), idx]
            self.confidence_floors[hub_col] = float(np.percentile(probs_true, 1.0))

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

            # CALIBRATION: Nonlinear error response
            raw_diff = floor - probs_observed
            soft_threshold = 0.1
            penalty = np.clip(
                (raw_diff - soft_threshold) / (1.0 - soft_threshold), 0, 1
            )

            # ATOMIC AGGREGATION: Use 1 - Product(1 - Penalty) for higher sensitivity
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
    Audits continuous features against categorical manifold using spectral reconstruction.
    """

    def __init__(self, random_state: int = 42):
        self.regressors: Dict[str, RidgeCV] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.z_thresholds: Dict[str, float] = {}
        self.pca: PCA | None = None
        self.encoder = ManifoldEncoder()
        self.random_state = random_state

    def fit(
        self,
        categorical_df: pd.DataFrame,
        continuous_df: pd.DataFrame,
        x_precomputed: pd.DataFrame | None = None,
        verbose: bool = True,
    ):
        """Fit spectral regressors on the training manifold."""
        valid_cols = [
            c for c in continuous_df.columns if continuous_df[c].nunique() > 1
        ]
        if not valid_cols:
            return

        active_df = continuous_df[valid_cols]

        if x_precomputed is not None:
            x_encoded = x_precomputed
            self.encoder.feature_names = x_encoded.columns.tolist()
            self.encoder.is_fitted = True
        else:
            self.encoder.fit(categorical_df)
            x_encoded = self.encoder.transform(categorical_df)

        n_feat = x_encoded.shape[1]
        n_samples = x_encoded.shape[0]
        if n_feat < 1 or n_samples < 2:
            return

        # SPEED HARDENING: Use a fixed component limit for fast spectral reconstruction
        n_comp = min(n_samples, n_feat, 100)
        if verbose:
            print(
                f"  [HIF NIC] Spectral Reconstruction ({n_feat} -> {n_comp} target)...",
                end="",
                flush=True,
            )

        self.pca = PCA(
            n_components=n_comp,
            svd_solver="randomized",
            random_state=self.random_state,
        )
        latent = self.pca.fit_transform(x_encoded)
        if verbose:
            print(f"Done ({self.pca.n_components_} components).")

        self.regressors = {}
        for col in active_df.columns:
            if verbose:
                print(f"  [HIF NIC] Regressing variable '{col}'...", end="", flush=True)
            y = active_df[col].values
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

            # Use RidgeCV for automated regularization optimization
            reg = RidgeCV(alphas=np.logspace(-2, 4, 7))
            reg.fit(latent, y_scaled)
            if verbose:
                print("Done.")

            y_pred = reg.predict(latent)
            residuals = np.abs(y_scaled - y_pred)

            self.regressors[col] = reg
            self.scalers[col] = scaler
            self.z_thresholds[col] = float(np.percentile(residuals, 95))

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
            x_aligned = x_precomputed.reindex(
                columns=self.encoder.feature_names, fill_value=0
            )
        else:
            x_aligned = self.encoder.transform(categorical_df)

        latent = self.pca.transform(x_aligned)
        row_penalties = np.zeros(len(continuous_df))

        for col in continuous_df.columns:
            if col not in self.regressors:
                continue

            y = continuous_df[col].values
            y_scaled = self.scalers[col].transform(y.reshape(-1, 1)).flatten()
            y_pred = self.regressors[col].predict(latent)
            residuals = np.abs(y_scaled - y_pred)

            threshold = self.z_thresholds[col]
            if threshold > 0:
                col_penalty = np.clip((residuals - threshold) / (threshold * 3.0), 0, 1)
            else:
                col_penalty = np.zeros_like(residuals)

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
    oracle.fit(real_f, hif_epochs=hif_epochs, verbose=verbose, x_precomputed=x_real_cat)
    _, cat_penalties, meta = oracle.audit(synthetic_f, x_precomputed=x_syn_cat)
    if verbose:
        print("Done.")

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
        nic_auditor = NeighborInvariantContinuity(random_state=random_state)
        # Re-use the same categorical encoding for NIC manifold
        nic_auditor.fit(
            real_f[oracle.hubs],
            real[skipped_cols],
            x_precomputed=x_real_cat,
            verbose=verbose,
        )
        _, nic_penalties = nic_auditor.score(
            synthetic_f[oracle.hubs], synthetic[skipped_cols], x_precomputed=x_syn_cat
        )
        nic_violation_rate = (nic_penalties > 0.5).mean()
        if verbose:
            print("Done.")

    # 3. Structural Layer: Logical Rules (Hard Constraints)
    if verbose:
        print(
            "  [HIF Rules] Mining and checking Implication Rules...", end="", flush=True
        )
    rule_result = rule_violation_score(
        real,
        synthetic,
        columns=columns,
        min_confidence=rule_min_confidence,
        min_support=rule_min_support,
        max_rules=rule_max_rules,
        min_lift=rule_min_lift,
        max_antecedents=rule_max_antecedents,
        random_state=random_state,
    )
    # Convert rule violations to row-level binary penalties via the pre-computed mask
    rule_penalties = np.zeros(len(synthetic))
    if rule_result["num_rule_violations"] > 0:
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

    return {
        "hif_score": round(float(hif_score_val), 4),
        "row_penalties": row_penalties,
        "violation_rate": round(violation_rate, 4),
        "mean_penalty": round(float(row_penalties.mean()), 4),
        "num_violations": int(num_violations),
        "violation_threshold": 0.5,
        "nic_violation_rate": round(float(nic_violation_rate), 4),
        "rule_violation_rate": round(float(rule_result["rule_violation_rate"]), 4),
        "num_rule_violations": rule_result["num_rule_violations"],
        "num_rules_mined": rule_result["num_rules_mined"],
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
        n_unique = col_data.nunique()
        if pd.api.types.is_numeric_dtype(col_data) and n_unique > 50:
            try:
                quantized = pd.qcut(col_data, 10, labels=None, duplicates="drop")
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
        MAX_CANDIDATES_PER_LEVEL = 10000
        if len(candidates) > MAX_CANDIDATES_PER_LEVEL:
            random.seed(random_state or 42)
            candidates = random.sample(candidates, MAX_CANDIDATES_PER_LEVEL)

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

    real_f, syn_f = _canonicalize_code_columns(
        _adaptive_binning(real, columns), _adaptive_binning(synthetic, columns), columns
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
        "num_rule_violations": int(total_violations),
        "num_rules_mined": int(len(rules)),
        "rows_with_rule_violations": int(row_violation_mask.sum()),
        "rows_evaluated": int(len(syn_f)),
        "row_violation_mask": row_violation_mask.astype(float),
        "top_violated_rules": rule_diagnostics[:10],
        "violation_examples": violation_examples,
    }
