"""
HIF: Neurosymbolic Logical Constraint Validator for Synthetic Tabular Data.

Trains an under-complete autoencoder on real data to extract semantic boundaries,
and evaluates synthetic data using a Continuous Semantic Severity Penalty (CSSP).

Mathematical Foundation
-----------------------
The frozen autoencoder defines a masked conditional semantic score by hiding one
feature group at a time and measuring how much probability mass it assigns to the
observed category given the remaining features.
CSSP(x_g,i) = 1 - P(category_chosen | x_{-g}) measures logical impossibility.
HIF Score ∈ [0, 1] is the geometric mean of the per-feature conditional
probabilities, averaged across synthetic rows.
"""

import random
import numpy as np
import pandas as pd
from typing import Any, Tuple, Dict, List
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score as mutual_info_score
from sklearn.decomposition import PCA


_ANTE_JOIN = " & "


def _adaptive_binning(
    df: pd.DataFrame, columns: list[str], n_bins: int = 5
) -> pd.DataFrame:
    """Discretize continuous numeric columns into quantile-based bins for logical analysis."""
    df_binned = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                # Use qcut for equal-frequency bins (more robust to outliers)
                # We use labels=False to get integers, then map to ranges for readability.
                bins = pd.qcut(df[col], q=n_bins, labels=False, duplicates="drop")
                # Create descriptive labels like "Income(bin_0)"
                df_binned[col] = bins.apply(
                    lambda x: f"bin_{val}" if not pd.isna(val := x) else x
                )
            except Exception:
                # Fallback: if qcut fails (e.g. all same values), treat as a single bin or drop
                df_binned[col] = df[col].astype(str)
    return df_binned


def _feature_groups_from_encoded_columns(
    encoded_columns: list[str], separator: str = "__"
) -> list[list[int]]:
    """Group one-hot encoded columns by their original feature prefix."""
    feature_to_indices: dict[str, list[int]] = {}
    for index, column_name in enumerate(encoded_columns):
        feature_name = column_name.split(separator, 1)[0]
        feature_to_indices.setdefault(feature_name, []).append(index)
    return [feature_to_indices[feature] for feature in sorted(feature_to_indices)]


def _feature_weight_vector(
    feature_groups: list[list[int]],
    weighting: str,
) -> np.ndarray:
    """Build normalized per-group weights for HIF aggregation."""
    n_groups = len(feature_groups)
    if n_groups == 0:
        return np.array([], dtype=np.float32)

    if weighting == "uniform":
        w = np.ones(n_groups, dtype=np.float64)
    elif weighting == "inverse_log_cardinality":
        cardinalities = np.array([max(1, len(g)) for g in feature_groups], dtype=float)
        w = 1.0 / np.log1p(cardinalities)
    else:
        raise ValueError(
            "Unknown feature_weighting: "
            f"{weighting}. Use 'uniform' or 'inverse_log_cardinality'."
        )

    w = np.clip(w, 1e-9, None)
    w = w / w.sum()
    return w.astype(np.float32)


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


class LogicalSentinelEnsemble:
    """
    Neuro-Symbolic Integrity Oracle (LSE).
    Learns 'Manifold Laws' using Random Forest Sentinels on high-dependency hubs.
    """

    def __init__(self, top_n_hubs: int = 5, random_state: int = 42):
        self.top_n_hubs = top_n_hubs
        self.random_state = random_state
        self.sentinels: Dict[str, RandomForestClassifier] = {}
        self.hubs: List[str] = []
        self.confidence_floors: Dict[str, float] = {}
        self.is_trained = False

    def _calculate_dependency_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discover 'Dependency Hubs' using Normalized Mutual Information."""
        cols = df.columns
        n = len(cols)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                mi = mutual_info_score(df[cols[i]], df[cols[j]])
                matrix[i, j] = mi
        return pd.DataFrame(matrix, index=cols, columns=cols)

    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train Sentinels on Ground-Truth 'Laws'."""
        if len(df) < 50:
            return

        mi_matrix = self._calculate_dependency_matrix(df)
        hub_scores = mi_matrix.sum(axis=1).sort_values(ascending=False)
        self.hubs = hub_scores.head(self.top_n_hubs).index.tolist()

        if verbose:
            print(f"  [HIF Hubs] Discovered {len(self.hubs)} Logical Hubs: {self.hubs}")

        for hub_col in self.hubs:
            other_cols = [c for c in df.columns if c != hub_col]
            # Use categorical encoding for the forest and save feature space
            X = pd.get_dummies(df[other_cols], drop_first=True)
            y = df[hub_col]

            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                random_state=self.random_state,
                n_jobs=-1,
            )
            clf.fit(X, y)
            self.sentinels[hub_col] = clf

            # Establish the 'Certainty Law' (Confidence Floor)
            probs = clf.predict_proba(X)
            max_probs = np.max(probs, axis=1)
            # Use 1st percentile as a strict law boundary
            self.confidence_floors[hub_col] = float(np.percentile(max_probs, 1))

        self.is_trained = True

    def audit(self, df: pd.DataFrame) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        """Audit synthetic rows for 'Logical Ruptures'."""
        if not self.is_trained:
            return 1.0, np.zeros(len(df)), {}

        row_penalties = np.zeros(len(df))
        traces = []

        for hub_col in self.hubs:
            clf = self.sentinels[hub_col]
            other_cols = [c for c in df.columns if c != hub_col]
            X = pd.get_dummies(df[other_cols], drop_first=True)
            X = X.reindex(columns=clf.feature_names_in_, fill_value=0)

            probs = clf.predict_proba(X)
            classes = clf.classes_
            observed_values = df[hub_col].astype(str).values

            # Map observed values to their probabilities (observed vs predicted)
            probs_observed = np.zeros(len(df))
            for class_idx, class_val in enumerate(classes):
                mask = observed_values == str(class_val)
                if mask.any():
                    probs_observed[mask] = probs[mask, class_idx]

            floor = self.confidence_floors[hub_col]
            # Semantic Penalty: Proportional distance of OBSERVED probability from the floor
            penalty = np.clip((floor - probs_observed) / max(1e-5, floor), 0, 1)
            row_penalties = np.maximum(row_penalties, penalty)

            ruptures = probs_observed < (floor * 0.1)  # 10x drop in confidence
            if ruptures.any():
                traces.append(
                    {
                        "column": hub_col,
                        "violations": int(ruptures.sum()),
                        "mean_prob": float(probs_observed[ruptures].mean()),
                    }
                )

        hif_score_val = 1.0 - row_penalties.mean()
        return float(hif_score_val), row_penalties, {"traces": traces}


class NeighborContinuityScorer:
    """
    Audits continuous features by measuring their 'Semantic Adjacency Residual'
    within the categorical manifold. Detects economic hallucinations.
    """

    def __init__(self, random_state: int = 42):
        self.regressors: Dict[str, Ridge] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.z_thresholds: Dict[str, float] = {}
        self.pca = PCA(n_components=32, random_state=random_state)
        self.random_state = random_state

    def fit(self, categorical_df: pd.DataFrame, continuous_df: pd.DataFrame):
        """Establish the 'Semantic Latent Space' using categorical features."""
        # Internal Manifold Projection
        self.manifold_features = pd.get_dummies(categorical_df, drop_first=True)
        latent = self.pca.fit_transform(self.manifold_features)

        for col in continuous_df.columns:
            y = continuous_df[col].values
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

            reg = Ridge(alpha=1.0, random_state=self.random_state)
            reg.fit(latent, y_scaled)

            y_pred = reg.predict(latent)
            residuals = np.abs(y_scaled - y_pred)

            self.regressors[col] = reg
            self.scalers[col] = scaler
            self.z_thresholds[col] = float(np.percentile(residuals, 95))

    def score(
        self, categorical_df: pd.DataFrame, continuous_df: pd.DataFrame
    ) -> Tuple[float, np.ndarray]:
        """Audit continuous features against the learned categorical manifold."""
        if not self.regressors:
            return 1.0, np.zeros(len(continuous_df))

        # Internal Manifold Alignment
        syn_dummy = pd.get_dummies(categorical_df, drop_first=True)
        # Force alignment with ground-truth manifold
        syn_dummy = syn_dummy.reindex(
            columns=self.manifold_features.columns, fill_value=0
        )
        latent = self.pca.transform(syn_dummy)

        row_penalties = np.zeros(len(continuous_df))
        for col in continuous_df.columns:
            y = continuous_df[col].values
            y_scaled = self.scalers[col].transform(y.reshape(-1, 1)).flatten()
            y_pred = self.regressors[col].predict(latent)

            residuals = np.abs(y_scaled - y_pred)

            # Normalize by real-world threshold.
            # Penalty scales from 0 to 1 as residual exceeds expected noise.
            threshold = self.z_thresholds[col]
            if threshold > 0:
                # We use 2.5x the noise floor as the 'Absolute Hallucination' boundary
                col_penalty = np.clip(residuals / (threshold * 2.5), 0, 1)
            else:
                col_penalty = np.zeros_like(residuals)

            # Aggregate using Max-L (maximum-logical-impossibility)
            row_penalties = np.maximum(row_penalties, col_penalty)

        avg_score = 1.0 - row_penalties.mean()
        return float(avg_score), row_penalties


def hif_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Compute HIF (Holistic Integrity Framework) score.

    Uses Logical Sentinel Ensembles (LSE) to discover and audit manifold laws.
    """
    np.random.seed(int(random_state))
    random.seed(int(random_state))

    # Determine columns to use
    if columns is None:
        columns = real.columns.intersection(synthetic.columns).tolist()

    valid_cols = []
    skipped_cols = []
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

    # 1. Pre-process: Discretize and Canonicalize
    real_f_data = _adaptive_binning(real[valid_cols], valid_cols)
    syn_f_data = _adaptive_binning(synthetic[valid_cols], valid_cols)
    real_f, synthetic_f = _canonicalize_code_columns(
        real_f_data, syn_f_data, valid_cols
    )

    # 2. Categorical Integrity Audit (LSE)
    oracle = LogicalSentinelEnsemble(random_state=random_state)
    oracle.fit(real_f, verbose=verbose)
    hif_score_val, row_penalties, meta = oracle.audit(synthetic_f)

    # 3. Continuity Audit (NIC Breakthrough)
    nic_violation_rate = 0.0
    if skipped_cols:
        if verbose:
            print(
                f"  [NIC Audit] Auditing {len(skipped_cols)} continuous features using LSE Manifold."
            )

        # We pass the Hub Categorical Features as the 'Semantic Manifold' foundation
        manifold_cols = oracle.hubs
        nic_auditor = NeighborContinuityScorer(random_state=random_state)
        nic_auditor.fit(real_f[manifold_cols], real[skipped_cols])
        _, nic_penalties = nic_auditor.score(
            synthetic_f[manifold_cols], synthetic[skipped_cols]
        )

        # Multiplicative Integrity: Compounding failures across semantic layers.
        # validity = (1 - cat_error) * (1 - num_error)
        row_validity = (1.0 - row_penalties) * (1.0 - nic_penalties)
        row_penalties = 1.0 - row_validity
        hif_score_val = row_validity.mean()
        nic_violation_rate = (nic_penalties > 0.5).mean()

    # 4. Final Thresholding (Rule-based Alignment)
    # Hallucinations are rows with significant logic gaps (>0.5 penalty)
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
        "columns_used": valid_cols + skipped_cols,
        "traces": meta.get("traces", []),
    }


def mine_implication_rules(
    real: pd.DataFrame,
    columns: list[str],
    min_confidence: float = 0.98,
    min_support: float = 0.01,
    max_rules: int = 200,
    min_lift: float = 1.0,
    max_antecedents: int = 1,
    random_state: int | None = None,
) -> list[dict[str, Any]]:
    """Mine implication rules from real data using level-wise itemset search."""
    n_rows = len(real)
    if n_rows == 0:
        return []

    min_support_count = max(1, int(np.ceil(min_support * n_rows)))
    rules: list[dict[str, Any]] = []

    # Pre-process into "Logical Predicates"
    cat = pd.DataFrame(index=real.index)

    for col in columns:
        col_data = real[col]
        n_unique = col_data.nunique()

        # Heuristic: Numeric columns with more than 50 unique values are quantized
        if pd.api.types.is_numeric_dtype(col_data) and n_unique > 50:
            try:
                # Use qcut for equal-frequency binning (robust logic)
                n_bins = 10
                quantized = pd.qcut(col_data, n_bins, labels=None, duplicates="drop")
                cat[col] = quantized.astype(str)
            except Exception:
                # Fallback to simple binning if qcut fails
                cat[col] = col_data.astype(str)
        else:
            cat[col] = col_data.astype(str)

    # Frequent 1-itemsets (individual column=value pairs)
    frequent_items: dict[tuple[str, str], int] = {}
    for col in columns:
        counts = cat[col].value_counts()
        for val, count in counts.items():
            if count >= min_support_count:
                frequent_items[(col, str(val))] = int(count)

    # Pre-calculate bitmasks for all frequent items (Vertical Data Format / TID-lists)
    item_masks: dict[tuple[str, str], np.ndarray] = {}
    for col, val in frequent_items.keys():
        item_masks[(col, val)] = cat[col].values == val

    # Current frequent itemsets found at size k
    frequent_sets_by_size: dict[int, list[tuple[tuple[str, str], ...]]] = {
        1: [(item,) for item in frequent_items.keys()]
    }

    # Map from itemset tuple to its support count
    support_counts: dict[tuple[tuple[str, str], ...], int] = {
        (item,): count for item, count in frequent_items.items()
    }

    max_k = max_antecedents + 1

    for k in range(2, max_k + 1):
        prev_frequent = frequent_sets_by_size.get(k - 1, [])
        if not prev_frequent:
            break

        candidates = []
        candidates_set = set()
        for i in range(len(prev_frequent)):
            for j in range(i + 1, len(prev_frequent)):
                l1, l2 = prev_frequent[i], prev_frequent[j]
                if l1[:-1] == l2[:-1]:
                    cols1 = {item[0] for item in l1}
                    if l2[-1][0] not in cols1:
                        cand_list = list(l1) + [l2[-1]]
                        candidate = tuple(sorted(cand_list))
                        candidates_set.add(candidate)

        candidates = list(candidates_set)
        if not candidates:
            break

        MAX_CANDIDATES_PER_LEVEL = 100000
        if len(candidates) > MAX_CANDIDATES_PER_LEVEL:
            import random

            random.seed(random_state or 42)
            candidates = random.sample(candidates, MAX_CANDIDATES_PER_LEVEL)

        if not candidates:
            break

        current_frequent = []
        for cand in candidates:
            mask = item_masks[cand[0]]
            for i in range(1, len(cand)):
                mask = mask & item_masks[cand[i]]

            count = int(mask.sum())

            if count >= min_support_count:
                support_counts[cand] = count
                current_frequent.append(cand)

                for i in range(len(cand)):
                    consequent_item = cand[i]
                    antecedent_items = tuple(cand[:i] + cand[i + 1 :])
                    antecedent_count = support_counts[antecedent_items]
                    confidence = count / antecedent_count

                    if confidence >= min_confidence:
                        consequent_support = support_counts[(consequent_item,)] / n_rows
                        lift = confidence / consequent_support

                        if lift >= min_lift:
                            antecedents = [
                                {"feature": f, "value": v} for f, v in antecedent_items
                            ]
                            rule = {
                                "antecedents": antecedents,
                                "antecedent_repr": _ANTE_JOIN.join(
                                    f"{a['feature']}={a['value']}" for a in antecedents
                                ),
                                "consequent_feature": consequent_item[0],
                                "consequent_value": consequent_item[1],
                                "support": round(count / n_rows, 4),
                                "confidence": round(confidence, 4),
                                "lift": round(lift, 4),
                                "support_count": count,
                                "antecedent_count": antecedent_count,
                            }
                            if len(antecedents) == 1:
                                rule["antecedent_feature"] = antecedents[0]["feature"]
                            rules.append(rule)

        if not current_frequent:
            break
        frequent_sets_by_size[k] = current_frequent

    rules.sort(key=lambda x: (x["confidence"], x["lift"], x["support"]), reverse=True)
    return rules[:max_rules]


def rule_violation_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str],
    min_confidence: float = 0.98,
    min_support: float = 0.01,
    max_rules: int = 200,
    min_lift: float = 1.0,
    max_antecedents: int = 1,
    max_violation_examples: int = 20,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Evaluate synthetic rows against implication rules mined from real data."""
    if not columns:
        return {
            "rule_violation_rate": 0.0,
            "num_rule_violations": 0,
            "num_rules_mined": 0,
            "rows_with_rule_violations": 0,
            "rows_evaluated": len(synthetic),
            "top_violated_rules": [],
            "violation_examples": [],
        }
    if max_rules < 1 or max_antecedents < 1:
        return {
            "rule_violation_rate": 0.0,
            "num_rule_violations": 0,
            "num_rules_mined": 0,
            "rows_with_rule_violations": 0,
            "rows_evaluated": len(synthetic),
            "top_violated_rules": [],
            "violation_examples": [],
        }

    real_norm, synthetic_norm = _canonicalize_code_columns(real, synthetic, columns)
    real_f_data = _adaptive_binning(real_norm, columns)
    synthetic_f_data = _adaptive_binning(synthetic_norm, columns)

    rules = mine_implication_rules(
        real_f_data,
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
            "rows_evaluated": len(synthetic_f_data),
            "top_violated_rules": [],
            "violation_examples": [],
        }

    row_violation_mask = np.zeros(len(synthetic_f_data), dtype=bool)
    total_violations = 0
    rule_diagnostics: list[dict[str, Any]] = []
    violation_examples: list[dict[str, Any]] = []

    for rule in rules:
        antecedents = rule.get("antecedents") or [
            {
                "feature": rule.get("antecedent_feature"),
                "value": rule.get("antecedent_value"),
            }
        ]
        cons_col = rule["consequent_feature"]
        cons_val = rule["consequent_value"]

        ant_mask = pd.Series(True, index=synthetic_f_data.index)
        for ant in antecedents:
            ant_mask &= (
                synthetic_f_data[ant["feature"]].astype(str).eq(str(ant["value"]))
            )
        if not ant_mask.any():
            continue

        violates = ant_mask & (~synthetic_f_data[cons_col].astype(str).eq(cons_val))
        row_violation_mask |= violates.to_numpy()
        violation_count = int(violates.sum())
        total_violations += violation_count

        if violation_count > 0:
            rule_diagnostics.append(
                {
                    "antecedent_feature": rule.get("antecedent_feature"),
                    "antecedent_value": rule.get("antecedent_value"),
                    "antecedent_repr": rule.get(
                        "antecedent_repr",
                        _ANTE_JOIN.join(
                            f"{a['feature']}={a['value']}" for a in antecedents
                        ),
                    ),
                    "consequent_feature": cons_col,
                    "consequent_value": cons_val,
                    "support": rule["support"],
                    "confidence": rule["confidence"],
                    "lift": rule.get("lift"),
                    "violation_count": violation_count,
                }
            )

            for row_index in synthetic_f_data.index[violates][:3]:
                if len(violation_examples) >= max_violation_examples:
                    break
                actual_value = str(synthetic_f_data.loc[row_index, cons_col])
                violation_examples.append(
                    {
                        "row_index": int(row_index)
                        if isinstance(row_index, (int, np.integer))
                        else str(row_index),
                        "antecedent": rule.get(
                            "antecedent_repr",
                            _ANTE_JOIN.join(
                                f"{a['feature']}={a['value']}" for a in antecedents
                            ),
                        ),
                        "expected": f"{cons_col}={cons_val}",
                        "actual": f"{cons_col}={actual_value}",
                    }
                )

    rows_with_violations = int(row_violation_mask.sum())
    denom = max(len(synthetic_f_data), 1)
    rule_diagnostics.sort(key=lambda d: d["violation_count"], reverse=True)
    return {
        "rule_violation_rate": round(rows_with_violations / denom, 4),
        "num_rule_violations": int(total_violations),
        "num_rules_mined": int(len(rules)),
        "rows_with_rule_violations": rows_with_violations,
        "rows_evaluated": int(len(synthetic_f_data)),
        "example_rules": rules[:10],
        "top_violated_rules": rule_diagnostics[:10],
        "violation_examples": violation_examples,
    }
