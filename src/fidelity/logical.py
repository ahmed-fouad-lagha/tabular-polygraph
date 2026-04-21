"""
HIF: Neurosymbolic Logical Constraint Validator for Synthetic Tabular Data.

Trains an under-complete autoencoder on real data to extract semantic boundaries,
and evaluates synthetic data using a Continuous Semantic Severity Penalty (CSSP).
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
            if df[col].nunique() <= 1:
                df_binned[col] = "bin_0"
                continue
            try:
                bins = pd.qcut(df[col], q=n_bins, labels=False, duplicates="drop")
                df_binned[col] = bins.apply(
                    lambda x: f"bin_{int(val)}" if not pd.isna(val := x) else x
                )
            except Exception:
                try:
                    df_binned[col] = pd.cut(df[col], bins=n_bins, labels=False).apply(
                        lambda x: f"bin_{int(val)}" if not pd.isna(val := x) else x
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
        self.is_trained = False

    def _calculate_dependency_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discover 'Dependency Hubs' using Symmetric Mutual Information."""
        cols = df.columns
        n = len(cols)
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                if df[cols[i]].nunique() <= 1 or df[cols[j]].nunique() <= 1:
                    mi = 0.0
                else:
                    mi = mutual_info_score(df[cols[i]], df[cols[j]])
                matrix[i, j] = mi
                matrix[j, i] = mi
        return pd.DataFrame(matrix, index=cols, columns=cols)

    def fit(self, df: pd.DataFrame, hif_epochs: int = 10, verbose: bool = True):
        """Train Sentinels on Ground-Truth 'Laws'."""
        if len(df) < 50:
            return

        mi_matrix = self._calculate_dependency_matrix(df)
        hub_scores = mi_matrix.sum(axis=1).sort_values(ascending=False)
        self.hubs = hub_scores.head(self.top_n_hubs).index.tolist()

        if verbose:
            print(f"  [HIF Hubs] Discovered {len(self.hubs)} Logical Hubs: {self.hubs}")

        X_full = pd.get_dummies(df, drop_first=True)

        for hub_col in self.hubs:
            other_cols = [c for c in df.columns if c != hub_col]
            prefixes = tuple(f"{c}_" for c in other_cols)
            hub_features = [c for c in X_full.columns if c.startswith(prefixes)]
            
            X = X_full[hub_features]
            y = df[hub_col]

            n_trees = max(10, hif_epochs * 10)
            clf = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            clf.fit(X, y)
            self.sentinels[hub_col] = clf

            probs = clf.predict_proba(X)
            max_probs = np.max(probs, axis=1)
            self.confidence_floors[hub_col] = float(np.percentile(max_probs, 0.5))

        self.is_trained = True

    def audit(self, df: pd.DataFrame) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        """Audit synthetic rows for 'Logical Ruptures'."""
        if not self.is_trained:
            return 1.0, np.zeros(len(df)), {}

        row_penalties = np.zeros(len(df))
        traces = []

        X_full_syn = pd.get_dummies(df, drop_first=True)

        for hub_col in self.hubs:
            clf = self.sentinels[hub_col]
            X = X_full_syn.reindex(columns=clf.feature_names_in_, fill_value=0)

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
            penalty = np.clip((raw_diff - soft_threshold) / (1.0 - soft_threshold), 0, 1)

            row_penalties = np.maximum(row_penalties, penalty)

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


class NeighborContinuityScorer:
    """Audits continuous features against categorical manifold."""
    def __init__(self, random_state: int = 42):
        self.regressors: Dict[str, Ridge] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.z_thresholds: Dict[str, float] = {}
        self.pca = None
        self.random_state = random_state

    def fit(self, categorical_df: pd.DataFrame, continuous_df: pd.DataFrame):
        valid_cols = [c for c in continuous_df.columns if continuous_df[c].nunique() > 1]
        if not valid_cols: return
        active_df = continuous_df[valid_cols]
        self.manifold_features = pd.get_dummies(categorical_df, drop_first=True)
        n_comp = min(32, self.manifold_features.shape[1], self.manifold_features.shape[0])
        if n_comp < 1: return
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        latent = self.pca.fit_transform(self.manifold_features)
        for col in active_df.columns:
            y = active_df[col].values
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
            reg = Ridge(alpha=1.0, random_state=self.random_state)
            reg.fit(latent, y_scaled)
            y_pred = reg.predict(latent)
            residuals = np.abs(y_scaled - y_pred)
            self.regressors[col] = reg
            self.scalers[col] = scaler
            self.z_thresholds[col] = float(np.percentile(residuals, 95))

    def score(self, categorical_df: pd.DataFrame, continuous_df: pd.DataFrame) -> Tuple[float, np.ndarray]:
        if self.pca is None or not self.regressors: return 1.0, np.zeros(len(continuous_df))
        syn_dummy = pd.get_dummies(categorical_df, drop_first=True).reindex(columns=self.manifold_features.columns, fill_value=0)
        latent = self.pca.transform(syn_dummy)
        row_penalties = np.zeros(len(continuous_df))
        for col in continuous_df.columns:
            if col not in self.regressors: continue
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
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    np.random.seed(int(random_state))
    random.seed(int(random_state))
    if columns is None: columns = real.columns.intersection(synthetic.columns).tolist()
    valid_cols, skipped_cols = [], []
    for col in columns:
        if pd.api.types.is_numeric_dtype(real[col]): skipped_cols.append(col)
        else: valid_cols.append(col)
    if not valid_cols:
        return {"hif_score": 1.0, "row_penalties": np.zeros(len(synthetic)), "violation_rate": 0.0, "mean_penalty": 0.0, "num_violations": 0, "columns_used": []}

    real_f, synthetic_f = _canonicalize_code_columns(_adaptive_binning(real[valid_cols], valid_cols), _adaptive_binning(synthetic[valid_cols], valid_cols), valid_cols)
    oracle = LogicalSentinelEnsemble(top_n_hubs=hif_hubs, max_depth=hif_depth, random_state=random_state)
    oracle.fit(real_f, hif_epochs=hif_epochs, verbose=verbose)
    _, cat_penalties, meta = oracle.audit(synthetic_f)

    nic_violation_rate = 0.0
    nic_penalties = np.zeros(len(synthetic))
    if skipped_cols:
        nic_auditor = NeighborContinuityScorer(random_state=random_state)
        nic_auditor.fit(real_f[oracle.hubs], real[skipped_cols])
        _, nic_penalties = nic_auditor.score(synthetic_f[oracle.hubs], synthetic[skipped_cols])
        nic_violation_rate = (nic_penalties > 0.5).mean()

    row_validity = np.sqrt((1.0 - cat_penalties) * (1.0 - nic_penalties))
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
    n_rows = len(real)
    if n_rows == 0: return []
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
            except Exception: cat[col] = col_data.astype(str)
        else: cat[col] = col_data.astype(str)
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
        if not prev_frequent: break
        candidates_set = set()
        for i in range(len(prev_frequent)):
            for j in range(i + 1, len(prev_frequent)):
                l1, l2 = prev_frequent[i], prev_frequent[j]
                if l1[:-1] == l2[:-1]:
                    if l2[-1][0] not in {item[0] for item in l1}:
                        candidates_set.add(tuple(sorted(list(l1) + [l2[-1]])))
        candidates = list(candidates_set)
        if not candidates: break
        MAX_CANDIDATES_PER_LEVEL = 100000
        if len(candidates) > MAX_CANDIDATES_PER_LEVEL:
            random.seed(random_state or 42)
            candidates = random.sample(candidates, MAX_CANDIDATES_PER_LEVEL)
        current_frequent = []
        for cand in candidates:
            mask = item_masks[cand[0]]
            for i in range(1, len(cand)): mask = mask & item_masks[cand[i]]
            count = int(mask.sum())
            if count >= min_support_count:
                support_counts[cand] = count
                current_frequent.append(cand)
                for i in range(len(cand)):
                    consequent_item = cand[i]
                    antecedent_items = tuple(cand[:i] + cand[i+1:])
                    confidence = count / support_counts[antecedent_items]
                    if confidence >= min_confidence:
                        consequent_support = support_counts[(consequent_item,)] / n_rows
                        lift = confidence / consequent_support
                        if lift >= min_lift:
                            antecedents = [{"feature": f, "value": v} for f, v in antecedent_items]
                            rules.append({
                                "antecedents": antecedents,
                                "antecedent_repr": _ANTE_JOIN.join(f"{a['feature']}={a['value']}" for a in antecedents),
                                "consequent_feature": consequent_item[0],
                                "consequent_value": consequent_item[1],
                                "support": round(count / n_rows, 4),
                                "confidence": round(confidence, 4),
                                "lift": round(lift, 4),
                                "support_count": count,
                                "antecedent_count": support_counts[antecedent_items],
                                "antecedent_feature": antecedents[0]["feature"] if len(antecedents) == 1 else None
                            })
        if not current_frequent: break
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
    if not columns or max_rules < 1: return {"rule_violation_rate": 0.0, "num_rule_violations": 0, "num_rules_mined": 0, "rows_with_rule_violations": 0, "rows_evaluated": len(synthetic), "top_violated_rules": [], "violation_examples": []}
    real_f, syn_f = _canonicalize_code_columns(_adaptive_binning(real, columns), _adaptive_binning(synthetic, columns), columns)
    rules = mine_implication_rules(real_f, columns=columns, min_confidence=min_confidence, min_support=min_support, max_rules=max_rules, min_lift=min_lift, max_antecedents=max_antecedents, random_state=random_state)
    if not rules: return {"rule_violation_rate": 0.0, "num_rule_violations": 0, "num_rules_mined": 0, "rows_with_rule_violations": 0, "rows_evaluated": len(syn_f), "top_violated_rules": [], "violation_examples": []}
    row_violation_mask = np.zeros(len(syn_f), dtype=bool)
    total_violations = 0
    rule_diagnostics, violation_examples = [], []
    for rule in rules:
        ants = rule["antecedents"]
        ant_mask = pd.Series(True, index=syn_f.index)
        for ant in ants: ant_mask &= syn_f[ant["feature"]].astype(str).eq(str(ant["value"]))
        if not ant_mask.any(): continue
        violates = ant_mask & (~syn_f[rule["consequent_feature"]].astype(str).eq(rule["consequent_value"]))
        row_violation_mask |= violates.to_numpy()
        v_count = int(violates.sum())
        total_violations += v_count
        if v_count > 0:
            rule_diagnostics.append({**rule, "violation_count": v_count})
            for ridx in syn_f.index[violates][:3]:
                if len(violation_examples) >= max_violation_examples: break
                violation_examples.append({"row_index": str(ridx), "antecedent": rule["antecedent_repr"], "expected": f"{rule['consequent_feature']}={rule['consequent_value']}", "actual": f"{rule['consequent_feature']}={syn_f.loc[ridx, rule['consequent_feature']]}"})
    rule_diagnostics.sort(key=lambda d: d["violation_count"], reverse=True)
    return {"rule_violation_rate": round(row_violation_mask.sum() / len(syn_f), 4), "num_rule_violations": int(total_violations), "num_rules_mined": int(len(rules)), "rows_with_rule_violations": int(row_violation_mask.sum()), "rows_evaluated": int(len(syn_f)), "top_violated_rules": rule_diagnostics[:10], "violation_examples": violation_examples}
