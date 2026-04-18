"""
LCV: Neurosymbolic Logical Constraint Validator for Synthetic Tabular Data.

Trains an under-complete autoencoder on real data to extract semantic boundaries,
and evaluates synthetic data using a Continuous Semantic Severity Penalty (CSSP).

Mathematical Foundation
-----------------------
The frozen autoencoder defines a masked conditional semantic score by hiding one
feature group at a time and measuring how much probability mass it assigns to the
observed category given the remaining features.
CSSP(x_g,i) = 1 - P(category_chosen | x_{-g}) measures logical impossibility.
LCV Score ∈ [0, 1] is the geometric mean of the per-feature conditional
probabilities, averaged across synthetic rows.
"""

from __future__ import annotations
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
from typing import Any, Tuple

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


_ANTE_JOIN = " AND "


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
    """Build normalized per-group weights for LCV aggregation."""
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


class LCVAutoencoder(nn.Module):
    """
    Under-complete denoising autoencoder trained on categorical tabular data.
    Serves as the frozen "Laws of Physics" oracle for semantic validation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        learning_rate: float = 0.005,
        feature_groups: list[list[int]] | None = None,
    ):
        super(LCVAutoencoder, self).__init__()
        assert hidden_dim < input_dim, (
            "Hidden dimension must compress input for under-complete design."
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = 1e-5
        self._group_sizes: list[int] | None = None

        # For categorical tabular data, dropout-denoising is more stable than Gaussian perturbation.
        self.input_dropout = nn.Dropout(p=0.1)

        dim_1 = min(256, max(64, input_dim * 2))
        dim_2 = min(128, max(32, input_dim))
        self._cond_dim = dim_2

        # Deep Bottleneck for semantic extraction
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim_1),
            nn.BatchNorm1d(dim_1),
            nn.ReLU(),
            nn.Linear(dim_1, dim_2),
            nn.ReLU(),
            nn.Linear(dim_2, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, dim_2),
            nn.ReLU(),
            nn.Linear(dim_2, dim_1),
            nn.ReLU(),
            nn.Linear(dim_1, input_dim),
            nn.Sigmoid(),
        )

        # Optional conditional heads for masked group-wise scoring.
        self.conditional_trunk: nn.Module | None = None
        self.group_heads: nn.ModuleList | None = None
        if feature_groups:
            self._init_group_heads(feature_groups)

        self.optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.criterion = nn.BCELoss()
        self.is_trained = False

    def _reset_optimizer(self) -> None:
        self.optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def _init_group_heads(self, feature_groups: list[list[int]]) -> None:
        if not feature_groups:
            raise ValueError("feature_groups must be non-empty for conditional heads.")

        self._group_sizes = [len(g) for g in feature_groups]
        self.conditional_trunk = nn.Sequential(
            nn.Linear(self.hidden_dim, self._cond_dim),
            nn.ReLU(),
        )
        self.group_heads = nn.ModuleList(
            [nn.Linear(self._cond_dim, size) for size in self._group_sizes]
        )
        self._reset_optimizer()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self.input_dropout(x)
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self._encode(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def _forward_group_logits(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.conditional_trunk is None or self.group_heads is None:
            raise RuntimeError(
                "Conditional heads are not initialized. Provide feature_groups to fit/evaluate."
            )
        latent = self._encode(x)
        shared = self.conditional_trunk(latent)
        return [head(shared) for head in self.group_heads]

    def fit(
        self,
        real_tensor: torch.Tensor,
        epochs: int = 30,
        batch_size: int = 128,
        feature_groups: list[list[int]] | None = None,
        verbose: bool = True,
    ):
        """
        Phase 1: Learn structural patterns from real data in O(n) time.

        Parameters
        ----------
        real_tensor : torch.Tensor
            One-hot encoded real data, shape (n_samples, n_features)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for SGD
        feature_groups : list[list[int]] | None
            Optional one-hot feature groups. If provided, trains with masked
            conditional objective by hiding one feature group per batch and
            maximizing the observed category probability within that group.
        verbose : bool
            Print training progress
        """
        if verbose:
            print(f"[LCV] Training on {len(real_tensor)} real records...")

        if len(real_tensor) == 0:
            raise ValueError("LCVAutoencoder.fit received empty training data.")

        if feature_groups:
            requested_sizes = [len(g) for g in feature_groups]
            if self.group_heads is None or self.conditional_trunk is None:
                self._init_group_heads(feature_groups)
            elif self._group_sizes != requested_sizes:
                raise ValueError(
                    "feature_groups mismatch with initialized conditional heads."
                )

        # Ensure at least one batch even for small datasets.
        batch_size = max(1, min(int(batch_size), len(real_tensor)))

        self.train()
        dataset = torch.utils.data.TensorDataset(real_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch_x,) in dataloader:
                self.optimizer.zero_grad()

                if feature_groups:
                    group_idx = int(torch.randint(len(feature_groups), (1,)).item())
                    indices = feature_groups[group_idx]
                    masked_batch = batch_x.clone()
                    masked_batch[:, indices] = 0.0

                    logits_list = self._forward_group_logits(masked_batch)
                    logits = logits_list[group_idx]
                    group_target = batch_x[:, indices]
                    valid_mask = group_target.sum(dim=1) > 0
                    if not valid_mask.any():
                        continue

                    target_idx = group_target[valid_mask].argmax(dim=1).long()
                    loss = F.cross_entropy(logits[valid_mask], target_idx)
                else:
                    outputs = self.forward(batch_x)
                    loss = self.criterion(outputs, batch_x)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if verbose and (epoch + 1) % max(1, epochs // 3) == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"[LCV] Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        self.is_trained = True
        if verbose:
            print("[LCV] Semantic extraction complete.")

    def evaluate(
        self,
        synth_tensor: torch.Tensor,
        feature_groups: list[list[int]] | None = None,
        feature_weights: list[float] | np.ndarray | None = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Phase 2: Grade synthetic data using masked feature-group conditionals.

        Parameters
        ----------
        synth_tensor : torch.Tensor
            One-hot encoded synthetic data, shape (n_samples, n_features)

        Returns
        -------
        lcv_score : float
            LCV fidelity ∈ [0, 1]. Higher is better (1.0 = perfect logical alignment).
        row_penalties : np.ndarray
            Per-row Continuous Semantic Severity Penalty ∈ [0, 1].
        """
        if not self.is_trained:
            raise RuntimeError("LCVAutoencoder must be fitted first.")

        self.eval()
        with torch.no_grad():
            if feature_groups:
                feature_scores = []
                for group_idx, indices in enumerate(feature_groups):
                    masked_tensor = synth_tensor.clone()
                    masked_tensor[:, indices] = 0.0

                    logits_list = self._forward_group_logits(masked_tensor)
                    probs = torch.softmax(logits_list[group_idx], dim=1)
                    group_target = synth_tensor[:, indices]
                    valid_mask = group_target.sum(dim=1) > 0

                    selected_probs = torch.ones(
                        synth_tensor.shape[0], device=synth_tensor.device
                    )
                    if valid_mask.any():
                        target_idx = group_target[valid_mask].argmax(dim=1).long()
                        selected_probs[valid_mask] = probs[valid_mask, target_idx]

                    feature_scores.append(selected_probs.clamp(1e-6, 1.0))

                feature_matrix = torch.stack(feature_scores, dim=1).clamp_min(1e-6)
                if feature_weights is None:
                    per_row_score = torch.exp(torch.log(feature_matrix).mean(dim=1))
                else:
                    weights = torch.as_tensor(
                        feature_weights,
                        dtype=feature_matrix.dtype,
                        device=feature_matrix.device,
                    )
                    if weights.ndim != 1 or weights.numel() != feature_matrix.shape[1]:
                        raise ValueError(
                            "feature_weights must be a 1D vector with one weight per feature group"
                        )
                    weights = weights.clamp_min(1e-9)
                    weights = weights / weights.sum()
                    per_row_score = torch.exp(
                        (torch.log(feature_matrix) * weights).sum(dim=1)
                    )
                row_penalties = (1.0 - per_row_score).cpu().numpy()
                lcv_score = float(per_row_score.mean().item())
                return lcv_score, row_penalties

            # Backward-compatible fallback: use the stricter minimum-over-hot-bits score.
            expected_probs = self.forward(synth_tensor)
            chosen_probs = expected_probs * synth_tensor
            masked_probs = chosen_probs.clone()
            masked_probs[synth_tensor == 0] = 1.0
            min_row_prob = masked_probs.min(dim=1).values
            row_penalties_tensor = 1.0 - min_row_prob
            lcv_score = 1.0 - row_penalties_tensor.mean().item()
            return lcv_score, row_penalties_tensor.cpu().numpy()


def lcv_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
    epochs: int = 30,
    random_state: int = 42,
    feature_weighting: str = "inverse_log_cardinality",
    verbose: bool = True,
) -> dict:
    """
    Compute LCV logical constraint validation score.

    Trains an autoencoder on categorical real data and evaluates whether
    the synthetic data respects the learned semantic boundaries.

    Parameters
    ----------
    real : pd.DataFrame
        Real tabular data
    synthetic : pd.DataFrame
        Synthetic tabular data
    columns : list[str] | None
        Columns to evaluate. If None, uses all categorical columns present in both.
    epochs : int
        Training epochs for autoencoder
    random_state : int
        Random seed for deterministic LCV training/evaluation
    feature_weighting : str
        Aggregation weighting for feature groups: "uniform" or
        "inverse_log_cardinality".
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Contains:
        - lcv_score : float ∈ [0, 1]
        - row_penalties : np.ndarray
        - violation_rate : float (fraction of rows with penalty > 0.5)
        - mean_penalty : float
        - columns_used : list[str]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("LCV requires PyTorch. Install with: pip install torch")

    # Keep LCV reproducible across repeated evaluations.
    np.random.seed(int(random_state))
    torch.manual_seed(int(random_state))

    # Determine columns to use
    if columns is None:
        columns = real.columns.intersection(synthetic.columns).tolist()

    cols = columns

    # Pre-process: Discretize numerics so the autoencoder can extract semantic boundaries
    real_logic = _adaptive_binning(real[cols], cols)
    syn_logic = _adaptive_binning(synthetic[cols], cols)

    real, synthetic = _canonicalize_code_columns(real_logic, syn_logic, cols)

    # One-hot encode
    real_encoded = pd.get_dummies(real[cols], drop_first=False, prefix_sep="__").astype(
        np.float32
    )
    syn_encoded = pd.get_dummies(
        synthetic[cols], drop_first=False, prefix_sep="__"
    ).astype(np.float32)

    # Align feature spaces
    all_features = sorted(set(real_encoded.columns) | set(syn_encoded.columns))
    real_encoded = real_encoded.reindex(columns=all_features, fill_value=0.0)
    syn_encoded = syn_encoded.reindex(columns=all_features, fill_value=0.0)

    real_encoded = real_encoded[all_features].values
    syn_encoded = syn_encoded[all_features].values

    feature_groups = _feature_groups_from_encoded_columns(all_features)

    if len(all_features) <= 1:
        row_penalties = np.zeros(len(syn_encoded), dtype=float)
        return {
            "lcv_score": 1.0,
            "row_penalties": row_penalties,
            "violation_rate": 0.0,
            "mean_penalty": 0.0,
            "num_violations": 0,
            "columns_used": cols,
        }

    # Keep inputs in float32 to match model parameter dtype in PyTorch.
    real_tensor = torch.tensor(real_encoded, dtype=torch.float32)
    syn_tensor = torch.tensor(syn_encoded, dtype=torch.float32)

    # Train and evaluate
    input_dim = real_tensor.shape[1]
    hidden_dim = max(1, int(input_dim * 0.5))  # Under-complete bottleneck

    model = LCVAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        feature_groups=feature_groups,
    )
    model.fit(
        real_tensor,
        epochs=epochs,
        feature_groups=feature_groups,
        verbose=verbose,
    )

    feature_weights = _feature_weight_vector(feature_groups, feature_weighting)

    with torch.no_grad():
        lcv_score, row_penalties = model.evaluate(
            syn_tensor,
            feature_groups=feature_groups,
            feature_weights=feature_weights,
        )

    # Compute violation metrics
    violation_threshold = 0.5
    num_violations = (row_penalties > violation_threshold).sum()
    violation_rate = float(num_violations / len(row_penalties))

    return {
        "lcv_score": round(float(lcv_score), 4),
        "row_penalties": row_penalties,
        "violation_rate": round(violation_rate, 4),
        "mean_penalty": round(float(row_penalties.mean()), 4),
        "num_violations": int(num_violations),
        "columns_used": cols,
    }


def mine_implication_rules(
    real: pd.DataFrame,
    columns: list[str],
    min_confidence: float = 0.98,
    min_support: float = 0.01,
    max_rules: int = 200,
    min_lift: float = 1.0,
    max_antecedents: int = 1,
) -> list[dict[str, Any]]:
    """Mine implication rules from real data."""
    n_rows = len(real)
    if n_rows == 0:
        return []

    min_support_count = max(1, int(np.ceil(min_support * n_rows)))
    rules: list[dict[str, Any]] = []
    cat = real[columns].copy().astype(str)
    consequent_counts: dict[str, pd.Series] = {
        col: cat[col].value_counts() for col in columns
    }

    max_k = max(1, min(max_antecedents, len(columns) - 1))
    for k in range(1, max_k + 1):
        for antecedent_cols in combinations(columns, k):
            antecedent_cols_list = list(antecedent_cols)
            antecedent_count_series = cat[antecedent_cols_list].value_counts()
            valid_antecedents = antecedent_count_series[
                antecedent_count_series >= min_support_count
            ]
            if valid_antecedents.empty:
                continue

            consequent_candidates = [
                c for c in columns if c not in antecedent_cols_list
            ]
            if not consequent_candidates:
                continue

            for consequent_col in consequent_candidates:
                joint_count_series = cat[
                    antecedent_cols_list + [consequent_col]
                ].value_counts()
                if joint_count_series.empty:
                    continue

                for joint_key, pair_count in joint_count_series.items():
                    if not isinstance(joint_key, tuple):
                        joint_key = (joint_key,)

                    antecedent_key = tuple(joint_key[:k]) if k > 1 else joint_key[0]
                    if antecedent_key not in valid_antecedents.index:
                        continue

                    antecedent_count = int(valid_antecedents[antecedent_key])
                    consequent_value = str(joint_key[-1])
                    confidence = pair_count / max(antecedent_count, 1)
                    if confidence < min_confidence:
                        continue

                    consequent_support = int(
                        consequent_counts[consequent_col].get(consequent_value, 0)
                    ) / max(n_rows, 1)
                    if consequent_support <= 0:
                        continue

                    lift = confidence / consequent_support
                    if lift < min_lift:
                        continue

                    if k == 1:
                        ant_values = [str(antecedent_key)]
                    else:
                        ant_values = [str(v) for v in antecedent_key]

                    antecedents = [
                        {"feature": f, "value": v}
                        for f, v in zip(antecedent_cols_list, ant_values)
                    ]
                    antecedent_repr = _ANTE_JOIN.join(
                        f"{a['feature']}={a['value']}" for a in antecedents
                    )

                    rule = {
                        "antecedents": antecedents,
                        "antecedent_repr": antecedent_repr,
                        "consequent_feature": consequent_col,
                        "consequent_value": consequent_value,
                        "support": round(float(pair_count / n_rows), 4),
                        "confidence": round(float(confidence), 4),
                        "lift": round(float(lift), 4),
                        "support_count": int(pair_count),
                        "antecedent_count": antecedent_count,
                    }
                    if len(antecedents) == 1:
                        rule["antecedent_feature"] = antecedents[0]["feature"]
                        rule["antecedent_value"] = antecedents[0]["value"]
                    rules.append(rule)

    rules.sort(
        key=lambda r: (r["lift"], r["confidence"], r["support_count"]), reverse=True
    )
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

    # Discretize numerics for rule mining
    real_logic = _adaptive_binning(real_norm, columns)
    synthetic_logic = _adaptive_binning(synthetic_norm, columns)

    rules = mine_implication_rules(
        real_logic,
        columns=columns,
        min_confidence=min_confidence,
        min_support=min_support,
        max_rules=max_rules,
        min_lift=min_lift,
        max_antecedents=max_antecedents,
    )
    if not rules:
        return {
            "rule_violation_rate": 0.0,
            "num_rule_violations": 0,
            "num_rules_mined": 0,
            "rows_with_rule_violations": 0,
            "rows_evaluated": len(synthetic_logic),
            "top_violated_rules": [],
            "violation_examples": [],
        }

    row_violation_mask = np.zeros(len(synthetic_logic), dtype=bool)
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

        ant_mask = pd.Series(True, index=synthetic_logic.index)
        for ant in antecedents:
            ant_mask &= (
                synthetic_logic[ant["feature"]].astype(str).eq(str(ant["value"]))
            )
        if not ant_mask.any():
            continue

        violates = ant_mask & (~synthetic_logic[cons_col].astype(str).eq(cons_val))
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

            for row_index in synthetic_logic.index[violates][:3]:
                if len(violation_examples) >= max_violation_examples:
                    break
                actual_value = str(synthetic_logic.loc[row_index, cons_col])
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
    denom = max(len(synthetic_logic), 1)
    rule_diagnostics.sort(key=lambda d: d["violation_count"], reverse=True)
    return {
        "rule_violation_rate": round(rows_with_violations / denom, 4),
        "num_rule_violations": int(total_violations),
        "num_rules_mined": int(len(rules)),
        "rows_with_rule_violations": rows_with_violations,
        "rows_evaluated": int(len(synthetic_logic)),
        "example_rules": rules[:10],
        "top_violated_rules": rule_diagnostics[:10],
        "violation_examples": violation_examples,
    }
