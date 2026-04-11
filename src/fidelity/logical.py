"""
LCV: Neurosymbolic Logical Constraint Validator for Synthetic Tabular Data.

Trains an under-complete autoencoder on real data to extract semantic boundaries,
and evaluates synthetic data using a Continuous Semantic Severity Penalty (CSSP).

Mathematical Foundation
-----------------------
The frozen autoencoder approximates P(x_i = c | x_{-i}) for each feature.
CSSP(x_g,i) = 1 - P(category_chosen | others) measures logical impossibility.
LCV Score ∈ [0, 1] is the geometric mean of the per-feature chosen-category
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
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _feature_groups_from_encoded_columns(
    encoded_columns: list[str], separator: str = "__"
) -> list[list[int]]:
    """Group one-hot encoded columns by their original feature prefix."""
    feature_to_indices: dict[str, list[int]] = {}
    for index, column_name in enumerate(encoded_columns):
        feature_name = column_name.split(separator, 1)[0]
        feature_to_indices.setdefault(feature_name, []).append(index)
    return [feature_to_indices[feature] for feature in sorted(feature_to_indices)]


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

    def __init__(self, input_dim: int, hidden_dim: int, learning_rate: float = 0.005):
        super(LCVAutoencoder, self).__init__()
        assert hidden_dim < input_dim, (
            "Hidden dimension must compress input for under-complete design."
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # For categorical tabular data, dropout-denoising is more stable than Gaussian perturbation.
        self.input_dropout = nn.Dropout(p=0.1)

        dim_1 = min(256, max(64, input_dim * 2))
        dim_2 = min(128, max(32, input_dim))

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

        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.criterion = nn.BCELoss()
        self.is_trained = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self.input_dropout(x)
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def fit(
        self,
        real_tensor: torch.Tensor,
        epochs: int = 30,
        batch_size: int = 128,
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
        verbose : bool
            Print training progress
        """
        if verbose:
            print(f"[LCV] Training on {len(real_tensor)} real records...")

        self.train()
        dataset = torch.utils.data.TensorDataset(real_tensor, real_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, _ in dataloader:
                self.optimizer.zero_grad()
                outputs = self.forward(batch_x)
                loss = self.criterion(outputs, batch_x)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if verbose and (epoch + 1) % max(1, epochs // 3) == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(
                    f"[LCV] Epoch [{epoch + 1}/{epochs}], Reconstruction Loss: {avg_loss:.4f}"
                )

        self.is_trained = True
        if verbose:
            print("[LCV] Semantic extraction complete.")

    def evaluate(
        self,
        synth_tensor: torch.Tensor,
        feature_groups: list[list[int]] | None = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Phase 2: Grade synthetic data using frozen oracle probabilities in O(1) per row.

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
            expected_probs = self.forward(synth_tensor)

            if feature_groups:
                feature_scores = []
                for indices in feature_groups:
                    selected_probs = (
                        expected_probs[:, indices] * synth_tensor[:, indices]
                    ).sum(dim=1)
                    feature_scores.append(selected_probs)

                feature_matrix = torch.stack(feature_scores, dim=1).clamp_min(1e-6)
                per_row_score = torch.exp(torch.log(feature_matrix).mean(dim=1))
                row_penalties = (1.0 - per_row_score).cpu().numpy()
                lcv_score = float(per_row_score.mean().item())
                return lcv_score, row_penalties

            # Backward-compatible fallback: use the stricter minimum-over-hot-bits score.
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

    # Determine columns to use
    if columns is None:
        # Use categorical columns present in both
        real_cat = real.select_dtypes(include=["object", "category"]).columns.tolist()
        syn_cat = synthetic.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        columns = [c for c in real_cat if c in syn_cat]
        if not columns:
            raise ValueError(
                "No categorical columns found in both real and synthetic data."
            )

    cols = [c for c in columns if c in real.columns and c in synthetic.columns]
    if not cols:
        raise ValueError("Requested columns not found in both DataFrames.")

    real, synthetic = _canonicalize_code_columns(real, synthetic, cols)

    # One-hot encode
    real_encoded = pd.get_dummies(real[cols], drop_first=False, prefix_sep="__").astype(
        np.float32
    )
    syn_encoded = pd.get_dummies(
        synthetic[cols], drop_first=False, prefix_sep="__"
    ).astype(np.float32)

    # Align feature spaces
    all_features = sorted(set(real_encoded.columns) | set(syn_encoded.columns))
    for feat in all_features:
        if feat not in real_encoded.columns:
            real_encoded[feat] = 0.0
        if feat not in syn_encoded.columns:
            syn_encoded[feat] = 0.0

    real_encoded = real_encoded[all_features].values
    syn_encoded = syn_encoded[all_features].values

    feature_groups = _feature_groups_from_encoded_columns(all_features)

    # Keep inputs in float32 to match model parameter dtype in PyTorch.
    real_tensor = torch.tensor(real_encoded, dtype=torch.float32)
    syn_tensor = torch.tensor(syn_encoded, dtype=torch.float32)

    # Train and evaluate
    input_dim = real_tensor.shape[1]
    hidden_dim = max(1, int(input_dim * 0.5))  # Under-complete bottleneck

    model = LCVAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    model.fit(real_tensor, epochs=epochs, verbose=verbose)

    with torch.no_grad():
        lcv_score, row_penalties = model.evaluate(
            syn_tensor, feature_groups=feature_groups
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
    """Mine implication rules from real categorical data with optional multi-antecedents."""
    if not columns:
        return []
    if max_rules < 1:
        return []
    if max_antecedents < 1:
        return []
    if not (0.0 <= min_support <= 1.0):
        return []
    if not (0.0 <= min_confidence <= 1.0):
        return []
    if min_lift < 0.0:
        return []

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
                    antecedent_repr = " AND ".join(
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
    rules = mine_implication_rules(
        real_norm,
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
            "rows_evaluated": len(synthetic_norm),
            "top_violated_rules": [],
            "violation_examples": [],
        }

    row_violation_mask = np.zeros(len(synthetic_norm), dtype=bool)
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

        ant_mask = pd.Series(True, index=synthetic_norm.index)
        for ant in antecedents:
            ant_mask &= synthetic_norm[ant["feature"]].astype(str).eq(str(ant["value"]))
        if not ant_mask.any():
            continue

        violates = ant_mask & (~synthetic_norm[cons_col].astype(str).eq(cons_val))
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
                        " AND ".join(
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

            for row_index in synthetic_norm.index[violates][:3]:
                if len(violation_examples) >= max_violation_examples:
                    break
                actual_value = str(synthetic_norm.loc[row_index, cons_col])
                violation_examples.append(
                    {
                        "row_index": int(row_index)
                        if isinstance(row_index, (int, np.integer))
                        else str(row_index),
                        "antecedent": rule.get(
                            "antecedent_repr",
                            " AND ".join(
                                f"{a['feature']}={a['value']}" for a in antecedents
                            ),
                        ),
                        "expected": f"{cons_col}={cons_val}",
                        "actual": f"{cons_col}={actual_value}",
                    }
                )

    rows_with_violations = int(row_violation_mask.sum())
    denom = max(len(synthetic_norm), 1)
    rule_diagnostics.sort(key=lambda d: d["violation_count"], reverse=True)
    return {
        "rule_violation_rate": round(rows_with_violations / denom, 4),
        "num_rule_violations": int(total_violations),
        "num_rules_mined": int(len(rules)),
        "rows_with_rule_violations": rows_with_violations,
        "rows_evaluated": int(len(synthetic_norm)),
        "example_rules": rules[:10],
        "top_violated_rules": rule_diagnostics[:10],
        "violation_examples": violation_examples,
    }
