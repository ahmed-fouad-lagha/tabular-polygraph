"""
Neuro-LCV: Neurosymbolic Logical Constraint Validator for Synthetic Tabular Data.

Trains an under-complete autoencoder on real data to extract semantic boundaries,
and evaluates synthetic data using a Continuous Semantic Severity Penalty (CSSP).

Mathematical Foundation
-----------------------
The frozen autoencoder approximates P(x_i = c | x_{-i}) for each feature.
CSSP(x_g,i) = 1 - P(category_chosen | others) measures logical impossibility.
Neuro-LCV Score ∈ [0, 1] is the geometric mean of the per-feature chosen-category
probabilities, averaged across synthetic rows.
"""
from __future__ import annotations
import time
import sys
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from typing import Tuple

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _feature_groups_from_encoded_columns(encoded_columns: list[str], separator: str = "__") -> list[list[int]]:
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


class NeuroLCVAutoencoder(nn.Module):
    """
    Under-complete denoising autoencoder trained on categorical tabular data.
    Serves as the frozen "Laws of Physics" oracle for semantic validation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, learning_rate: float = 0.005):
        super(NeuroLCVAutoencoder, self).__init__()
        assert hidden_dim < input_dim, "Hidden dimension must compress input for under-complete design."
        
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
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, dim_2),
            nn.ReLU(),
            nn.Linear(dim_2, dim_1),
            nn.ReLU(),
            nn.Linear(dim_1, input_dim),
            nn.Sigmoid()
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.BCELoss()
        self.is_trained = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self.input_dropout(x)
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def fit(self, real_tensor: torch.Tensor, epochs: int = 30, batch_size: int = 128, verbose: bool = True):
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
            print(f"[Neuro-LCV] Training on {len(real_tensor)} real records...")
        
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
                print(f"[Neuro-LCV] Epoch [{epoch+1}/{epochs}], Reconstruction Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        if verbose:
            print("[Neuro-LCV] Semantic extraction complete.")

    def evaluate(self, synth_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        Phase 2: Grade synthetic data using frozen oracle probabilities in O(1) per row.
        
        Parameters
        ----------
        synth_tensor : torch.Tensor
            One-hot encoded synthetic data, shape (n_samples, n_features)
        
        Returns
        -------
        lcv_score : float
            Neuro-LCV fidelity ∈ [0, 1]. Higher is better (1.0 = perfect logical alignment).
        row_penalties : np.ndarray
            Per-row Continuous Semantic Severity Penalty ∈ [0, 1].
        """
        if not self.is_trained:
            raise RuntimeError("NeuroLCVAutoencoder must be fitted first.")
        
        self.eval()
        with torch.no_grad():
            expected_probs = self.forward(synth_tensor)
            feature_groups = getattr(self, "feature_groups", None)

            if feature_groups:
                feature_scores = []
                for indices in feature_groups:
                    selected_probs = (expected_probs[:, indices] * synth_tensor[:, indices]).sum(dim=1)
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
            row_penalties = 1.0 - min_row_prob
            lcv_score = 1.0 - row_penalties.mean().item()
            return lcv_score, row_penalties.cpu().numpy()


def neuro_lcv_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
    epochs: int = 30,
    verbose: bool = True,
) -> dict:
    """
    Compute Neuro-LCV logical constraint validation score.
    
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
        - neuro_lcv_score : float ∈ [0, 1]
        - row_penalties : np.ndarray
        - violation_rate : float (fraction of rows with penalty > 0.5)
        - mean_penalty : float
        - columns_used : list[str]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("Neuro-LCV requires PyTorch. Install with: pip install torch")
    
    # Determine columns to use
    if columns is None:
        # Use categorical columns present in both
        real_cat = real.select_dtypes(include=['object', 'category']).columns.tolist()
        syn_cat = synthetic.select_dtypes(include=['object', 'category']).columns.tolist()
        columns = [c for c in real_cat if c in syn_cat]
        if not columns:
            raise ValueError("No categorical columns found in both real and synthetic data.")
    
    cols = [c for c in columns if c in real.columns and c in synthetic.columns]
    if not cols:
        raise ValueError("Requested columns not found in both DataFrames.")

    real, synthetic = _canonicalize_code_columns(real, synthetic, cols)
    
    # One-hot encode
    real_encoded = pd.get_dummies(real[cols], drop_first=False, prefix_sep="__").astype(np.float32)
    syn_encoded = pd.get_dummies(synthetic[cols], drop_first=False, prefix_sep="__").astype(np.float32)
    
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
    
    model = NeuroLCVAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    model.feature_groups = feature_groups
    model.fit(real_tensor, epochs=epochs, verbose=verbose)

    with torch.no_grad():
        lcv_score, row_penalties = model.evaluate(syn_tensor)
    
    # Compute violation metrics
    violation_threshold = 0.5
    num_violations = (row_penalties > violation_threshold).sum()
    violation_rate = float(num_violations / len(row_penalties))
    
    return {
        "neuro_lcv_score": round(float(lcv_score), 4),
        "row_penalties": row_penalties,
        "violation_rate": round(violation_rate, 4),
        "mean_penalty": round(float(row_penalties.mean()), 4),
        "num_violations": int(num_violations),
        "columns_used": cols,
    }
