"""
Neuro-LCV: Neurosymbolic Logical Constraint Validator for Synthetic Tabular Data.

Trains an under-complete autoencoder on real data to extract semantic boundaries,
and evaluates synthetic data using a Continuous Semantic Severity Penalty (CSSP).

Mathematical Foundation
-----------------------
The frozen autoencoder approximates P(x_i = c | x_{-i}) for each feature.
CSSP(x_g,i) = 1 - P(category_chosen | others) measures logical impossibility.
Neuro-LCV Score ∈ [0, 1] is the mean row probability across all synthetic data.
"""
from __future__ import annotations
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
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.BCELoss()
        self.is_trained = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, _ in dataloader:
                # Denoising: add noise to inputs to improve robustness
                noise = torch.randn_like(batch_x) * 0.05
                noisy_x = torch.clamp(batch_x + noise, 0., 1.)
                
                self.optimizer.zero_grad()
                outputs = self.forward(noisy_x)
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
            # Get expected probabilities from frozen oracle
            expected_probs = self.forward(synth_tensor)
            
            # CSSP: isolate probabilities for the categories the generator chose
            chosen_probs = expected_probs * synth_tensor
            
            # Average probability per row
            row_probs_sum = chosen_probs.sum(dim=1)
            num_features = synth_tensor.sum(dim=1).clamp(min=1e-9)
            avg_row_prob = row_probs_sum / num_features
            
            # Row penalties: 1 - probability (high penalty = illogical)
            row_penalties = 1.0 - avg_row_prob
            
            # Overall Neuro-LCV Score: 1 is perfect, 0 is entirely impossible
            lcv_score = 1.0 - row_penalties.mean().item()
        
        return lcv_score, row_penalties.numpy()


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
        raise ValueError(f"Requested columns not found in both DataFrames.")
    
    # One-hot encode
    real_encoded = pd.get_dummies(real[cols], drop_first=False).astype(np.float32)
    syn_encoded = pd.get_dummies(synthetic[cols], drop_first=False).astype(np.float32)
    
    # Align feature spaces
    all_features = sorted(set(real_encoded.columns) | set(syn_encoded.columns))
    for feat in all_features:
        if feat not in real_encoded.columns:
            real_encoded[feat] = 0.0
        if feat not in syn_encoded.columns:
            syn_encoded[feat] = 0.0
    
    real_encoded = real_encoded[all_features].values
    syn_encoded = syn_encoded[all_features].values
    
    real_tensor = torch.tensor(real_encoded)
    syn_tensor = torch.tensor(syn_encoded)
    
    # Train and evaluate
    input_dim = real_tensor.shape[1]
    hidden_dim = max(1, int(input_dim * 0.5))  # Under-complete bottleneck
    
    model = NeuroLCVAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    model.fit(real_tensor, epochs=epochs, verbose=verbose)
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
