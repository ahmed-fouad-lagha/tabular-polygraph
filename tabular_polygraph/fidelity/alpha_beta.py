"""
Alpha-precision and beta-recall (Alaa et al., ICML 2022).

Measures how well synthetic data covers the real data distribution
(alpha-precision) and how much of the real distribution is captured
by the synthetic data (beta-recall). Also reports authenticity: the
fraction of real samples whose nearest synthetic neighbor is closer
than their nearest real neighbor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _encode_for_alpha_precision(
    real: pd.DataFrame, syn: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess data for distance metrics: impute, one-hot encode, and standard scale."""
    num_cols = real.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = real.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if num_cols:
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        return np.array([]), np.array([])

    preprocessor = ColumnTransformer(transformers)

    # Fit on real data, transform both
    real_transformed = preprocessor.fit_transform(real)
    syn_transformed = preprocessor.transform(syn)

    return real_transformed.astype(float), syn_transformed.astype(float)


def alpha_precision_beta_recall(
    real: pd.DataFrame, syn: pd.DataFrame, n_steps: int = 30
) -> dict:
    """Compute alpha-precision and beta-recall (Alaa et al., ICML 2022).

    Parameters
    ----------
    real : pd.DataFrame
        Real data (must have same number of rows as *syn*).
    syn : pd.DataFrame
        Synthetic data (same row count and columns as *real*).
    n_steps : int
        Number of quantile thresholds for the precision/recall curves.

    Returns
    -------
    dict with keys:
        alpha_precision : float
            Fraction of synthetic samples inside each real quantile ball.
        beta_recall : float
            Fraction of real samples whose nearest synthetic neighbor
            falls within the corresponding quantile.
        authenticity : float
            Fraction of real samples whose nearest synthetic neighbor
            is closer than their nearest real neighbor.

    Raises
    ------
    ValueError
        If real and synthetic have different numbers of rows.
    """
    X, X_syn = _encode_for_alpha_precision(real, syn)
    n = len(X)
    if len(X_syn) != n:
        raise ValueError(
            f"Real ({len(X)}) and synthetic ({len(X_syn)}) must have the same "
            "number of rows for alpha-precision/beta-recall computation."
        )

    emb_center = np.mean(X, axis=0)

    alphas = np.linspace(0, 1, n_steps)

    # Compute quantile radii from real-to-center distances (the paper's definition)
    Radii = np.quantile(np.sqrt(np.sum((X - emb_center) ** 2, axis=1)), alphas)
    synth_to_center = np.sqrt(np.sum((X_syn - emb_center) ** 2, axis=1))

    # Nearest real neighbor (for authenticity)
    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(X)
    real_to_real, _ = nbrs_real.kneighbors(X)
    real_to_real = real_to_real[:, 1].reshape(-1)

    # Nearest synthetic neighbor (for beta-recall + authenticity)
    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_syn)
    real_to_synth_args = nbrs_synth.kneighbors(X, return_distance=False)
    real_synth_closest = X_syn[real_to_synth_args.reshape(-1)]
    # Distance from each real point's nearest synth neighbor to the real center
    # (measures whether the synth neighbor falls within the real distribution's ball)
    real_synth_closest_d = np.sqrt(
        np.sum((real_synth_closest - emb_center) ** 2, axis=1)
    )
    real_to_synth_d = np.sqrt(np.sum((real_synth_closest - X) ** 2, axis=1))

    alpha_precision_curve = []
    beta_coverage_curve = []

    for k in range(len(Radii)):
        precision_audit_mask = synth_to_center <= Radii[k]
        alpha_precision_curve.append(np.mean(precision_audit_mask))

        # Beta-recall: fraction of real samples whose nearest synthetic
        # neighbor falls within the α-quantile ball of the real distribution
        beta_coverage_curve.append(np.mean(real_synth_closest_d <= Radii[k]))

    denom: float = float(np.sum(alphas))
    if denom < 1e-9:
        denom = 1.0

    delta_precision = (
        1 - np.sum(np.abs(alphas - np.array(alpha_precision_curve))) / denom
    )
    delta_coverage = 1 - np.sum(np.abs(alphas - np.array(beta_coverage_curve))) / denom

    # Authenticity: nearest synthetic neighbor is closer than nearest real neighbor
    authen = real_to_synth_d < real_to_real
    authenticity = np.mean(authen)

    return {
        "alpha_precision": float(max(0, delta_precision)),
        "beta_recall": float(max(0, delta_coverage)),
        "authenticity": float(authenticity),
    }
