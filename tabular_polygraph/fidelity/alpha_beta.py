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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


def _encode_for_alpha_precision(
    real: pd.DataFrame, syn: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Encode categorical columns to integers for numeric metric computation."""
    real_enc = real.copy()
    syn_enc = syn.copy()
    cat_cols = real_enc.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([real_enc[col], syn_enc[col]], ignore_index=True).astype(
            str
        )
        le.fit(combined)
        real_enc[col] = le.transform(real_enc[col].astype(str))
        syn_enc[col] = le.transform(syn_enc[col].astype(str))
    return real_enc.values.astype(float), syn_enc.values.astype(float)


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
    synth_center = np.mean(X_syn, axis=0)

    alphas = np.linspace(0, 1, n_steps)

    Radii = np.quantile(np.sqrt(np.sum((X - emb_center) ** 2, axis=1)), alphas)
    synth_to_center = np.sqrt(np.sum((X_syn - emb_center) ** 2, axis=1))

    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(X)
    real_to_real, _ = nbrs_real.kneighbors(X)
    real_to_real = real_to_real[:, 1].squeeze()

    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_syn)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(X)

    real_synth_closest = X_syn[real_to_synth_args.squeeze()]
    real_synth_closest_d = np.sqrt(
        np.sum((real_synth_closest - synth_center) ** 2, axis=1)
    )
    closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)

    alpha_precision_curve = []
    beta_coverage_curve = []

    for k in range(len(Radii)):
        precision_audit_mask = synth_to_center <= Radii[k]
        alpha_precision_curve.append(np.mean(precision_audit_mask))

        beta_coverage_curve.append(
            np.mean(
                (real_to_synth <= real_to_real)
                * (real_synth_closest_d <= closest_synth_Radii[k])
            )
        )

    delta_precision = 1 - np.sum(
        np.abs(alphas - np.array(alpha_precision_curve))
    ) / np.sum(alphas)
    delta_coverage = 1 - np.sum(
        np.abs(alphas - np.array(beta_coverage_curve))
    ) / np.sum(alphas)

    authen = real_to_real[real_to_synth_args.squeeze()] < real_to_synth.squeeze()
    authenticity = np.mean(authen)

    return {
        "alpha_precision": float(max(0, delta_precision)),
        "beta_recall": float(max(0, delta_coverage)),
        "authenticity": float(authenticity),
    }
