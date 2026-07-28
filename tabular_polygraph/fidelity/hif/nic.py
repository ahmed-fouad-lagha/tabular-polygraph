"""
tabular_polygraph.fidelity.hif.nic
-------------------------------
Neighbor-Invariant Continuity (NIC) auditor of HIF.

Audits continuous features against the categorical manifold using
non-linear regression.  Large reconstruction residuals signal
physically implausible numeric relations in synthetic data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from tabular_polygraph._config import (
    DEFAULT_NIC_COLLAPSE_PENALTY,
    DEFAULT_NIC_COLLAPSE_THRESHOLD,
    DEFAULT_NIC_GAMMA_PERCENTILE,
    DEFAULT_NIC_Z_PERCENTILE,
)

from .sentinel import ManifoldEncoder

logger = logging.getLogger(__name__)

NIC_COLLAPSE_THRESHOLD = DEFAULT_NIC_COLLAPSE_THRESHOLD
NIC_COLLAPSE_PENALTY = DEFAULT_NIC_COLLAPSE_PENALTY
NIC_Z_PERCENTILE = DEFAULT_NIC_Z_PERCENTILE
NIC_GAMMA_PERCENTILE = DEFAULT_NIC_GAMMA_PERCENTILE

__all__ = [
    "NIC_COLLAPSE_THRESHOLD",
    "NIC_COLLAPSE_PENALTY",
    "NIC_Z_PERCENTILE",
    "NIC_GAMMA_PERCENTILE",
    "NeighborInvariantContinuity",
]


class NeighborInvariantContinuity:
    """
    Neighbor-Invariant Continuity (NIC).
    Audits continuous features against categorical manifold using non-linear reconstruction.
    """

    def __init__(self, random_state: int = 42):
        self.regressors: dict[str, HistGradientBoostingRegressor] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.z_thresholds: dict[str, float] = {}
        self.gamma_scalings: dict[str, float] = {}
        self.marginal_references: dict[str, np.ndarray] = {}
        self.pca: TruncatedSVD | None = None
        self.latent_scaler = StandardScaler(with_mean=False)
        self.encoder = ManifoldEncoder()
        self.random_state = random_state

    def _continuous_columns(self, continuous_df: pd.DataFrame) -> list[str]:
        return [c for c in continuous_df.columns if continuous_df[c].nunique() > 1]

    def _encoded_categorical(
        self,
        categorical_df: pd.DataFrame,
        x_precomputed: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if x_precomputed is not None:
            return x_precomputed

        self.encoder.fit(categorical_df)
        return self.encoder.transform(categorical_df)

    def _fit_latent_projection(
        self, x_encoded: pd.DataFrame, verbose: bool
    ) -> tuple[np.ndarray | None, int]:
        n_feat = x_encoded.shape[1]
        n_samples = x_encoded.shape[0]
        if n_feat < 1 or n_samples < 2:
            return None, 0

        n_comp = max(1, min(n_samples // 2, n_feat, 32))
        if verbose:
            logger.debug(f"Spectral Embedding ({n_feat} -> {n_comp} target)...")

        self.pca = TruncatedSVD(
            n_components=n_comp,
            algorithm="randomized",
            random_state=self.random_state,
        )
        x_scaled = self.latent_scaler.fit_transform(x_encoded)
        latent = self.pca.fit_transform(x_scaled)
        if verbose:
            logger.debug(f"Spectral Embedding done ({n_comp} components).")

        return latent, n_comp

    def _fit_continuous_column(
        self,
        column_name: str,
        column_values: pd.Series,
        latent: np.ndarray,
        verbose: bool,
    ) -> None:
        if verbose:
            logger.debug(f"Regressing variable '{column_name}'...")

        y_raw = np.asarray(column_values, dtype=float)
        valid_mask = ~np.isnan(y_raw)
        if not valid_mask.any():
            if verbose:
                logger.debug(f"Variable '{column_name}' skipped (all NaN).")
            return

        y_valid = y_raw[valid_mask]
        self.marginal_references[column_name] = np.sort(y_valid)
        latent_valid = latent[valid_mask]

        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y_valid.reshape(-1, 1)).ravel()

        reg = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state,
            l2_regularization=1.0,
        )
        reg.fit(latent_valid, y_scaled)
        if verbose:
            logger.debug(f"Regressing variable '{column_name}' complete.")

        y_pred = reg.predict(latent_valid)
        residuals = np.abs(y_scaled - y_pred)

        self.regressors[column_name] = reg
        self.scalers[column_name] = scaler

        mad = float(median_abs_deviation(residuals))
        med = float(np.median(residuals))
        p98 = float(np.percentile(residuals, 98))

        self.z_thresholds[column_name] = max(p98, med + 2.0 * mad)
        self.gamma_scalings[column_name] = max(
            2.0 * self.z_thresholds[column_name], 3.0 * mad, 0.1
        )

    def fit(
        self,
        categorical_df: pd.DataFrame,
        continuous_df: pd.DataFrame,
        x_precomputed: pd.DataFrame | None = None,
        verbose: bool = True,
    ) -> None:
        """Fit non-linear regressors on the training manifold."""
        valid_cols = self._continuous_columns(continuous_df)
        if not valid_cols:
            return

        active_df = continuous_df[valid_cols]
        x_encoded = self._encoded_categorical(categorical_df, x_precomputed)
        latent, _ = self._fit_latent_projection(x_encoded, verbose)
        if latent is None:
            return

        self.regressors = {}
        for col in active_df.columns:
            self._fit_continuous_column(col, active_df[col], latent, verbose)

    def score(
        self,
        categorical_df: pd.DataFrame,
        continuous_df: pd.DataFrame,
        x_precomputed: pd.DataFrame | None = None,
    ) -> tuple[float, np.ndarray]:
        """Score continuous features for manifold continuity violations."""
        if self.pca is None or not self.regressors:
            raise ValueError(
                "NeighborInvariantContinuity must be fitted before score()."
            )

        if x_precomputed is not None:
            x_aligned = x_precomputed
        else:
            x_aligned = self.encoder.transform(categorical_df)

        x_scaled = self.latent_scaler.transform(x_aligned)
        latent = self.pca.transform(x_scaled)
        row_penalties = np.zeros(len(continuous_df))

        for col in continuous_df.columns:
            if col not in self.regressors:
                continue

            y = continuous_df[col].values
            valid_mask = ~np.isnan(y)
            if not valid_mask.any():
                continue

            y_valid = y[valid_mask]
            latent_valid = latent[valid_mask]

            y_scaled = self.scalers[col].transform(y_valid.reshape(-1, 1)).flatten()
            y_pred = self.regressors[col].predict(latent_valid)
            residuals = np.abs(y_scaled - y_pred)

            threshold = self.z_thresholds[col]
            gamma = self.gamma_scalings[col]
            col_penalty = np.zeros(len(y))
            if threshold > 0:
                col_penalty[valid_mask] = np.clip((residuals - threshold) / gamma, 0, 1)

            row_penalties = np.maximum(row_penalties, col_penalty)

        return float(1.0 - row_penalties.mean()), row_penalties
