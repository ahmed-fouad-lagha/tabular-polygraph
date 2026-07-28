from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabular_polygraph._config import (
    DEFAULT_ALPHA_BETA_MAX_ROWS,
    DEFAULT_ALPHA_BETA_MIN_ROWS,
    DEFAULT_ALPHA_BETA_N_STEPS,
)
from tabular_polygraph._types import Metric

from . import register


def _encode(real: pd.DataFrame, syn: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    num_cols = real.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = real.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            )
        )

    if not transformers:
        return np.array([]), np.array([])

    preprocessor = ColumnTransformer(transformers)
    real_t = preprocessor.fit_transform(real)
    syn_t = preprocessor.transform(syn)
    return real_t.astype(float), syn_t.astype(float)


@register
class AlphaBeta(Metric):
    name = "alpha_beta"

    def validate(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> str | None:
        if len(real) != len(synthetic):
            return f"Row count mismatch: real={len(real)}, synthetic={len(synthetic)}"
        if len(real) < DEFAULT_ALPHA_BETA_MIN_ROWS:
            return f"Too few rows: {len(real)} < {DEFAULT_ALPHA_BETA_MIN_ROWS}"
        return None

    def compute(
        self, real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]
    ) -> dict:
        n_min = min(len(real), DEFAULT_ALPHA_BETA_MAX_ROWS)
        if n_min < DEFAULT_ALPHA_BETA_MIN_ROWS:
            return {"alpha_precision": None, "beta_recall": None, "authenticity": None}

        real_s = real.sample(n_min, random_state=42).reset_index(drop=True)
        syn_s = synthetic.sample(n_min, random_state=42).reset_index(drop=True)

        X, X_syn = _encode(real_s, syn_s)
        n = len(X)
        if n < DEFAULT_ALPHA_BETA_MIN_ROWS or len(X_syn) != n:
            return {"alpha_precision": None, "beta_recall": None, "authenticity": None}

        emb_center = np.mean(X, axis=0)
        alphas = np.linspace(0, 1, DEFAULT_ALPHA_BETA_N_STEPS)
        Radii = np.quantile(np.sqrt(np.sum((X - emb_center) ** 2, axis=1)), alphas)
        synth_to_center = np.sqrt(np.sum((X_syn - emb_center) ** 2, axis=1))

        nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=None, p=2).fit(X)
        real_to_real = nbrs_real.kneighbors(X)[0][:, 1].reshape(-1)

        nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=None, p=2).fit(X_syn)
        real_synth_idx = nbrs_synth.kneighbors(X, return_distance=False).reshape(-1)
        real_synth_closest = X_syn[real_synth_idx]
        real_synth_closest_d = np.sqrt(
            np.sum((real_synth_closest - emb_center) ** 2, axis=1)
        )
        real_to_synth_d = np.sqrt(np.sum((real_synth_closest - X) ** 2, axis=1))

        precision_curve = [(synth_to_center <= r).mean() for r in Radii]
        coverage_curve = [(real_synth_closest_d <= r).mean() for r in Radii]

        denom = np.sum(alphas) or 1.0
        alpha_precision = max(
            0, 1 - np.sum(np.abs(alphas - np.array(precision_curve))) / denom
        )
        beta_recall = max(
            0, 1 - np.sum(np.abs(alphas - np.array(coverage_curve))) / denom
        )
        authenticity = float((real_to_synth_d < real_to_real).mean())

        return {
            "alpha_precision": float(alpha_precision),
            "beta_recall": float(beta_recall),
            "authenticity": authenticity,
        }
