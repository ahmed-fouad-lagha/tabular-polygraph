from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def to_numeric_array(
    series: pd.Series,
    fill_method: str = "dropna",
    fill_value: float | None = None,
) -> np.ndarray:
    s = series.copy()

    if fill_method == "dropna":
        return s.dropna().astype(float).values
    elif fill_method in ("mean", "median"):
        val = s.mean() if fill_method == "mean" else s.median()
        if pd.isna(val):
            warnings.warn(
                f"to_numeric_array('{fill_method}') encountered an all-NaN series. Filling with 0.0 fallback.",
                UserWarning,
                stacklevel=2,
            )
            val = 0.0
        s = s.fillna(val)
    elif fill_method == "zero":
        s = s.fillna(0.0)
    elif fill_method == "forward":
        s = s.ffill().bfill()
        if s.isna().any():
            s = s.fillna(0.0)
    elif fill_method == "value":
        if fill_value is None:
            raise ValueError("fill_value required when fill_method='value'")
        s = s.fillna(fill_value)
    else:
        raise ValueError(
            f"Unknown fill_method: {fill_method}. "
            f"Choose from: dropna, mean, median, zero, forward, value"
        )

    return s.astype(float).values


def normalize(
    arr: np.ndarray,
    epsilon: float = 1e-9,
    return_params: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arr.size == 0:
        empty_params = np.array([]) if arr.ndim <= 1 else np.full(arr.shape[1], np.nan)
        if return_params:
            return arr.copy(), empty_params, empty_params
        return arr.copy()

    mu = np.mean(arr, axis=0)
    sigma = np.std(arr, axis=0)
    sigma = np.where(sigma < epsilon, 1.0, sigma)

    normalized = (arr - mu) / sigma

    if return_params:
        return normalized, mu, sigma
    return normalized
