"""
Shared utilities and helpers used across fidelity and generators modules.

Provides:
  - Column filtering (numeric, categorical, shared)
  - Data transformation (to_numeric_array, normalization)
  - Numerical stability helpers
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

# ============================================================================
# Global Constants
# ============================================================================

# DEFAULT_DROP_LIST: Truly generic identifier columns that should be
# automatically excluded from both generation and evaluation.
# Dataset-specific drops (e.g. zero-inflated columns) belong in
# the dataset metadata in loader.py, NOT here.
DEFAULT_DROP_LIST: set[str] = {
    "syn_id",
    "id",
    "row_id",
    "index",
    "uid",
    "uuid",
    "tract_id",
    "serial_no",
    "fips_code",
    "ip_address",
}


# ============================================================================
# Column Filtering
# ============================================================================


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return list of numeric column names in a DataFrame (excluding bool).

    Args:
        df: Input DataFrame.

    Returns:
        List of column names with numeric dtype (bool excluded).
    """
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return list of categorical/non-numeric column names in a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        List of column names with non-numeric dtype.
    """
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


# ============================================================================
# Data Transformation
# ============================================================================


def to_numeric_array(
    series: pd.Series,
    fill_method: str = "dropna",
    fill_value: float | None = None,
) -> np.ndarray:
    """Convert a pandas Series to a float numpy array with flexible filling.

    Args:
        series: Input Series to convert.
        fill_method: How to handle NaNs:
            - "dropna": Remove NaN values (default, reduces array length)
            - "mean": Fill with column mean
            - "median": Fill with column median
            - "zero": Fill with 0.0
            - "forward": Forward-fill then back-fill
            - "value": Fill with specified fill_value
        fill_value: Value to use if fill_method="value".

    Returns:
        1D numpy array of floats.

    Raises:
        ValueError: If fill_method is unrecognized or fill_value not provided
                    when fill_method="value".
    """
    s = series.copy()

    if fill_method == "dropna":
        return s.dropna().astype(float).values
    elif fill_method in ("mean", "median"):
        val = s.mean() if fill_method == "mean" else s.median()
        if pd.isna(val):
            import warnings

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


# ============================================================================
# Numerical Stability
# ============================================================================


def normalize(
    arr: np.ndarray,
    epsilon: float = 1e-9,
    return_params: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize array to mean=0, std=1 with numerical stability.

    Args:
        arr: Input array (1D or 2D).
        epsilon: Small constant added to std to avoid division by zero.
        return_params: If True, also return (mean, std) for applying to other arrays.

    Returns:
        Normalized array, or tuple of (normalized, mean, std) if return_params=True.
    """
    if arr.size == 0:
        empty_params = (
            np.array([])
            if arr.ndim <= 1
            else np.empty((0, arr.shape[1] if arr.ndim > 1 else 0))
        )
        if return_params:
            return arr.copy(), empty_params, empty_params
        return arr.copy()

    mu = np.mean(arr, axis=0)
    sigma = np.std(arr, axis=0)

    # Safety Switch: If a column has no variation, don't divide by epsilon.
    # We set it to 1.0 so we just subtract the mean and don't scale the noise.
    sigma = np.where(sigma < epsilon, 1.0, sigma)

    normalized = (arr - mu) / sigma

    if return_params:
        return normalized, mu, sigma
    return normalized


# ============================================================================
# Reproducibility
# ============================================================================


def set_seed(seed: int | None) -> None:
    """Set the random seed for reproducibility across multiple libraries.

    This helper centralizes seeding for random, numpy, and torch (if available),
    ensuring that generators and fidelity metrics produce deterministic results.

    Note:
        This function uses legacy ``np.random.seed()`` to set global state for
        backward compatibility. New code should prefer ``np.random.default_rng(seed)``
        for independent, thread-safe random generators.

    Args:
        seed: The seed value to use. If None, this function does nothing.
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
