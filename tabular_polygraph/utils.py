"""
Shared utilities and helpers used across calibration, fidelity, generators, and privacy modules.

Provides:
  - Column filtering (numeric, categorical, shared)
  - Data transformation (to_numeric_array, normalization)
  - Numerical stability helpers
"""

from __future__ import annotations

import random
from typing import TypeAlias

import numpy as np
import pandas as pd

# Type aliases
FloatArray: TypeAlias = np.ndarray  # 1D or 2D float array


# ============================================================================
# Global Constants
# ============================================================================

# DEFAULT_DROP_LIST: Columns that should be automatically excluded from
# both generation and evaluation (IDs, internal keys, high-cardinality noise).
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
    """Return list of numeric column names in a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        List of column names with numeric dtype.
    """
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return list of categorical/non-numeric column names in a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        List of column names with non-numeric dtype.
    """
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def shared_columns(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    exclude: list[str] | None = None,
) -> list[str]:
    """Return common column names between real and synthetic data.

    Args:
        real: Real data DataFrame.
        synthetic: Synthetic data DataFrame.
        exclude: Columns to exclude from result (e.g. ["syn_id"]).

    Returns:
        List of shared column names.
    """
    if exclude is None:
        exclude = ["syn_id"]
    shared = [c for c in real.columns if c in synthetic.columns]
    return [c for c in shared if c not in exclude]


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
    elif fill_method == "mean":
        s = s.fillna(s.mean())
    elif fill_method == "median":
        s = s.fillna(s.median())
    elif fill_method == "zero":
        s = s.fillna(0.0)
    elif fill_method == "forward":
        s = s.ffill().bfill()
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
    mu = np.mean(arr, axis=0)
    sigma = np.std(arr, axis=0)

    # Safety Switch: If a column has no variation, don't divide by epsilon.
    # We set it to 1.0 so we just subtract the mean and don't scale the noise.
    sigma = np.where(sigma < epsilon, 1.0, sigma)

    normalized = (arr - mu) / sigma

    if return_params:
        return normalized, mu, sigma
    return normalized


def denormalize(
    arr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Reverse normalization applied by normalize().

    Args:
        arr: Normalized array.
        mean: Mean from normalize(return_params=True).
        std: Std from normalize(return_params=True).

    Returns:
        Original-scale array.
    """
    return (arr * std) + mean


# ============================================================================
# Display Utilities (Low Priority - Optional CLI Use)
# ============================================================================


def format_score_bar(
    score: float,
    max_score: float = 100.0,
    width: int = 20,
) -> str:
    """Generate a Unicode progress bar for a score.

    Args:
        score: Numeric score value.
        max_score: Maximum possible score (default 100).
        width: Width of bar in characters (default 20).

    Returns:
        String with filled/unfilled blocks representing the score.

    Example:
        >>> format_score_bar(85, max_score=100, width=20)
        '████████████████░░░░'  (17 filled, 3 empty)
    """
    filled = int((score / max_score) * width)
    return "█" * filled + "░" * (width - filled)


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
