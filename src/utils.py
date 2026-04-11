"""
src.utils
---------
Shared utilities and helpers used across calibration, fidelity, generators, and privacy modules.

Provides:
  - Column filtering (numeric, categorical, shared)
  - Data transformation (to_numeric_array, normalization)
  - Numerical stability helpers
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import TypeAlias

# Type aliases
FloatArray: TypeAlias = np.ndarray  # 1D or 2D float array


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
        s = s.fillna(method="ffill").fillna(method="bfill")  # type: ignore
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
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0) + epsilon
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
