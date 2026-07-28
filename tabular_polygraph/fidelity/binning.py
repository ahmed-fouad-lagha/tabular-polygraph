"""
tabular_polygraph.fidelity.binning
-----------------------------------
Adaptive binning and code canonicalisation for HIF rule mining.

Bin edges are learned from REAL data only and then applied identically to
both real and synthetic DataFrames to ensure fair comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph._config import DEFAULT_RULE_QUANTIZATION_BINS

RULE_QUANTIZATION_BINS = DEFAULT_RULE_QUANTIZATION_BINS

__all__ = [
    "RULE_QUANTIZATION_BINS",
    "fit_binning",
    "apply_binning",
    "canonicalize_code_columns",
]


def fit_binning(
    df: pd.DataFrame, columns: list[str], n_bins: int = RULE_QUANTIZATION_BINS
) -> dict[str, np.ndarray | None]:
    """Learn bin edges from REAL data only.

    Returns a dict mapping column name to its bin edges (for pd.cut) or None
    if the column is already discrete (<= n_bins unique values).
    """
    edges: dict[str, np.ndarray | None] = {}
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() <= 1:
            edges[col] = None
            continue
        if df[col].nunique() <= n_bins:
            edges[col] = None
            continue
        try:
            _, bin_edges = pd.qcut(
                df[col], q=n_bins, labels=False, duplicates="drop", retbins=True
            )
            edges[col] = bin_edges
        except (ValueError, TypeError):
            try:
                _, bin_edges = pd.cut(df[col], bins=n_bins, labels=False, retbins=True)
                edges[col] = bin_edges
            except (ValueError, TypeError):
                edges[col] = None
    return edges


def apply_binning(
    df: pd.DataFrame,
    columns: list[str],
    edges: dict[str, np.ndarray | None],
) -> pd.DataFrame:
    """Apply pre-fitted bin edges to any DataFrame (real or synthetic).

    Columns with None edges are either categorical (kept as-is) or have
    <= n_bins unique values (labeled as ``bin_<value>``).
    """
    df_binned = df.copy()
    for col in columns:
        if col not in edges or edges[col] is None:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= 1:
                    df_binned[col] = df[col].apply(
                        lambda v: "bin_0" if pd.notna(v) else v
                    )
                else:
                    df_binned[col] = df[col].apply(
                        lambda v: f"bin_{v}" if pd.notna(v) else v
                    )
            else:
                df_binned[col] = df[col].apply(lambda v: v if pd.isna(v) else str(v))
            continue

        bin_edges = edges[col]
        assert bin_edges is not None
        try:
            bin_indices = np.digitize(df[col].values, bin_edges[1:-1])
            df_binned[col] = [
                f"bin_{int(i)}" if pd.notna(v) else v
                for i, v in zip(bin_indices, df[col].values, strict=True)
            ]
        except (ValueError, TypeError):
            df_binned[col] = df[col].apply(lambda v: v if pd.isna(v) else str(v))
    return df_binned


def canonicalize_code_columns(
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
