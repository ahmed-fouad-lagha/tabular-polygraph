from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

from tabular_polygraph._config import DEFAULT_JOINT_MIN_COLS
from tabular_polygraph._types import Metric

from . import register


def _cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    crosstab = pd.crosstab(x, y)
    if crosstab.size == 0:
        return 0.0
    chi2 = chi2_contingency(crosstab, correction=False)[0]
    n = crosstab.sum().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = crosstab.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def _correlation_ratio(categories: np.ndarray, measurements: np.ndarray) -> float:
    df = pd.DataFrame({"c": categories, "m": measurements})
    y_mean = df["m"].mean()
    if pd.isna(y_mean):
        return 0.0
    grouped = df.groupby("c")["m"]
    numerator = sum(len(g) * (g.mean() - y_mean) ** 2 for _, g in grouped)
    denominator = sum((df["m"] - y_mean) ** 2)
    if denominator == 0:
        return 0.0
    eta2 = float(numerator / denominator)
    return float(np.sqrt(max(0.0, min(1.0, eta2))))


def _association_matrix(df: pd.DataFrame, is_num: dict[str, bool]) -> np.ndarray:
    n = df.shape[1]
    cols = df.columns.tolist()
    mat = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            ca, cb = cols[i], cols[j]
            mask = df[ca].notna() & df[cb].notna()
            if mask.sum() < 2:
                continue
            xa = df.loc[mask, ca].values
            xb = df.loc[mask, cb].values

            if is_num[ca] and is_num[cb]:
                xa = pd.to_numeric(xa, errors="coerce")
                xb = pd.to_numeric(xb, errors="coerce")
                keep = pd.notna(xa) & pd.notna(xb)
                if keep.sum() < 2:
                    val = 0.0
                else:
                    r, _ = spearmanr(xa[keep].astype(float), xb[keep].astype(float))
                    val = r if np.isfinite(r) else 0.0
            elif not is_num[ca] and not is_num[cb]:
                val = _cramers_v(xa, xb)
            else:
                if is_num[ca]:
                    xa = pd.to_numeric(xa, errors="coerce")
                    keep = pd.notna(xa)
                    val = (
                        _correlation_ratio(xb[keep], xa[keep].astype(float))
                        if keep.sum() >= 2
                        else 0.0
                    )
                else:
                    xb = pd.to_numeric(xb, errors="coerce")
                    keep = pd.notna(xb)
                    val = (
                        _correlation_ratio(xa[keep], xb[keep].astype(float))
                        if keep.sum() >= 2
                        else 0.0
                    )
            mat[i, j] = mat[j, i] = val
    return mat


@register
class Correlation(Metric):
    name = "correlation"

    def required_column_types(self) -> set[str]:
        return {"all"}

    def compute(
        self, real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]
    ) -> dict:
        if len(columns) < DEFAULT_JOINT_MIN_COLS:
            return {"correlation_distance_score": 100.0, "pairwise_deltas": {}}

        is_num = {c: pd.api.types.is_numeric_dtype(real[c]) for c in columns}
        R_real = _association_matrix(real[columns], is_num)
        R_syn = _association_matrix(synthetic[columns], is_num)

        max_possible = 2.0 * np.sqrt(len(columns) * (len(columns) - 1))
        dist = np.linalg.norm(R_real - R_syn, "fro")
        score = max(0.0, 1 - float(dist) / max(float(max_possible), 1e-8)) * 100

        pairwise = {}
        for i, ca in enumerate(columns):
            for j, cb in enumerate(columns):
                if j <= i:
                    continue
                delta = round(float(abs(R_real[i, j] - R_syn[i, j])), 4)
                pairwise[f"{ca} \u00d7 {cb}"] = delta

        return {
            "correlation_distance_score": round(float(score), 2),
            "pairwise_deltas": pairwise,
        }
