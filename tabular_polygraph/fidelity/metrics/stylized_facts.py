from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tabular_polygraph._config import (
    DEFAULT_STYLIZED_CONCENTRATION_TOP,
    DEFAULT_STYLIZED_MIN_SAMPLES,
    DEFAULT_STYLIZED_TAIL_PERCENTILES,
)
from tabular_polygraph._types import Metric

from . import register


def _tail_integrity(r: np.ndarray, s: np.ndarray) -> float:
    p_high, p_low = DEFAULT_STYLIZED_TAIL_PERCENTILES
    r_p99, r_p50 = np.percentile(r, [p_high, p_low])
    s_p99, s_p50 = np.percentile(s, [p_high, p_low])
    r_tail = r_p99 / max(abs(r_p50), 1e-9)
    s_tail = s_p99 / max(abs(s_p50), 1e-9)
    return max(0.0, 100.0 - abs(r_tail - s_tail) / max(abs(r_tail), 1e-9) * 100)


def _predictive_parity(real: pd.DataFrame, syn: pd.DataFrame, cols: list[str]) -> float:
    r_corr = real[cols].corr(method="spearman")
    s_corr = syn[cols].corr(method="spearman")
    common = r_corr.index.intersection(s_corr.index)
    if len(common) < 2:
        return 100.0
    r_vals = r_corr.loc[common, common].values[
        np.triu_indices_from(r_corr.loc[common, common], k=1)
    ]
    s_vals = s_corr.loc[common, common].values[
        np.triu_indices_from(s_corr.loc[common, common], k=1)
    ]
    rank_match, _ = stats.spearmanr(r_vals, s_vals)
    if np.isnan(rank_match):
        return 100.0
    return max(0.0, float(rank_match) * 100)


def _concentration_match(r: np.ndarray, s: np.ndarray) -> float:
    top = DEFAULT_STYLIZED_CONCENTRATION_TOP
    r_sorted = np.sort(np.abs(r))[::-1]
    s_sorted = np.sort(np.abs(s))[::-1]
    r_top = r_sorted[: max(1, int(len(r_sorted) * top))].sum()
    r_bot = r_sorted.sum()
    s_top = s_sorted[: max(1, int(len(s_sorted) * top))].sum()
    s_bot = s_sorted.sum()
    r_conc = r_top / max(r_bot, 1e-9)
    s_conc = s_top / max(s_bot, 1e-9)
    return max(0.0, 100.0 - abs(r_conc - s_conc) / max(abs(r_conc), 1e-9) * 100)


@register
class StylizedFacts(Metric):
    name = "stylized_facts"

    def required_column_types(self) -> set[str]:
        return {"numeric"}

    def compute(
        self, real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]
    ) -> dict:
        per_column: dict = {}
        tested = 0

        for col in columns:
            r = real[col].dropna().values.astype(float)
            s = synthetic[col].dropna().values.astype(float)
            if (
                len(r) < DEFAULT_STYLIZED_MIN_SAMPLES
                or len(s) < DEFAULT_STYLIZED_MIN_SAMPLES
            ):
                continue
            tail = _tail_integrity(r, s)
            conc = _concentration_match(r, s)
            per_column[col] = {
                "tail_integrity": round(tail, 2),
                "concentration_match": round(conc, 2),
            }
            tested += 1

        parity = (
            _predictive_parity(real, synthetic, columns) if len(columns) >= 2 else None
        )

        if parity is not None:
            for col in per_column:
                per_column[col]["predictive_parity"] = round(parity, 2)

        scores = [v["tail_integrity"] for v in per_column.values()]
        scores.extend(v["concentration_match"] for v in per_column.values())
        if parity is not None:
            scores.append(parity)

        return {
            "per_column": per_column,
            "mean_score": round(float(np.mean(scores)), 2) if scores else None,
            "columns_tested": tested,
            "applicable": len(per_column) > 0,
        }
