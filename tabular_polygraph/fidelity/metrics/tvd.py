from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph._config import DEFAULT_TVD_MIN_SAMPLES
from tabular_polygraph._types import Metric

from . import register


@register
class TVD(Metric):
    name = "tvd"

    def required_column_types(self) -> set[str]:
        return {"categorical"}

    def compute(
        self, real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]
    ) -> dict:
        scores: dict[str, float] = {}
        for col in columns:
            if col not in real.columns or col not in synthetic.columns:
                continue
            r = real[col].dropna()
            s = synthetic[col].dropna()
            if len(r) < DEFAULT_TVD_MIN_SAMPLES or len(s) < DEFAULT_TVD_MIN_SAMPLES:
                continue

            r_raw = r.value_counts(normalize=True)
            s_raw = s.value_counts(normalize=True)
            all_cats = set(r_raw.index).union(set(s_raw.index))
            r_freq = r_raw.reindex(all_cats, fill_value=0.0)
            s_freq = s_raw.reindex(all_cats, fill_value=0.0)
            tvd = 0.5 * float((r_freq - s_freq).abs().sum())
            score = max(0.0, min(100.0, (1.0 - tvd) * 100.0))
            scores[col] = round(float(score), 2)

        return {"column_scores": scores, "mean_score": self._mean(scores)}

    @staticmethod
    def _mean(scores: dict[str, float]) -> float:
        return round(float(np.mean(list(scores.values()))), 2) if scores else 0.0
