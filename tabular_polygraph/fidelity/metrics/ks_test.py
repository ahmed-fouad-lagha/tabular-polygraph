from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tabular_polygraph._config import DEFAULT_KS_MIN_SAMPLES
from tabular_polygraph._types import Metric
from tabular_polygraph._utils import to_numeric_array

from . import register


@register
class KSTest(Metric):
    name = "ks_test"

    def required_column_types(self) -> set[str]:
        return {"numeric"}

    def compute(
        self, real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]
    ) -> dict:
        scores: dict[str, float] = {}
        for col in columns:
            if col not in real.columns or col not in synthetic.columns:
                continue
            r = to_numeric_array(real[col], fill_method="dropna")
            s = to_numeric_array(synthetic[col], fill_method="dropna")
            if len(r) < DEFAULT_KS_MIN_SAMPLES or len(s) < DEFAULT_KS_MIN_SAMPLES:
                continue

            ks_stat, _ = stats.ks_2samp(r, s)
            if not np.isfinite(ks_stat):
                ks_stat = 1.0
            scores[col] = round(float(max(0.0, min(100.0, (1.0 - ks_stat) * 100.0))), 2)

        return {"column_scores": scores, "mean_score": self._mean(scores)}

    @staticmethod
    def _mean(scores: dict[str, float]) -> float:
        return round(float(np.mean(list(scores.values()))), 2) if scores else 0.0
