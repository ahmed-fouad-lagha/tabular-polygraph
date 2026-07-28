from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tabular_polygraph._config import (
    DEFAULT_MOMENT_EPS,
    DEFAULT_MOMENT_MIN_SAMPLES,
    DEFAULT_MOMENT_WEIGHTS,
)
from tabular_polygraph._types import Metric
from tabular_polygraph._utils import to_numeric_array

from . import register


@register
class MomentMatching(Metric):
    name = "moment_matching"

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
            if (
                len(r) < DEFAULT_MOMENT_MIN_SAMPLES
                or len(s) < DEFAULT_MOMENT_MIN_SAMPLES
            ):
                continue

            mean_r, std_r = float(r.mean()), float(r.std(ddof=0))
            mean_s, std_s = float(s.mean()), float(s.std(ddof=0))
            skew_r = float(stats.skew(r, nan_policy="omit"))
            skew_s = float(stats.skew(s, nan_policy="omit"))
            kurt_r = float(stats.kurtosis(r, fisher=False, nan_policy="omit"))
            kurt_s = float(stats.kurtosis(s, fisher=False, nan_policy="omit"))

            if not np.isfinite(skew_r):
                skew_r = 0.0
            if not np.isfinite(skew_s):
                skew_s = 0.0
            if not np.isfinite(kurt_r):
                kurt_r = 0.0
            if not np.isfinite(kurt_s):
                kurt_s = 0.0

            eps = DEFAULT_MOMENT_EPS
            mean_err = min(abs(mean_s - mean_r) / max(abs(mean_r), eps), 1.0)
            std_err = min(abs(std_s - std_r) / max(std_r, eps), 1.0)
            skew_err = min(abs(skew_s - skew_r) / (abs(skew_r) + 0.5), 1.0)
            kurt_err = min(abs(kurt_s - kurt_r) / (abs(kurt_r) + 1.0), 1.0)

            w = DEFAULT_MOMENT_WEIGHTS
            score = 100.0 * (
                1.0
                - w[0] * mean_err
                - w[1] * std_err
                - w[2] * skew_err
                - w[3] * kurt_err
            )
            scores[col] = round(float(max(0.0, min(100.0, score))), 2)

        return {"column_scores": scores, "mean_score": self._mean(scores)}

    @staticmethod
    def _mean(scores: dict[str, float]) -> float:
        return round(float(np.mean(list(scores.values()))), 2) if scores else 0.0
