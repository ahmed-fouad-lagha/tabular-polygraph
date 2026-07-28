from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from tabular_polygraph._config import (
    DEFAULT_DOWNSTREAM_CLASS_THRESHOLD,
    DEFAULT_DOWNSTREAM_N_ESTIMATORS,
    DEFAULT_DOWNSTREAM_TEST_FRAC,
)
from tabular_polygraph._types import Metric
from tabular_polygraph._utils import categorical_columns, numeric_columns

from . import register


@register
class Downstream(Metric):
    name = "downstream"

    def __init__(self, target_col: str | None = None):
        self._target_col = target_col

    def required_column_types(self) -> set[str]:
        return {"all"}

    def validate(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> str | None:
        if not self._target_col or self._target_col not in real.columns:
            return "No valid target column provided"
        return None

    def compute(
        self, real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]
    ) -> dict:
        feature_cols = [c for c in columns if c != self._target_col]
        real = real[[self._target_col] + feature_cols].dropna().reset_index(drop=True)
        syn = synthetic[feature_cols].dropna().reset_index(drop=True)

        if len(real) < 50 or len(syn) < 10:
            return {"error": "Too few rows for TSTR evaluation"}

        y_real = real[self._target_col]
        X_real = real[feature_cols]
        n_unique = y_real.nunique()
        if n_unique <= DEFAULT_DOWNSTREAM_CLASS_THRESHOLD:
            task = "class"
            metric_fn = f1_score
            metric_name = "f1_macro"
        else:
            task = "reg"
            metric_fn = r2_score
            metric_name = "r2"

        num_cols = numeric_columns(X_real)
        cat_cols = categorical_columns(X_real)
        transformers = []
        if num_cols:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_cols,
                )
            )
        if cat_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "ohe",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    cat_cols,
                )
            )

        if not transformers:
            return {"error": "No valid features for TSTR"}

        preprocessor = ColumnTransformer(transformers)
        test_frac = DEFAULT_DOWNSTREAM_TEST_FRAC
        X_real_tr, X_test, y_real_tr, y_test = train_test_split(
            X_real, y_real, test_size=test_frac, random_state=42
        )

        X_syn_pre = preprocessor.fit_transform(X_real_tr)
        X_real_pre = preprocessor.transform(X_real_tr)
        X_test_pre = preprocessor.transform(X_test)

        if task == "class":
            le = LabelEncoder()
            y_real_enc = le.fit_transform(y_real_tr.astype(str))
            y_test_enc = le.transform(y_test.astype(str))

            mask = np.isin(y_real_enc, le.classes_)
            if mask.sum() < 10:
                return {
                    "error": "Too few valid synthetic samples after label alignment"
                }

            rf_tstr = RandomForestClassifier(
                n_estimators=DEFAULT_DOWNSTREAM_N_ESTIMATORS, random_state=42
            )
            rf_trr = RandomForestClassifier(
                n_estimators=DEFAULT_DOWNSTREAM_N_ESTIMATORS, random_state=42
            )

            rf_tstr.fit(X_syn_pre[mask], y_real_enc[mask])
            rf_trr.fit(X_real_pre, y_real_enc)

            tstr = float(
                metric_fn(y_test_enc, rf_tstr.predict(X_test_pre), average="macro")
            )
            trr = float(
                metric_fn(y_test_enc, rf_trr.predict(X_test_pre), average="macro")
            )
        else:
            rf_tstr = RandomForestRegressor(
                n_estimators=DEFAULT_DOWNSTREAM_N_ESTIMATORS, random_state=42
            )
            rf_trr = RandomForestRegressor(
                n_estimators=DEFAULT_DOWNSTREAM_N_ESTIMATORS, random_state=42
            )
            rf_tstr.fit(X_syn_pre, y_real_tr)
            rf_trr.fit(X_real_pre, y_real_tr)

            tstr = float(metric_fn(y_test, rf_tstr.predict(X_test_pre)))
            trr = float(metric_fn(y_test, rf_trr.predict(X_test_pre)))
            metric_name = "r2"

        ratio = round(tstr / max(abs(trr), 1e-6), 4)

        return {
            "target_col": self._target_col,
            "metric": metric_name,
            "task": task,
            "tstr_score": round(tstr, 4),
            "trr_score": round(trr, 4),
            "ratio": ratio,
        }
