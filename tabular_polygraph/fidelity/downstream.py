"""
Train-on-Synthetic, Test-on-Real (TSTR) utility evaluation.

The gold standard for synthetic data quality: train a model on synthetic
data, evaluate on held-out real data. Compare to Train-on-Real (TRR).

TSTR score = TSTR_metric / TRR_metric  (ratio, ideally close to 1.0)

Supports: classification (default_12m, action_taken) and regression tasks.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def tstr_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    task: str = "auto",
    test_frac: float = 0.3,
    seed: int = 42,
) -> dict:
    """
    Compute TSTR score: train on synthetic, evaluate on real using Random Forest.
    """
    all_cols = [c for c in real.columns if c in synthetic.columns and c != target_col]
    if feature_cols is None:
        feature_cols = all_cols

    if not feature_cols or target_col not in real.columns:
        return {"error": f"target '{target_col}' not found or no features available"}

    # Infer task
    if task == "auto":
        n_unique = real[target_col].nunique()
        task = "classification" if n_unique <= 10 else "regression"

    # Prepare data
    real_clean = real[feature_cols + [target_col]].dropna()
    syn_clean = synthetic[feature_cols + [target_col]].dropna()

    if len(real_clean) < 50:
        warnings.warn(
            f"Real evaluation dataset clean sample size ({len(real_clean)} rows) is < 50 after dropna().",
            UserWarning,
            stacklevel=2,
        )
    if len(syn_clean) < 50:
        warnings.warn(
            f"Synthetic evaluation dataset clean sample size ({len(syn_clean)} rows) is < 50 after dropna().",
            UserWarning,
            stacklevel=2,
        )

    real_train, real_test = train_test_split(
        real_clean, test_size=test_frac, random_state=seed
    )

    X_real_tr = real_train[feature_cols]
    y_real_tr = real_train[target_col].values
    X_test = real_test[feature_cols]
    y_test = real_test[target_col].values

    X_syn = syn_clean[feature_cols]
    y_syn = syn_clean[target_col].values

    num_cols = X_real_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_real_tr.select_dtypes(exclude=[np.number]).columns.tolist()

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
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            )
        )

    if not transformers:
        return {"error": "No valid numerical or categorical columns found for TSTR."}

    from sklearn.base import clone

    preprocessor = ColumnTransformer(transformers)

    if task == "classification":
        model_tstr = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("rf", RandomForestClassifier(n_estimators=100, random_state=seed)),
            ]
        )
        model_trr = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("rf", RandomForestClassifier(n_estimators=100, random_state=seed)),
            ]
        )

        le = LabelEncoder()
        y_real_tr = le.fit_transform(y_real_tr.astype(str))
        y_test = le.transform(y_test.astype(str))

        y_syn_str = y_syn.astype(str)
        mask = np.isin(y_syn_str, le.classes_)
        if not mask.any():
            return {"error": "Synthetic target has no overlap with real classes"}

        X_syn = X_syn.iloc[mask]
        y_syn = le.transform(y_syn_str[mask])

        model_tstr.fit(X_syn, y_syn)
        model_trr.fit(X_real_tr, y_real_tr)

        tstr = f1_score(y_test, model_tstr.predict(X_test), average="macro")
        trr = f1_score(y_test, model_trr.predict(X_test), average="macro")
        metric = "f1_macro"
    else:
        model_tstr = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("rf", RandomForestRegressor(n_estimators=100, random_state=seed)),
            ]
        )
        model_trr = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("rf", RandomForestRegressor(n_estimators=100, random_state=seed)),
            ]
        )

        model_tstr.fit(X_syn, y_syn)
        model_trr.fit(X_real_tr, y_real_tr)

        tstr = r2_score(y_test, model_tstr.predict(X_test))
        trr = r2_score(y_test, model_trr.predict(X_test))
        metric = "r2"

    ratio = round(float(tstr) / max(abs(float(trr)), 1e-6), 4)

    return {
        "task": task,
        "metric": metric,
        "tstr_score": round(float(tstr), 4),
        "trr_score": round(float(trr), 4),
        "ratio": ratio,
        "target_col": target_col,
        "n_features": len(feature_cols),
        "n_synthetic_train": len(syn_clean),
        "n_real_test": len(real_test),
    }
