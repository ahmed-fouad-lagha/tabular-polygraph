"""
Train-on-Synthetic, Test-on-Real (TSTR) utility evaluation.

The gold standard for synthetic data quality: train a model on synthetic
data, evaluate on held-out real data. Compare to Train-on-Real (TRR).

TSTR score = TSTR_metric / TRR_metric  (ratio, ideally close to 1.0)

Supports: classification (default_12m, action_taken) and regression tasks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tabular_polygraph.utils import numeric_columns


def _gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini coefficient from sorted predictions (proxy for AUC)."""
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    n = len(y_sorted)
    cumulative = np.cumsum(y_sorted)
    lorenz = cumulative / (cumulative[-1] + 1e-9)
    gini = (lorenz.sum() / n) - 0.5
    return float(2 * gini)


def _simple_logreg(X_train, y_train, X_test) -> np.ndarray:
    """Minimal logistic regression via gradient descent (no sklearn needed)."""
    X = np.column_stack([np.ones(len(X_train)), X_train])
    Xt = np.column_stack([np.ones(len(X_test)), X_test])
    w = np.zeros(X.shape[1])
    lr = 0.01
    for _ in range(200):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -10, 10)))
        grad = X.T @ (p - y_train) / len(y_train)
        w -= lr * grad
    return 1 / (1 + np.exp(-np.clip(Xt @ w, -10, 10)))


def _simple_linreg_r2(X_train, y_train, X_test, y_test) -> float:
    """R² from OLS."""
    X = np.column_stack([np.ones(len(X_train)), X_train])
    Xt = np.column_stack([np.ones(len(X_test)), X_test])
    try:
        b = np.linalg.lstsq(X, y_train, rcond=None)[0]
        pred = Xt @ b
        ss_res = np.sum((y_test - pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        return float(1 - ss_res / max(ss_tot, 1e-9))
    except Exception:
        return 0.0


def _standardize_with_train_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize train and test features with train-set statistics."""
    X_train = train_df[feature_cols].values.astype(float)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-9
    X_test = test_df[feature_cols].values.astype(float)
    return (X_train - mu) / sd, (X_test - mu) / sd


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
    all_cols = [c for c in real.columns if c in synthetic.columns]
    if feature_cols is None:
        feature_cols = [
            c for c in numeric_columns(real) if c in all_cols and c != target_col
        ]

    if not feature_cols or target_col not in real.columns:
        return {"error": f"target '{target_col}' not found or no numeric features"}

    # Infer task
    if task == "auto":
        n_unique = real[target_col].nunique()
        task = "classification" if n_unique <= 10 else "regression"

    # Prepare data
    real_clean = real[feature_cols + [target_col]].dropna()
    syn_clean = synthetic[feature_cols + [target_col]].dropna()

    real_train, real_test = train_test_split(
        real_clean, test_size=test_frac, random_state=seed
    )

    X_real_tr, X_test = _standardize_with_train_stats(
        real_train, real_test, feature_cols
    )
    y_real_tr = real_train[target_col].values
    y_test = real_test[target_col].values

    # Scale synthetic using REAL training stats (as it would be in a real TSTR scenario)
    # We treat syn as the "training" set but it must be in the same feature space as real
    mu = real_train[feature_cols].values.astype(float).mean(axis=0)
    sd = real_train[feature_cols].values.astype(float).std(axis=0) + 1e-9
    X_syn = (syn_clean[feature_cols].values.astype(float) - mu) / sd
    y_syn = syn_clean[target_col].values

    if task == "classification":
        model_tstr = RandomForestClassifier(n_estimators=100, random_state=seed)
        model_trr = RandomForestClassifier(n_estimators=100, random_state=seed)

        le = LabelEncoder()
        y_real_tr = le.fit_transform(y_real_tr.astype(str))
        y_test = le.transform(y_test.astype(str))
        # Handle potential missing classes in synthetic
        y_syn_str = y_syn.astype(str)
        # Filter syn to only classes present in real
        mask = y_syn_str.isin(le.classes_)
        if not mask.any():
            return {"error": "Synthetic target has no overlap with real classes"}
        X_syn = X_syn[mask]
        y_syn = le.transform(y_syn_str[mask])

        model_tstr.fit(X_syn, y_syn)
        model_trr.fit(X_real_tr, y_real_tr)

        tstr = f1_score(y_test, model_tstr.predict(X_test), average="macro")
        trr = f1_score(y_test, model_trr.predict(X_test), average="macro")
        metric = "f1_macro"
    else:
        model_tstr = RandomForestRegressor(n_estimators=100, random_state=seed)
        model_trr = RandomForestRegressor(n_estimators=100, random_state=seed)

        model_tstr.fit(X_syn, y_syn)
        model_trr.fit(X_real_tr, y_real_tr)

        tstr = r2_score(y_test, model_tstr.predict(X_test))
        trr = r2_score(y_test, model_trr.predict(X_test))
        metric = "r2"

    ratio = round(tstr / max(abs(trr), 1e-6), 4)

    return {
        "task": task,
        "metric": metric,
        "tstr_score": round(tstr, 4),
        "trr_score": round(trr, 4),
        "ratio": ratio,
        "target_col": target_col,
        "n_features": len(feature_cols),
        "n_synthetic_train": len(syn_clean),
        "n_real_test": len(real_test),
    }
