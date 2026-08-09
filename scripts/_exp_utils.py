"""Shared helpers for experiment scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabular_polygraph.dataset import load_dataset
from tabular_polygraph.fidelity.hif.orchestrator import hif_score
from tabular_polygraph.fidelity.hif.rules import rule_violation_score
from tabular_polygraph.fidelity.metrics.alpha_beta import AlphaBeta
from tabular_polygraph.fidelity.metrics.correlation import Correlation
from tabular_polygraph.fidelity.metrics.ks import KSTest
from tabular_polygraph.fidelity.metrics.moment_matching import MomentMatching
from tabular_polygraph.fidelity.metrics.tvd import TVD
from tabular_polygraph.generators import (
    CTGANGenerator,
    GaussianCopulaGenerator,
    TVAEGenerator,
    VineCopulaGenerator,
)

GENERATORS = {
    "gaussian": GaussianCopulaGenerator,
    "gaussian_copula": GaussianCopulaGenerator,
    "copula": GaussianCopulaGenerator,
    "vine": VineCopulaGenerator,
    "ctgan": CTGANGenerator,
    "tvae": TVAEGenerator,
}


def load_real(dataset_id: str, n: int = 2000) -> pd.DataFrame:
    """Load cached real data, re-casting integer/float dtypes explicitly."""
    real = load_dataset(dataset_id, n=n)
    for col in real.columns:
        if pd.api.types.is_integer_dtype(real[col]):
            real[col] = real[col].astype("int64")
        elif pd.api.types.is_float_dtype(real[col]):
            real[col] = real[col].astype("float64")
    return real.reset_index(drop=True)


def generate(
    real: pd.DataFrame,
    n: int,
    seed: int,
    generator: str,
    epochs: int | None = None,
) -> pd.DataFrame:
    """Fit a generator on ``real`` and draw ``n`` synthetic rows (no syn_id).

    Seeds the global RNG *before fitting* so stochastic generators (CTGAN,
    TVAE) train deterministically for a given ``seed``.
    """
    from tabular_polygraph._utils import set_seed

    set_seed(seed)
    cls = GENERATORS[generator]
    if generator == "ctgan":
        gen = CTGANGenerator(epochs=epochs) if epochs else CTGANGenerator()
    elif generator == "tvae":
        gen = TVAEGenerator(epochs=epochs) if epochs else TVAEGenerator()
    else:
        gen = cls()
    gen.fit(real)
    syn = gen.generate(n, seed=seed)
    return syn.drop(columns=["syn_id"], errors="ignore")


def audit_hif(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    seed: int = 42,
    **kwargs,
) -> dict:
    """Run the current HIF orchestrator and return its result dict."""
    columns = real.columns.intersection(synthetic.columns).tolist()
    return hif_score(real, synthetic, columns=columns, random_state=seed, **kwargs)


def rule_mask(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    seed: int = 42,
    min_confidence: float = 0.95,
    min_support: float = 0.005,
    max_rules: int = 25,
    min_lift: float = 1.0,
    max_antecedents: int = 2,
) -> np.ndarray:
    """Return row-level rule-violation mask (1 = violation) for synthetic rows."""
    columns = real.columns.intersection(synthetic.columns).tolist()
    result = rule_violation_score(
        real,
        synthetic,
        columns=columns,
        min_confidence=min_confidence,
        min_support=min_support,
        max_rules=max_rules,
        min_lift=min_lift,
        max_antecedents=max_antecedents,
        random_state=seed,
    )
    return result.get("row_violation_mask", np.zeros(len(synthetic)))


def _utility_feature_frame(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target: str,
    max_cat_cardinality: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build a feature set: numeric features + OHE of low-cardinality cats."""
    num_cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]
    cat_cols = [c for c in real.columns if not pd.api.types.is_numeric_dtype(real[c])]

    real_util = real.copy()
    syn_util = syn.copy()
    encoded_cols: list[str] = []

    for col in cat_cols:
        if col == target or col not in syn.columns:
            continue
        if real[col].nunique() > max_cat_cardinality:
            continue
        dummies = pd.get_dummies(real[col], prefix=f"ohe__{col}").astype(float)
        real_util = pd.concat([real_util, dummies], axis=1)
        syn_dummies = pd.get_dummies(syn[col], prefix=f"ohe__{col}").astype(float)
        for d_col in dummies.columns:
            if d_col in syn_dummies.columns:
                syn_util[d_col] = syn_dummies[d_col]
            else:
                syn_util[d_col] = 0.0
        encoded_cols.extend(dummies.columns)

    feature_cols = [
        c for c in num_cols if c != target and c in syn.columns
    ] + encoded_cols
    return real_util, syn_util, feature_cols


def utility_metrics(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target: str,
    seed: int = 42,
) -> dict:
    """TSTR: train a Random Forest on synthetic, evaluate on a real holdout.

    Continuous targets are discretized to a binary median split, matching the
    protocol used by the significance/ablation experiments.
    """
    if target not in real.columns or target not in syn.columns:
        return {"f1": np.nan, "accuracy": np.nan, "trr": np.nan}

    real_util, syn_util, feature_cols = _utility_feature_frame(real, syn, target)
    if len(feature_cols) < 1:
        return {"f1": np.nan, "accuracy": np.nan, "trr": np.nan}

    u_real = real_util.copy()
    u_syn = syn_util.copy()
    if pd.api.types.is_numeric_dtype(u_real[target]) and u_real[target].nunique() > 2:
        m = u_real[target].median()
        u_real[target] = (u_real[target] > m).astype(int)
        u_syn[target] = (u_syn[target] > m).astype(int)
    elif not pd.api.types.is_numeric_dtype(u_real[target]):
        cats = u_real[target].astype("category").cat.categories.tolist()
        u_real[target] = u_real[target].map({c: i for i, c in enumerate(cats)})
        u_syn[target] = u_syn[target].map({c: i for i, c in enumerate(cats)}).fillna(-1)

    if u_real[target].nunique() < 2:
        return {"f1": np.nan, "accuracy": np.nan, "trr": np.nan}

    X_real = u_real[feature_cols]
    y_real = u_real[target].astype(int)
    X_syn = u_syn[feature_cols]
    y_syn = u_syn[target].astype(int)

    if len(u_syn) < 10:
        return {"f1": np.nan, "accuracy": np.nan, "trr": np.nan}

    X_tr, X_test, y_tr, y_test = train_test_split(
        X_real, y_real, test_size=0.3, random_state=seed
    )

    pipe = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            [
                                c
                                for c in feature_cols
                                if pd.api.types.is_numeric_dtype(u_real[c])
                            ],
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    (
                                        "imputer",
                                        SimpleImputer(strategy="most_frequent"),
                                    ),
                                    (
                                        "ohe",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            sparse_output=False,
                                        ),
                                    ),
                                ]
                            ),
                            [
                                c
                                for c in feature_cols
                                if not pd.api.types.is_numeric_dtype(u_real[c])
                            ],
                        ),
                    ]
                ),
            ),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=seed)),
        ]
    )
    pipe.fit(X_syn, y_syn)
    preds = pipe.predict(X_test)
    f1 = float(f1_score(y_test, preds, average="macro", zero_division=0.0))
    acc = float(accuracy_score(y_test, preds))

    trr_pipe = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            [
                                c
                                for c in feature_cols
                                if pd.api.types.is_numeric_dtype(u_real[c])
                            ],
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    (
                                        "imputer",
                                        SimpleImputer(strategy="most_frequent"),
                                    ),
                                    (
                                        "ohe",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            sparse_output=False,
                                        ),
                                    ),
                                ]
                            ),
                            [
                                c
                                for c in feature_cols
                                if not pd.api.types.is_numeric_dtype(u_real[c])
                            ],
                        ),
                    ]
                ),
            ),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=seed)),
        ]
    )
    trr_pipe.fit(X_tr, y_tr)
    trr = float(
        f1_score(y_test, trr_pipe.predict(X_test), average="macro", zero_division=0.0)
    )

    return {"f1": f1, "accuracy": acc, "trr": trr}


def aggregate_metrics(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    """JCD, mean moment-matching, mean KS, mean TVD from the current metric classes."""
    columns = real.columns.intersection(synthetic.columns).tolist()
    out: dict = {"jcd": np.nan, "mm": np.nan, "ks": np.nan, "tvd": np.nan}

    try:
        res = Correlation().compute(real, synthetic, columns)
        out["jcd"] = res.get("correlation_distance_score", np.nan)
    except Exception:
        pass
    try:
        num_cols = [c for c in columns if pd.api.types.is_numeric_dtype(real[c])]
        if num_cols:
            res = MomentMatching().compute(real, synthetic, num_cols)
            out["mm"] = res.get("mean_score", np.nan)
            res = KSTest().compute(real, synthetic, num_cols)
            out["ks"] = res.get("mean_score", np.nan)
    except Exception:
        pass
    try:
        cat_cols = [c for c in columns if not pd.api.types.is_numeric_dtype(real[c])]
        if cat_cols:
            res = TVD().compute(real, synthetic, cat_cols)
            out["tvd"] = res.get("mean_score", np.nan)
    except Exception:
        pass
    try:
        res = AlphaBeta().compute(real, synthetic, columns)
        out["alpha_precision"] = res.get("alpha_precision", np.nan)
        out["beta_recall"] = res.get("beta_recall", np.nan)
        out["authenticity"] = res.get("authenticity", np.nan)
    except Exception:
        pass
    return out
