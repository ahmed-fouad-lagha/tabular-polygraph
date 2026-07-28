"""
Pilot Study v2: Fixed targets, regression for supermarket_sales,
multiple filtering thresholds to find the sweet spot.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = Path("/home/lagha/repos/tabular-polygraph")
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from tabular_polygraph.dataset import load_dataset  # noqa: E402
from tabular_polygraph.fidelity import hif_score  # noqa: E402
from tabular_polygraph.generators import (  # noqa: E402
    GaussianCopulaGenerator,
    VineCopulaGenerator,
)
from tabular_polygraph.generators.base import BaseGenerator  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42
sys.stdout.reconfigure(line_buffering=True)


def make_generator(name: str) -> BaseGenerator:
    return (
        GaussianCopulaGenerator()
        if name == "gaussian_copula"
        else VineCopulaGenerator()
    )


def downstream(real, syn, target, task, syn_filtered=None):
    num_cols = [
        c
        for c in real.columns
        if c != target and pd.api.types.is_numeric_dtype(real[c])
    ]
    cat_cols = [
        c
        for c in real.columns
        if c != target and not pd.api.types.is_numeric_dtype(real[c])
    ]
    feature_cols = num_cols + cat_cols

    def prepare(df):
        X = pd.DataFrame(df[feature_cols]).copy()
        for c in cat_cols:
            X[c] = pd.Categorical(X[c]).codes
        y = pd.Series(df[target])
        if task == "classification":
            y = pd.Categorical(y).codes
        return X.values, np.array(y)

    clf = (
        GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        if task == "classification"
        else GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    )
    metric = "accuracy" if task == "classification" else "r2"

    X_real, y_real = prepare(real)
    real_score = cross_val_score(clf, X_real, y_real, cv=2, scoring=metric).mean()

    X_syn, y_syn = prepare(syn)
    syn_score = cross_val_score(clf, X_syn, y_syn, cv=2, scoring=metric).mean()

    results = {"real": round(real_score, 4), "synthetic": round(syn_score, 4)}

    if syn_filtered is not None and len(syn_filtered) > 20:
        X_f, y_f = prepare(syn_filtered)
        if len(np.unique(y_f)) >= 2:
            filt_score = cross_val_score(clf, X_f, y_f, cv=2, scoring=metric).mean()
            results["filtered"] = round(filt_score, 4)
        else:
            results["filtered"] = None
    else:
        results["filtered"] = None

    return results


def run():
    print("=" * 90)
    print("PILOT STUDY v2")
    print("=" * 90)

    configs = [
        ("adult", "income", "classification", 3000),
        ("credit", "default_payment", "classification", 3000),
        ("supermarket_sales", "customer_rating", "regression", 1000),
    ]
    generators = ["gaussian_copula", "vine_copula"]
    thresholds = [100, 80, 70, 60]  # keep top X% by HIF

    for dataset_id, target, task, n in configs:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_id} | Target: {target} | Task: {task}")
        print(f"{'=' * 70}")

        real = load_dataset(dataset_id, n=n)
        num_cols = [
            c
            for c in real.columns
            if c != target and pd.api.types.is_numeric_dtype(real[c])
        ]
        cat_cols = [
            c
            for c in real.columns
            if c != target and not pd.api.types.is_numeric_dtype(real[c])
        ]
        all_cols = num_cols + cat_cols

        for gen_name in generators:
            print(f"\n  {gen_name}:")
            try:
                gen = make_generator(gen_name)
                gen.fit(real)
                syn = gen.generate(len(real), seed=SEED).drop(
                    columns=["syn_id"], errors="ignore"
                )
            except Exception as e:
                print(f"    FAILED: {e}")
                continue

            hif_result = hif_score(
                real,
                syn,
                columns=all_cols,
                hif_hubs=5,
                random_state=SEED,
                verbose=False,
            )
            rp = hif_result["row_penalties"]

            util_all = downstream(real, syn, target, task)
            line = f"    HIF={hif_result['hif_score']:.3f} viol={hif_result['violation_rate']:.1%} | "
            line += f"Real={util_all['real']:.4f} Syn={util_all['synthetic']:.4f}"

            for pct in thresholds:
                if pct == 100:
                    continue
                thresh = np.percentile(rp, 100 - pct)
                syn_filt = syn[rp <= thresh]
                util_filt = downstream(real, syn, target, task, syn_filt)
                if util_filt.get("filtered") is not None:
                    delta = util_filt["filtered"] - util_all["synthetic"]
                    sign = "+" if delta >= 0 else ""
                    line += f" | {pct}%={util_filt['filtered']:.4f}({sign}{delta:.4f})"

            print(line)


if __name__ == "__main__":
    run()
