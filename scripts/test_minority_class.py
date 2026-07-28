"""
Test: Does HIF filtering improve minority-class F1 specifically?
Hypothesis: HIF flags rows with broken inter-column dependencies.
Removing them should help the classifier learn minority-class patterns.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

PROJECT_ROOT = Path("/home/lagha/repos/tabular-polygraph")
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from tabular_polygraph.dataset import load_dataset  # noqa: E402
from tabular_polygraph.fidelity import hif_score  # noqa: E402
from tabular_polygraph.generators import (  # noqa: E402
    GaussianCopulaGenerator,
    VineCopulaGenerator,
)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

SEED = 42


def prepare(df, target, feature_cols):
    X = pd.DataFrame(df[feature_cols]).copy()
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c]).codes
    y = pd.Categorical(df[target]).codes
    return X.values, np.array(y)


def test_minority_class(dataset_id, target, n=3000):
    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_id} | Target: {target}")
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
    feature_cols = num_cols + cat_cols
    all_cols = feature_cols

    X_real, y_real = prepare(real, target, feature_cols)

    # Check class distribution
    unique, counts = np.unique(y_real, return_counts=True)
    print(f"  Real class distribution: {dict(zip(unique, counts, strict=False))}")
    minority_class = unique[np.argmin(counts)]
    print(f"  Minority class: {minority_class} ({counts.min() / len(y_real):.1%})")

    for gen_name, gen_cls in [
        ("gaussian_copula", GaussianCopulaGenerator),
        ("vine_copula", VineCopulaGenerator),
    ]:
        print(f"\n  {gen_name}:")

        gen = gen_cls()
        gen.fit(real)
        syn = gen.generate(len(real), seed=SEED).drop(
            columns=["syn_id"], errors="ignore"
        )

        # Get HIF penalties
        hif_result = hif_score(
            real, syn, columns=all_cols, hif_hubs=5, random_state=SEED, verbose=False
        )
        rp = hif_result["row_penalties"]

        X_syn, y_syn = prepare(syn, target, feature_cols)

        # Check synthetic class distribution
        unique_syn, counts_syn = np.unique(y_syn, return_counts=True)
        print(
            f"    Syn class distribution: {dict(zip(unique_syn, counts_syn, strict=False))}"
        )

        # === BASELINE: Train on ALL synthetic data ===
        clf_all = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
        clf_all.fit(X_syn, y_syn)
        y_pred_all = clf_all.predict(X_real)
        f1_all_macro = f1_score(y_real, y_pred_all, average="macro")
        f1_all_minority = f1_score(
            y_real, y_pred_all, pos_label=minority_class, zero_division=0
        )
        f1_all_weighted = f1_score(y_real, y_pred_all, average="weighted")

        # === HIF-CLEAN: Filter bottom 50% ===
        median_penalty = np.median(rp)
        clean_mask = rp <= median_penalty
        clf_clean = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
        clf_clean.fit(X_syn[clean_mask], y_syn[clean_mask])
        y_pred_clean = clf_clean.predict(X_real)
        f1_clean_macro = f1_score(y_real, y_pred_clean, average="macro")
        f1_clean_minority = f1_score(
            y_real, y_pred_clean, pos_label=minority_class, zero_division=0
        )
        f1_clean_weighted = f1_score(y_real, y_pred_clean, average="weighted")

        # === HIF-CLEAN: Filter bottom 25% (most aggressive) ===
        p75 = np.percentile(rp, 75)
        clean_75 = rp <= p75
        clf_75 = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
        clf_75.fit(X_syn[clean_75], y_syn[clean_75])
        y_pred_75 = clf_75.predict(X_real)
        f1_75_macro = f1_score(y_real, y_pred_75, average="macro")
        f1_75_minority = f1_score(
            y_real, y_pred_75, pos_label=minority_class, zero_division=0
        )
        f1_75_weighted = f1_score(y_real, y_pred_75, average="weighted")

        # === RANDOM filter (control) ===
        rng = np.random.RandomState(SEED)
        random_clean = rng.rand(len(syn)) > 0.5
        clf_rand = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
        clf_rand.fit(X_syn[random_clean], y_syn[random_clean])
        y_pred_rand = clf_rand.predict(X_real)
        f1_rand_macro = f1_score(y_real, y_pred_rand, average="macro")
        f1_rand_minority = f1_score(
            y_real, y_pred_rand, pos_label=minority_class, zero_division=0
        )

        print(
            f"    {'Method':<30} {'Macro F1':>10} {'Minority F1':>12} {'Weighted F1':>12}"
        )
        print(f"    {'-' * 66}")
        print(
            f"    {'ALL synthetic':<30} {f1_all_macro:>10.4f} {f1_all_minority:>12.4f} {f1_all_weighted:>12.4f}"
        )
        print(
            f"    {'RANDOM filter (50%)':<30} {f1_rand_macro:>10.4f} {f1_rand_minority:>12.4f} {'':>12}"
        )
        print(
            f"    {'HIF-clean (bottom 50%)':<30} {f1_clean_macro:>10.4f} {f1_clean_minority:>12.4f} {f1_clean_weighted:>12.4f}"
        )
        print(
            f"    {'HIF-clean (bottom 25%)':<30} {f1_75_macro:>10.4f} {f1_75_minority:>12.4f} {f1_75_weighted:>12.4f}"
        )

        delta_minority_50 = (f1_clean_minority - f1_all_minority) * 100
        delta_minority_25 = (f1_75_minority - f1_all_minority) * 100
        delta_random = (f1_rand_minority - f1_all_minority) * 100
        print(
            f"\n    Minority F1 delta: HIF-50={delta_minority_50:+.2f}% | HIF-25={delta_minority_25:+.2f}% | Random={delta_random:+.2f}%"
        )

        # Check: do HIF-flagged rows contain more minority class samples?
        broken_mask = rp > median_penalty
        minority_rate_real = (y_real == minority_class).mean()
        minority_rate_clean = (
            (y_syn[clean_mask] == minority_class).mean() if clean_mask.any() else 0
        )
        minority_rate_broken = (
            (y_syn[broken_mask] == minority_class).mean() if broken_mask.any() else 0
        )
        print(
            f"    Minority rate: real={minority_rate_real:.1%} | clean={minority_rate_clean:.1%} | broken={minority_rate_broken:.1%}"
        )


if __name__ == "__main__":
    test_minority_class("adult", "income")
    test_minority_class("credit", "default_payment")
