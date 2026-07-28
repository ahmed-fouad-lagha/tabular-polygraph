"""
Diagnostic: Do HIF penalties correlate with classifier errors?
If HIF flags the RIGHT rows, removing them should help.
If HIF flags the WRONG rows, the framework has a problem.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

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


def diagnose(dataset_id, target, n=3000):
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_id} | Target: {target}")
    print(f"{'=' * 60}")

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

        # Train classifier on SYNTHETIC data, test on REAL data
        feature_cols = num_cols + cat_cols
        X_syn, y_syn = prepare(syn, target, feature_cols)
        X_real, y_real = prepare(real, target, feature_cols)

        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf.fit(X_syn, y_syn)

        # Predict on each synthetic row — which ones does the classifier get WRONG?
        syn_preds = clf.predict(X_syn)
        syn_errors = (syn_preds != y_syn).astype(float)

        # Key question: do rows with HIGH HIF penalty have MORE classifier errors?
        print(f"    Overall error rate on synthetic: {syn_errors.mean():.1%}")

        # Split into HIF-clean (bottom 50%) and HIF-broken (top 50%)
        median_penalty = np.median(rp)
        clean_mask = rp <= median_penalty
        broken_mask = rp > median_penalty

        error_clean = syn_errors[clean_mask].mean()
        error_broken = syn_errors[broken_mask].mean()
        print(f"    Error rate on HIF-clean rows:    {error_clean:.1%}")
        print(f"    Error rate on HIF-broken rows:   {error_broken:.1%}")
        print(f"    Ratio (broken/clean):            {error_broken / error_clean:.2f}x")

        # Also check: correlation between HIF penalty and classifier error
        corr = np.corrcoef(rp, syn_errors)[0, 1]
        print(f"    Correlation(HIF penalty, error): {corr:.4f}")

        # A synthetic row is "useful" if training on it helps predict real data
        # But simpler: check if HIF-clean rows are closer to real data distribution
        from scipy.stats import ks_2samp

        for col in num_cols[:3]:
            real_vals = real[col].dropna().values
            clean_vals = syn.loc[clean_mask, col].dropna().values
            broken_vals = syn.loc[broken_mask, col].dropna().values
            ks_clean = ks_2samp(real_vals, clean_vals).statistic
            ks_broken = ks_2samp(real_vals, broken_vals).statistic
            print(f"    KS({col}): clean={ks_clean:.4f} broken={ks_broken:.4f}")


if __name__ == "__main__":
    diagnose("adult", "income")
    diagnose("credit", "default_payment")
