"""
Correct diagnostic: Does filtering HIF-flagged rows improve generalization to REAL data?
Also: sanity check that HIF detects adversarial corruption.
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


def test_real_generalization(dataset_id, target, n=3000):
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

    # Prepare real test data
    X_real, y_real = prepare(real, target, feature_cols)

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

        # CORRECT TEST: Train on subsets, test on REAL data
        median_penalty = np.median(rp)
        clean_mask = rp <= median_penalty
        broken_mask = rp > median_penalty

        # Baseline: train on ALL synthetic data
        clf_all = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
        clf_all.fit(X_syn, y_syn)
        acc_all = clf_all.score(X_real, y_real)

        # HIF-clean only: train on clean rows
        clf_clean = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
        clf_clean.fit(X_syn[clean_mask], y_syn[clean_mask])
        acc_clean = clf_clean.score(X_real, y_real)

        # HIF-broken only: train on broken rows
        if broken_mask.sum() > 10:
            clf_broken = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, random_state=42
            )
            clf_broken.fit(X_syn[broken_mask], y_syn[broken_mask])
            acc_broken = clf_broken.score(X_real, y_real)
        else:
            acc_broken = float("nan")

        # Also: aggressive filtering (remove top 25%)
        p75 = np.percentile(rp, 75)
        clean_75 = rp <= p75
        clf_75 = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
        clf_75.fit(X_syn[clean_75], y_syn[clean_75])
        acc_75 = clf_75.score(X_real, y_real)

        print(f"    Train on ALL synthetic:          {acc_all:.4f}")
        print(
            f"    Train on HIF-clean (bottom 50%): {acc_clean:.4f}  {'+' if acc_clean >= acc_all else ''}{(acc_clean - acc_all) * 100:+.2f}%"
        )
        print(
            f"    Train on HIF-broken (top 50%):   {acc_broken:.4f}  {'+' if acc_broken >= acc_all else ''}{(acc_broken - acc_all) * 100:+.2f}%"
        )
        print(
            f"    Train on HIF-clean (bottom 25%): {acc_75:.4f}  {'+' if acc_75 >= acc_all else ''}{(acc_75 - acc_all) * 100:+.2f}%"
        )

        # Show what HIF flags
        print("\n    HIF penalty stats:")
        print(
            f"      Mean: {rp.mean():.4f} | Max: {rp.max():.4f} | Violation rate: {hif_result['violation_rate']:.1%}"
        )
        print(
            f"      LSE rate: {hif_result['lse_violation_rate']:.1%} | NIC rate: {hif_result['nic_violation_rate']:.1%} | Rules rate: {hif_result['rule_violation_rate']:.1%}"
        )

        # What do flagged rows look like vs clean rows?
        if broken_mask.any() and clean_mask.any():
            print("\n    Sample comparison (first numeric col):")
            first_num = num_cols[0] if num_cols else None
            if first_num:
                clean_vals = syn.loc[clean_mask, first_num]
                broken_vals = syn.loc[broken_mask, first_num]
                real_vals = real[first_num]
                print(
                    f"      Real  median: {real_vals.median():.2f} | IQR: [{real_vals.quantile(0.25):.2f}, {real_vals.quantile(0.75):.2f}]"
                )
                print(
                    f"      Clean median: {clean_vals.median():.2f} | IQR: [{clean_vals.quantile(0.25):.2f}, {clean_vals.quantile(0.75):.2f}]"
                )
                print(
                    f"      Flag  median: {broken_vals.median():.2f} | IQR: [{broken_vals.quantile(0.25):.2f}, {broken_vals.quantile(0.75):.2f}]"
                )


def test_corruption_detection(dataset_id, target, n=3000):
    """Sanity check: HIF should detect adversarial corruption."""
    print(f"\n{'=' * 70}")
    print(f"CORRUPTION DETECTION: {dataset_id}")
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

    gen = GaussianCopulaGenerator()
    gen.fit(real)
    syn = gen.generate(len(real), seed=SEED).drop(columns=["syn_id"], errors="ignore")

    # Corrupt 20% of synthetic rows
    corrupt_fraction = 0.2
    n_corrupt = int(len(syn) * corrupt_fraction)
    corrupt_idx = np.random.RandomState(SEED).choice(len(syn), n_corrupt, replace=False)

    syn_corrupt = syn.copy()
    for col in num_cols[:3]:
        vals = syn_corrupt[col].values.copy().astype(float)
        vals[corrupt_idx] = vals[corrupt_idx] * 10 + 1000
        syn_corrupt[col] = vals

    hif_clean = hif_score(
        real, syn, columns=all_cols, hif_hubs=5, random_state=SEED, verbose=False
    )
    hif_corrupt = hif_score(
        real,
        syn_corrupt,
        columns=all_cols,
        hif_hubs=5,
        random_state=SEED,
        verbose=False,
    )

    rp_clean = hif_clean["row_penalties"]
    rp_corrupt = hif_corrupt["row_penalties"]

    print(
        f"  Clean:  mean_penalty={rp_clean.mean():.4f} | violation_rate={hif_clean['violation_rate']:.1%}"
    )
    print(
        f"  Corrupt: mean_penalty={rp_corrupt.mean():.4f} | violation_rate={hif_corrupt['violation_rate']:.1%}"
    )

    # Can HIF identify the corrupted rows?
    is_corrupt = np.zeros(len(syn_corrupt), dtype=bool)
    is_corrupt[corrupt_idx] = True

    # Use top-20% HIF penalty as "flagged"
    threshold = np.percentile(rp_corrupt, 80)
    flagged = rp_corrupt >= threshold
    precision = (flagged & is_corrupt).sum() / flagged.sum() if flagged.sum() > 0 else 0
    recall = (
        (flagged & is_corrupt).sum() / is_corrupt.sum() if is_corrupt.sum() > 0 else 0
    )
    print(f"  Detection: precision={precision:.3f} | recall={recall:.3f}")
    print(f"  (Random baseline precision: {corrupt_fraction:.1%})")


if __name__ == "__main__":
    test_corruption_detection("adult", "income")
    test_corruption_detection("credit", "default_payment")
    test_real_generalization("adult", "income")
    test_real_generalization("credit", "default_payment")
