"""
Correct diagnostic v2 — faster.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

PROJECT_ROOT = Path("/home/lagha/repos/tabular-polygraph")
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from scipy.stats import ks_2samp  # noqa: E402

from tabular_polygraph.dataset import load_dataset  # noqa: E402
from tabular_polygraph.fidelity import hif_score  # noqa: E402
from tabular_polygraph.generators import GaussianCopulaGenerator  # noqa: E402

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


def full_diagnostic(dataset_id, target, n=2000):
    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_id} | Target: {target} | n={n}")
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

    gen = GaussianCopulaGenerator()
    gen.fit(real)
    syn = gen.generate(len(real), seed=SEED).drop(columns=["syn_id"], errors="ignore")

    # HIF scoring
    hif_result = hif_score(
        real, syn, columns=all_cols, hif_hubs=5, random_state=SEED, verbose=False
    )
    rp = hif_result["row_penalties"]

    print(
        f"\n  HIF aggregate: score={hif_result['hif_score']:.4f} | violation_rate={hif_result['violation_rate']:.1%}"
    )
    print(
        f"  Component rates: LSE={hif_result['lse_violation_rate']:.1%} NIC={hif_result['nic_violation_rate']:.1%} Rules={hif_result['rule_violation_rate']:.1%}"
    )

    X_real, y_real = prepare(real, target, feature_cols)
    X_syn, y_syn = prepare(syn, target, feature_cols)

    median_penalty = np.median(rp)
    clean_mask = rp <= median_penalty
    broken_mask = rp > median_penalty

    # === TEST 1: Generalization to REAL data ===
    print("\n  --- TEST 1: Train on synthetic, test on REAL ---")
    clf_all = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42)
    clf_all.fit(X_syn, y_syn)
    acc_all = clf_all.score(X_real, y_real)

    clf_clean = GradientBoostingClassifier(
        n_estimators=80, max_depth=3, random_state=42
    )
    clf_clean.fit(X_syn[clean_mask], y_syn[clean_mask])
    acc_clean = clf_clean.score(X_real, y_real)

    if broken_mask.sum() > 20:
        clf_broken = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, random_state=42
        )
        clf_broken.fit(X_syn[broken_mask], y_syn[broken_mask])
        acc_broken = clf_broken.score(X_real, y_real)
    else:
        acc_broken = float("nan")

    # Aggressive: top 25%
    p75 = np.percentile(rp, 75)
    clean_75 = rp <= p75
    clf_75 = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42)
    clf_75.fit(X_syn[clean_75], y_syn[clean_75])
    acc_75 = clf_75.score(X_real, y_real)

    print(f"    ALL synthetic:       {acc_all:.4f}")
    print(
        f"    HIF-clean (bottom 50%): {acc_clean:.4f}  ({(acc_clean - acc_all) * 100:+.2f}%)"
    )
    print(
        f"    HIF-broken (top 50%):   {acc_broken:.4f}  ({(acc_broken - acc_all) * 100:+.2f}%)"
    )
    print(
        f"    HIF-clean (bottom 25%): {acc_75:.4f}  ({(acc_75 - acc_all) * 100:+.2f}%)"
    )

    # === TEST 2: Do flagged rows look different? ===
    print("\n  --- TEST 2: Flagged vs clean row statistics ---")
    for col in num_cols[:4]:
        r = real[col].dropna()
        c = syn.loc[clean_mask, col].dropna()
        b = syn.loc[broken_mask, col].dropna() if broken_mask.any() else c
        ks_clean = ks_2samp(r, c).statistic if len(c) > 0 else 0
        ks_broken = ks_2samp(r, b).statistic if len(b) > 0 else 0
        print(
            f"    KS({col}): clean={ks_clean:.4f} broken={ks_broken:.4f} {'(broken is closer!)' if ks_broken < ks_clean else ''}"
        )

    # === TEST 3: Corruption detection sanity check ===
    print("\n  --- TEST 3: Corruption detection ---")
    syn_corrupt = syn.copy()
    corrupt_frac = 0.2
    n_corrupt = int(len(syn) * corrupt_frac)
    corrupt_idx = np.random.RandomState(SEED).choice(len(syn), n_corrupt, replace=False)
    for col in num_cols[:3]:
        vals = syn_corrupt[col].values.copy().astype(float)
        vals[corrupt_idx] = vals[corrupt_idx] * 10 + 1000
        syn_corrupt[col] = vals

    hif_corrupt = hif_score(
        real,
        syn_corrupt,
        columns=all_cols,
        hif_hubs=5,
        random_state=SEED,
        verbose=False,
    )
    rp_corrupt = hif_corrupt["row_penalties"]

    is_corrupt = np.zeros(len(syn_corrupt), dtype=bool)
    is_corrupt[corrupt_idx] = True
    threshold = np.percentile(rp_corrupt, 80)
    flagged = rp_corrupt >= threshold
    precision = (flagged & is_corrupt).sum() / flagged.sum() if flagged.sum() > 0 else 0
    recall = (
        (flagged & is_corrupt).sum() / is_corrupt.sum() if is_corrupt.sum() > 0 else 0
    )
    print(
        f"    Corruption detection: P={precision:.3f} R={recall:.3f} (random baseline: {corrupt_frac:.0%})"
    )
    print(
        f"    Clean penalty: {rp.mean():.4f} -> Corrupt penalty: {rp_corrupt.mean():.4f}"
    )


if __name__ == "__main__":
    full_diagnostic("adult", "income")
