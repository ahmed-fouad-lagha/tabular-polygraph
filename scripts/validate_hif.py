"""
Full validation: train on synthetic, test on REAL — across multiple datasets and generators.
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
N = 2000

GENERATORS = {
    "gaussian_copula": GaussianCopulaGenerator,
    "vine_copula": VineCopulaGenerator,
}

DATASETS = [
    ("adult", "income"),
    ("credit", "default_payment"),
]


def prepare(df, target, feature_cols):
    X = pd.DataFrame(df[feature_cols]).copy()
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c]).codes
    y = pd.Categorical(df[target]).codes
    return X.values, np.array(y)


results = []

for dataset_id, target in DATASETS:
    real = load_dataset(dataset_id, n=N)
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

    for gen_name, gen_cls in GENERATORS.items():
        print(f"{dataset_id} | {gen_name}...", end=" ", flush=True)

        gen = gen_cls()
        gen.fit(real)
        syn = gen.generate(len(real), seed=SEED).drop(
            columns=["syn_id"], errors="ignore"
        )

        hif_result = hif_score(
            real, syn, columns=all_cols, hif_hubs=5, random_state=SEED, verbose=False
        )
        rp = hif_result["row_penalties"]

        X_syn, y_syn = prepare(syn, target, feature_cols)

        # Train on ALL
        clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42)
        clf.fit(X_syn, y_syn)
        acc_all = clf.score(X_real, y_real)

        # Train on HIF-clean (bottom 50%)
        clean_mask = rp <= np.median(rp)
        clf_clean = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, random_state=42
        )
        clf_clean.fit(X_syn[clean_mask], y_syn[clean_mask])
        acc_clean = clf_clean.score(X_real, y_real)

        # Train on HIF-clean (bottom 25%)
        p75 = np.percentile(rp, 75)
        clean_75 = rp <= p75
        clf_75 = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, random_state=42
        )
        clf_75.fit(X_syn[clean_75], y_syn[clean_75])
        acc_75 = clf_75.score(X_real, y_real)

        delta_50 = (acc_clean - acc_all) * 100
        delta_25 = (acc_75 - acc_all) * 100

        results.append(
            {
                "dataset": dataset_id,
                "generator": gen_name,
                "hif_score": hif_result["hif_score"],
                "violation_rate": hif_result["violation_rate"],
                "acc_all": acc_all,
                "acc_clean_50": acc_clean,
                "acc_clean_25": acc_75,
                "delta_50": delta_50,
                "delta_25": delta_25,
            }
        )

        print(
            f"ALL={acc_all:.4f} CLEAN50={acc_clean:.4f}({delta_50:+.2f}%) CLEAN25={acc_75:.4f}({delta_25:+.2f}%)"
        )

print("\n\n=== SUMMARY ===")
df = pd.DataFrame(results)
print(df.to_string(index=False))
print(f"\nMean delta (50% filter): {df['delta_50'].mean():+.2f}%")
print(f"Mean delta (25% filter): {df['delta_25'].mean():+.2f}%")
print(f"Improvement rate (50%): {(df['delta_50'] > 0).mean():.0%}")
print(f"Improvement rate (25%): {(df['delta_25'] > 0).mean():.0%}")
