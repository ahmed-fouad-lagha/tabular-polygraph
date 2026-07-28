"""
Pilot Study: Aggregate vs Row-Level Structural Fidelity
3 datasets × 3 generators × downstream utility validation
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = Path("/home/lagha/repos/tabular-polygraph")
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from tabular_polygraph.dataset import load_dataset  # noqa: E402
from tabular_polygraph.fidelity import (  # noqa: E402
    correlation_distance_score,
    hif_score,
    mean_moment_matching_score,
    moment_matching_scores,
)
from tabular_polygraph.fidelity.alpha_beta import (  # noqa: E402
    alpha_precision_beta_recall,
)
from tabular_polygraph.generators import (  # noqa: E402
    GaussianCopulaGenerator,
    VineCopulaGenerator,
)
from tabular_polygraph.generators.base import BaseGenerator  # noqa: E402

warnings.filterwarnings("ignore")

DATASETS = ["adult", "credit", "supermarket_sales"]
GENERATORS = ["gaussian_copula", "vine_copula"]
ROWS = 3000
SEED = 42


def make_generator(name: str) -> BaseGenerator:
    if name == "gaussian_copula":
        return GaussianCopulaGenerator()
    elif name == "vine_copula":
        return VineCopulaGenerator()
    raise ValueError(name)


def aggregate_score(real: pd.DataFrame, syn: pd.DataFrame, num_cols: list[str]) -> dict:
    corr = correlation_distance_score(real, syn, num_cols)
    mm = moment_matching_scores(real, syn, num_cols)
    mm_mean = mean_moment_matching_score(mm)
    ks_scores = []
    for col in num_cols:
        s, _ = ks_2samp(real[col].dropna(), syn[col].dropna())
        ks_scores.append(1 - s)
    ks_mean = np.mean(ks_scores) if ks_scores else 0
    try:
        real_sub = real.sample(n=len(syn), random_state=42)
        ab = alpha_precision_beta_recall(real_sub, syn)
        alpha = ab.get("alpha_precision", 0)
    except Exception:
        alpha = 0
    composite = (
        0.3 * (corr / 100) + 0.25 * (mm_mean / 100) + 0.25 * alpha + 0.2 * ks_mean
    )
    return {
        "composite": round(composite, 4),
        "corr_dist": round(corr, 2),
        "ks_sim": round(ks_mean, 4),
        "alpha": round(alpha, 4),
    }


def downstream_utility(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target: str,
    syn_filtered: pd.DataFrame | None = None,
) -> dict:
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
        X = df[feature_cols].copy()
        for c in cat_cols:
            X[c] = pd.Categorical(X[c]).codes
        y = pd.Categorical(df[target]).codes
        return X.values, y

    X_real, y_real = prepare(real)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=4)

    real_acc = cross_val_score(clf, X_real, y_real, cv=3, scoring="accuracy").mean()

    X_syn, y_syn = prepare(syn)
    syn_acc = cross_val_score(clf, X_syn, y_syn, cv=3, scoring="accuracy").mean()

    if syn_filtered is not None and len(syn_filtered) > 10:
        X_filt, y_filt = prepare(syn_filtered)
        filt_acc = cross_val_score(clf, X_filt, y_filt, cv=3, scoring="accuracy").mean()
    else:
        filt_acc = None

    return {
        "real": round(real_acc, 4),
        "synthetic": round(syn_acc, 4),
        "filtered": round(filt_acc, 4) if filt_acc else None,
    }


def get_target(dataset_id: str) -> str:
    targets = {
        "adult": "income",
        "credit": "default_payment_next_month",
        "supermarket_sales": "gross_income",
    }
    return targets[dataset_id]


def run_pilot():
    print("=" * 90)
    print("PILOT STUDY: Aggregate vs Row-Level Structural Fidelity")
    print("=" * 90)

    results = []

    for dataset_id in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_id}")
        print(f"{'=' * 70}")

        real = load_dataset(dataset_id, n=5000)
        target = get_target(dataset_id)
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
        print(
            f"  Shape: {real.shape} | Num: {len(num_cols)} | Cat: {len(cat_cols)} | Target: {target}"
        )

        for gen_name in GENERATORS:
            print(f"\n  Generator: {gen_name}")

            try:
                gen = make_generator(gen_name)
                gen.fit(real)
                syn = gen.generate(ROWS, seed=SEED).drop(
                    columns=["syn_id"], errors="ignore"
                )
            except Exception as e:
                print(f"    FAILED: {e}")
                continue

            agg = aggregate_score(real, syn, num_cols)
            hif = hif_score(
                real,
                syn,
                columns=all_cols,
                hif_hubs=5,
                random_state=SEED,
                verbose=False,
            )
            rp = hif["row_penalties"]
            n_broken = int((rp > 0.5).sum())

            threshold = np.percentile(rp, 80)
            syn_good = syn[rp <= threshold]
            print(
                f"    Aggregate: {agg['composite']:.4f} | HIF: {hif['hif_score']:.4f} | "
                f"Violations: {hif['violation_rate']:.1%} | Broken rows: {n_broken}/{len(syn)}"
            )
            print(
                f"    P90 penalty: {np.percentile(rp, 90):.4f} | P99: {np.percentile(rp, 99):.4f}"
            )
            print(
                f"    Filtered: keeping {len(syn_good)}/{len(syn)} rows (top 80% by HIF)"
            )

            try:
                util = downstream_utility(real, syn, target, syn_good)
                print(
                    f"    Downstream acc — Real: {util['real']:.4f} | Synthetic: {util['synthetic']:.4f} | "
                    f"Filtered: {util['filtered']:.4f}"
                    if util["filtered"]
                    else f"    Downstream acc — Real: {util['real']:.4f} | Synthetic: {util['synthetic']:.4f}"
                )
            except Exception as e:
                util = {
                    "real": None,
                    "synthetic": None,
                    "filtered": None,
                    "error": str(e),
                }
                print(f"    Downstream FAILED: {e}")

            results.append(
                {
                    "dataset": dataset_id,
                    "generator": gen_name,
                    "aggregate": agg["composite"],
                    "hif": hif["hif_score"],
                    "violation_rate": hif["violation_rate"],
                    "n_broken": n_broken,
                    "p90": round(float(np.percentile(rp, 90)), 4),
                    "real_acc": util.get("real"),
                    "syn_acc": util.get("synthetic"),
                    "filt_acc": util.get("filtered"),
                }
            )

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    out = PROJECT_ROOT / "outputs" / "pilot_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    print("\n--- KEY QUESTIONS ---")
    print(
        "1. Does aggregate score stay flat while HIF detects corruption? (repeat experiment2_adversarial.py per dataset)"
    )
    print("2. Does filtering by HIF improve downstream accuracy?")
    print("3. Is the pattern consistent across datasets and generators?")


if __name__ == "__main__":
    run_pilot()
