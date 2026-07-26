"""
Experiment: Violation-Aware vs Geometric vs Random Filtering.

Tests: Does knowing WHAT's wrong (HIF) beat knowing HOW CLOSE you are (Chamfer)?
Only Gaussian Copula (fast). CTGAN takes too long.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabular_polygraph.dataset import load_dataset
from tabular_polygraph.fidelity import hif_score
from tabular_polygraph.generators import GaussianCopulaGenerator
from tabular_polygraph.utils import numeric_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_real_data(dataset_id):
    df = load_dataset(dataset_id)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df.reset_index(drop=True)


def encode_for_distance(real, syn, numeric_cols):
    real_enc = real.copy()
    syn_enc = syn.copy()
    for col in [c for c in real.columns if c not in numeric_cols]:
        cats = sorted(set(real[col].unique()) | set(syn[col].unique()))
        for df in [real_enc, syn_enc]:
            for cat in cats:
                df[f"{col}_{cat}"] = (df[col] == cat).astype(float)
            df.drop(columns=[col], inplace=True, errors="ignore")
    num_cols = [
        c for c in real_enc.columns if pd.api.types.is_numeric_dtype(real_enc[c])
    ]
    scaler = StandardScaler()
    return (
        scaler.fit_transform(real_enc[num_cols].fillna(0)),
        scaler.transform(syn_enc[num_cols].fillna(0)),
    )


def chamfer_filter(real_vals, syn_vals, keep_frac):
    dists = cdist(syn_vals, real_vals, metric="euclidean")
    min_dists = dists.min(axis=1)
    return np.argsort(min_dists)[: int(len(syn_vals) * keep_frac)]


def random_filter(n, keep_frac, seed):
    return np.random.RandomState(seed).choice(n, size=int(n * keep_frac), replace=False)


def hif_filter(real, syn, cols, keep_frac, seed):
    res = hif_score(real, syn, columns=cols, random_state=seed, verbose=False)
    return np.argsort(res["row_penalties"])[: int(len(syn) * keep_frac)], res


def tstr_f1(real_df, syn_df, target, seed):
    feat_cols = [c for c in syn_df.columns if c != target]

    def prepare(df):
        X = df[feat_cols].copy()
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.Categorical(X[col]).codes
        return X.values

    y_syn = syn_df[target].values
    y_real = real_df[target].values
    if len(np.unique(y_syn)) < 2 or len(np.unique(y_real)) < 2:
        return np.nan
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(prepare(syn_df), y_syn)
    return f1_score(y_real, clf.predict(prepare(real_df)), average="macro")


def discretize_target(df, target):
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[target]):
        vals = sorted(df[target].dropna().unique())
        if len(vals) <= 2:
            df[target] = (
                df[target].map({v: i for i, v in enumerate(vals)}).fillna(0).astype(int)
            )
        else:
            df[target] = pd.qcut(df[target], q=5, labels=False, duplicates="drop")
    else:
        codes = df[target].astype("category").cat.codes
        df[target] = (codes >= codes.median()).astype(int)
    return df


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {"census_acs": "employment_status"}
    fractions = [0.5, 0.7, 0.9]
    seeds = 5

    print("=" * 60)
    print("  FILTERING COMPARISON: HIF vs Chamfer vs Random")
    print("=" * 60)

    all_results = []

    for dataset_id, target in datasets.items():
        print(f"\n── {dataset_id} ──")
        real_df = load_real_data(dataset_id)
        real_df = discretize_target(real_df, target)
        numeric_cols = numeric_columns(real_df)
        real_train, real_test = train_test_split(
            real_df, test_size=0.3, random_state=42
        )

        gen = GaussianCopulaGenerator()
        gen.fit(real_train)

        for seed in range(42, 42 + seeds):
            print(f"  Seed {seed}...", end="", flush=True)
            t0 = time.time()

            syn = gen.generate(2000, seed=seed).drop(
                columns=["syn_id"], errors="ignore"
            )
            hif_cols = list(real_train.columns)
            real_vals, syn_vals = encode_for_distance(real_train, syn, numeric_cols)

            for frac in fractions:
                keep_hif, hif_res = hif_filter(real_train, syn, hif_cols, frac, seed)
                keep_chamfer = chamfer_filter(real_vals, syn_vals, frac)
                keep_random = random_filter(len(syn), frac, seed)

                for method, keep in [
                    ("Full", np.arange(len(syn))),
                    ("HIF", keep_hif),
                    ("Chamfer", keep_chamfer),
                    ("Random", keep_random),
                ]:
                    f1 = tstr_f1(
                        real_test, syn.iloc[keep].reset_index(drop=True), target, seed
                    )
                    all_results.append(
                        {
                            "dataset": dataset_id,
                            "seed": seed,
                            "keep_frac": frac,
                            "method": method,
                            "f1": f1,
                            "n_retained": len(keep),
                            "violation_rate": hif_res["violation_rate"]
                            if method == "HIF"
                            else None,
                        }
                    )

            print(f" {time.time() - t0:.0f}s")

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    for ds in df["dataset"].unique():
        print(f"\n  {ds}:")
        dd = df[df["dataset"] == ds]
        pivot = dd.groupby(["keep_frac", "method"])["f1"].agg(["mean", "std"]).round(4)
        print(pivot.to_string())

    print("\n  KEY COMPARISON (mean F1 delta vs Full):")
    for frac in fractions:
        sub = df[df["keep_frac"] == frac]
        full = sub[sub["method"] == "Full"]["f1"].mean()
        hif = sub[sub["method"] == "HIF"]["f1"].mean()
        chm = sub[sub["method"] == "Chamfer"]["f1"].mean()
        rnd = sub[sub["method"] == "Random"]["f1"].mean()
        print(f"\n  {frac:.0%} retention:")
        print(
            f"    Full={full:.4f} | HIF={hif:.4f} ({hif - full:+.4f}) | Chamfer={chm:.4f} ({chm - full:+.4f}) | Random={rnd:.4f}"
        )

    df.to_csv(out_dir / "filtering_comparison.csv", index=False)
    print(f"\n  Saved: {out_dir}/filtering_comparison.csv")


if __name__ == "__main__":
    main()
