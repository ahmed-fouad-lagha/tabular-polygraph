"""
Example 7: Utility Filtering Audit (Generating Table 1).
Evaluates how filtering synthetic data by HIF scores improves downstream utility.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tabular_polygraph.catalog import load_dataset  # noqa: E402
from tabular_polygraph.fidelity import hif_score  # noqa: E402
from tabular_polygraph.fidelity.logical import rule_violation_score  # noqa: E402
from tabular_polygraph.utils import numeric_columns  # noqa: E402


def get_utility_features(real, syn, target, num_cols, cat_cols, reference_df=None):
    # Ensure target is present
    if target not in real.columns:
        raise ValueError(f"Target {target} not in real columns: {real.columns}")

    # Prepare dataframes (Exclude target from features)
    feat_cols = [c for c in num_cols + cat_cols if c != target]
    real_util = real[feat_cols].copy()
    syn_util = syn[feat_cols].copy()

    # Whitespace stripping for categorical feature columns only (target excluded).
    for col in cat_cols:
        if col == target or col not in real_util.columns or col not in syn_util.columns:
            continue
        real_util[col] = real_util[col].astype(str).str.strip()
        syn_util[col] = syn_util[col].astype(str).str.strip()

    # Isolate target labels robustly (avoid unseen-category collapse to -1).
    y_real_num = pd.to_numeric(real[target], errors="coerce")
    y_syn_num = pd.to_numeric(syn[target], errors="coerce")
    if y_real_num.notna().mean() > 0.9 and y_syn_num.notna().mean() > 0.9:
        y_real = y_real_num.fillna(0).astype(int).to_numpy()
        y_syn = y_syn_num.fillna(0).astype(int).to_numpy()
    else:
        all_cats = pd.Index(real[target].astype(str)).union(
            pd.Index(syn[target].astype(str))
        )
        y_real = pd.Categorical(real[target].astype(str), categories=all_cats).codes
        y_syn = pd.Categorical(syn[target].astype(str), categories=all_cats).codes

    # Filter features for OHE (exclude target, limit cardinality)
    ohe_feats = [c for c in cat_cols if c != target and real[c].nunique() <= 20]

    # One-Hot Encode
    real_util = pd.get_dummies(real_util, columns=ohe_feats)
    syn_util = pd.get_dummies(syn_util, columns=ohe_feats)

    # Drop any remaining non-numeric columns
    real_util = real_util.select_dtypes(include=[np.number])
    syn_util = syn_util.select_dtypes(include=[np.number])

    # Alignment
    if reference_df is not None:
        missing_real = set(reference_df.columns) - set(real_util.columns)
        for c in missing_real:
            real_util[c] = 0.0
        real_util = real_util[reference_df.columns]
        missing = set(reference_df.columns) - set(syn_util.columns)
        for c in missing:
            syn_util[c] = 0.0
        syn_util = syn_util[reference_df.columns]

    return real_util, syn_util, real_util.columns.tolist(), y_real, y_syn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="census_acs")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--target", type=str, default="employment_status")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--generator", type=str, default="forest")
    args = parser.parse_args()

    # Load Data
    real = load_dataset(args.dataset)

    # Global binary target normalization (before split).
    # If already binary, preserve classes exactly to avoid one-class collapse.
    if pd.api.types.is_numeric_dtype(real[args.target]):
        uniq = sorted(pd.Series(real[args.target]).dropna().unique().tolist())
        if len(uniq) <= 2:
            if set(uniq).issubset({0, 1}):
                real[args.target] = real[args.target].astype(int)
            else:
                mapping = {v: i for i, v in enumerate(uniq)}
                real[args.target] = real[args.target].map(mapping).fillna(0).astype(int)
        else:
            m = real[args.target].median()
            real[args.target] = (real[args.target] >= m).astype(int)
    else:
        cat = real[args.target].astype("category")
        n_classes = len(cat.cat.categories)
        codes = cat.cat.codes
        if n_classes <= 2:
            real[args.target] = codes.astype(int)
        else:
            m = codes.median()
            real[args.target] = (codes >= m).astype(int)

    print(
        f"[Pipeline] Discretized target '{args.target}'. Real dist: {real[args.target].value_counts().to_dict()}"
    )

    # Recompute feature types after target normalization.
    num_cols = [c for c in numeric_columns(real) if c in real.columns]
    cat_cols = [c for c in real.columns if c not in num_cols]

    # Stable Train/Test Split
    real_train, real_test_full = train_test_split(real, test_size=0.3, random_state=42)

    results = []

    for seed in range(42, 42 + args.seeds):
        print(f"\n[Seed {seed}] Generating synthetic data...", flush=True)

        # Generator initialization and fit
        if args.generator == "ctgan":
            from tabular_polygraph.generators.deep.ctgan import CTGANGenerator

            gen = CTGANGenerator()
        elif args.generator == "forest":
            from tabular_polygraph.generators.deep.forest_diffusion import (
                ForestDiffusionGenerator,
            )

            gen = ForestDiffusionGenerator()
        elif args.generator == "gaussian":
            from tabular_polygraph.generators.cross_sectional.gaussian_copula import (
                GaussianCopulaGenerator,
            )

            gen = GaussianCopulaGenerator()
        else:
            from tabular_polygraph.generators.cross_sectional.vine_copula import (
                VineCopulaGenerator,
            )

            gen = VineCopulaGenerator()

        gen.fit(real_train)
        syn = gen.generate(args.rows, seed=seed)
        syn = syn.drop(columns=["syn_id"], errors="ignore")

        audit_cols = [c for c in (cat_cols + num_cols) if c != args.target]

        # Strip whitespace
        for col in cat_cols:
            syn[col] = syn[col].astype(str).str.strip()

        # Prepare evaluation set once per seed
        eval_size = min(5000, len(real_test_full))
        r_test_raw = real_test_full.sample(n=eval_size, random_state=42)

        # Get encoded features for evaluation
        X_test_df, _, feat, y_test, _ = get_utility_features(
            r_test_raw, syn, args.target, num_cols, cat_cols, reference_df=None
        )
        X_test = X_test_df.values

        def eval_subset(
            syn_subset_raw,
            label,
            cur_seed=seed,
            cur_X_test=X_test,
            cur_y_test=y_test,
            cur_ref_df=X_test_df,
        ):
            _, X_train_df, _, _, y_train = get_utility_features(
                real_train,
                syn_subset_raw,
                args.target,
                num_cols,
                cat_cols,
                reference_df=cur_ref_df,
            )
            X_train = X_train_df.values
            y_unique = np.unique(y_train)
            if len(y_unique) < 2:
                print(
                    f"  [warn] {label}: skipped (single class in synthetic target: {y_unique.tolist()})"
                )
                return np.nan, np.nan

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, f1_score

            clf = RandomForestClassifier(
                n_estimators=100, max_depth=None, random_state=cur_seed, n_jobs=-1
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(cur_X_test)
            f1 = f1_score(cur_y_test, preds, average="macro")
            acc = accuracy_score(cur_y_test, preds)
            return f1, acc

        # 1. Full Synthetic
        f1_full, acc_full = eval_subset(syn, "Full")
        if not np.isnan(f1_full):
            results.append(
                {
                    "variant": "Full synthetic",
                    "seed": seed,
                    "retention": 100.0,
                    "f1": f1_full,
                    "acc": acc_full,
                }
            )
            print(f"  Full Synthetic: F1={f1_full:.4f} Acc={acc_full:.4f}")

        # 2. Rule-only Baseline
        rv = rule_violation_score(real_train, syn, columns=audit_cols, max_rules=50)
        mask_rules = rv["row_violation_mask"] == 0
        syn_rules = syn[mask_rules]
        ret_rules = len(syn_rules) / len(syn) * 100
        if not syn_rules.empty:
            f1_rules, acc_rules = eval_subset(syn_rules, "Rule-only")
            if not np.isnan(f1_rules):
                results.append(
                    {
                        "variant": "Rule-only Baseline",
                        "seed": seed,
                        "retention": ret_rules,
                        "f1": f1_rules,
                        "acc": acc_rules,
                    }
                )
                print(
                    f"  Rule-only:      F1={f1_rules:.4f} Acc={acc_rules:.4f} (Ret={ret_rules:.1f}%)"
                )

        # 3. HIF Oracle (Adaptive Frontier: 80% Retention)
        # Apply NIC on the logically valid subset
        if syn_rules.empty:
            print("  [warn] HIF Oracle skipped (empty rule-valid subset).")
            continue

        hif = hif_score(
            real_train, syn_rules, columns=audit_cols, hif_hubs=5, random_state=seed
        )

        # Adaptive Frontier: Retain top 80% of the logically valid rows to preserve diversity
        retention_ratio = 0.80
        syn_hif = syn_rules.iloc[
            np.argsort(hif["row_penalties"])[: int(len(syn_rules) * retention_ratio)]
        ]

        if not syn_hif.empty:
            f1_hif, acc_hif = eval_subset(syn_hif, "HIF Oracle")
            if not np.isnan(f1_hif):
                results.append(
                    {
                        "variant": "HIF Oracle (Combined)",
                        "seed": seed,
                        "retention": (len(syn_hif) / len(syn)) * 100,
                        "f1": f1_hif,
                        "acc": acc_hif,
                    }
                )
                print(
                    f"  HIF Oracle:     F1={f1_hif:.4f} Acc={acc_hif:.4f} (Ret={(len(syn_hif) / len(syn)) * 100:.1f}%)"
                )

    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError(
            "No valid utility results were produced. Check target distribution and generator output."
        )
    summary = df.groupby("variant").agg(
        {"f1": ["mean", "std"], "acc": ["mean", "std"], "retention": "mean"}
    )
    print("\n" + "=" * 50)
    print(f"RESULTS SUMMARY (N={args.seeds} seeds, {args.rows} rows)")
    print(summary)
    print("=" * 50)

    output_path = Path("results/table1_utility_filtering.csv")
    df.to_csv(output_path, index=False)
    print(f"\nRaw results saved to {output_path}")


if __name__ == "__main__":
    main()
