"""
Example 7: Utility Filtering Audit (Generating Table 1).
Evaluates how filtering synthetic data by HIF scores improves downstream utility.
Cleaned and refactored for production-ready benchmarking.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tabular_polygraph.dataset import load_dataset  # noqa: E402
from tabular_polygraph.fidelity import hif_score  # noqa: E402
from tabular_polygraph.fidelity.logical import rule_violation_score  # noqa: E402
from tabular_polygraph.utils import numeric_columns  # noqa: E402


def discretize_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Discretizes the target column into binary labels (0/1)."""
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[target]):
        unique_vals = sorted(df[target].dropna().unique().tolist())
        if len(unique_vals) <= 2:
            if set(unique_vals).issubset({0, 1}):
                df[target] = df[target].astype(int)
            else:
                mapping = {v: i for i, v in enumerate(unique_vals)}
                df[target] = df[target].map(mapping).fillna(0).astype(int)
        elif len(unique_vals) <= 5:
            # Few unique numeric values: map to integer codes preserving order
            mapping = {v: i for i, v in enumerate(unique_vals)}
            df[target] = df[target].map(mapping).fillna(0).astype(int)
        else:
            # Use quintiles for the main manuscript experiment (5-class)
            try:
                df[target] = pd.qcut(df[target], q=5, labels=False, duplicates="drop")
                df[target] = df[target].astype(int)
            except Exception:
                # Fallback to median split if qcut fails for pathological distributions
                median_val = df[target].median()
                df[target] = (df[target] >= median_val).astype(int)
    else:
        cat_codes = df[target].astype("category").cat.codes
        if len(df[target].unique()) <= 2:
            df[target] = cat_codes.astype(int)
        else:
            median_code = cat_codes.median()
            df[target] = (cat_codes >= median_code).astype(int)
    return df


def prepare_utility_features(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target: str,
    num_cols: List[str],
    cat_cols: List[str],
    reference_df: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepares and aligns features for utility evaluation."""
    feat_cols = [c for c in num_cols + cat_cols if c != target]

    def process_df(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in[feat_cols].copy()
        for col in [c for c in cat_cols if c in df.columns]:
            df[col] = df[col].astype(str).str.strip()

        # Limit cardinality for OHE
        ohe_cols = [c for c in cat_cols if c in df.columns and df_in[c].nunique() <= 20]
        df = pd.get_dummies(df, columns=ohe_cols)
        return df.select_dtypes(include=[np.number])

    real_proc = process_df(real)
    syn_proc = process_df(syn)

    # Alignment
    ref_cols = reference_df.columns if reference_df is not None else real_proc.columns
    for df in [real_proc, syn_proc]:
        for col in set(ref_cols) - set(df.columns):
            df[col] = 0.0

    y_real = real[target].to_numpy()
    y_syn = syn[target].to_numpy()

    return real_proc[ref_cols], syn_proc[ref_cols], y_real, y_syn


def evaluate_utility(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Tuple[float, float]:
    """Trains a classifier and returns F1 and Accuracy scores."""
    if len(np.unique(y_train)) < 2:
        return np.nan, np.nan

    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    return f1_score(y_test, preds, average="macro"), accuracy_score(y_test, preds)


def run_benchmark_seed(
    seed: int,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    args: argparse.Namespace,
    num_cols: List[str],
    cat_cols: List[str],
) -> List[Dict]:
    """Runs the benchmark for a single seed."""
    from tabular_polygraph.generators.cross_sectional.gaussian_copula import (
        GaussianCopulaGenerator,
    )
    from tabular_polygraph.generators.deep.ctgan import CTGANGenerator
    from tabular_polygraph.generators.deep.forest_diffusion import (
        ForestDiffusionGenerator,
    )

    gen_map = {
        "gaussian": GaussianCopulaGenerator,
        "ctgan": CTGANGenerator,
        "forest": ForestDiffusionGenerator,
    }

    gen_class = gen_map.get(args.generator, GaussianCopulaGenerator)
    if args.generator == "ctgan":
        gen = gen_class(epochs=args.epochs)
    else:
        gen = gen_class()
    gen.fit(real_train)
    syn = gen.generate(args.rows, seed=seed).drop(columns=["syn_id"], errors="ignore")

    # Base evaluation data (X_test/y_test from REAL test set)
    X_test_df, _, y_test, _ = prepare_utility_features(
        real_test, syn, args.target, num_cols, cat_cols
    )

    results = []
    variants = {
        "Full synthetic": syn,
        "Rule-only Baseline": None,
        "HIF Oracle (Combined)": None,
    }

    # Audit logic
    audit_cols = [c for c in real_train.columns if c != args.target]
    rv = rule_violation_score(real_train, syn, columns=audit_cols, max_rules=50)
    syn_rules = syn[rv["row_violation_mask"] == 0]
    variants["Rule-only Baseline"] = syn_rules

    if not syn_rules.empty:
        hif = hif_score(
            real_train, syn_rules, columns=audit_cols, hif_hubs=5, random_state=seed
        )
        variants["HIF Oracle (Combined)"] = syn_rules.iloc[
            np.argsort(hif["row_penalties"])[: int(len(syn_rules) * 0.80)]
        ]

    for label, subset in variants.items():
        if subset is None or subset.empty:
            continue

        # Corrected: Train on Synthetic (subset), Test on Real (X_test_df)
        _, X_train_df, _, y_train = prepare_utility_features(
            real_train, subset, args.target, num_cols, cat_cols, reference_df=X_test_df
        )
        f1, acc = evaluate_utility(
            X_train_df.values, y_train, X_test_df.values, y_test, seed
        )

        if not np.isnan(f1):
            results.append(
                {
                    "variant": label,
                    "seed": seed,
                    "f1": f1,
                    "acc": acc,
                    "retention": (len(subset) / len(syn)) * 100,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cleaned HIF Utility Filtering Benchmark"
    )
    parser.add_argument("--dataset", type=str, default="bls")
    parser.add_argument("--target", type=str, default="avg_weekly_wage")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--generator", type=str, default="gaussian")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--output", type=str, default="results/table1_utility_filtering.csv"
    )
    args = parser.parse_args()

    real = load_dataset(args.dataset)
    real = discretize_target(real, args.target)

    num_cols = numeric_columns(real)
    cat_cols = [c for c in real.columns if c not in num_cols]

    real_train, real_test_full = train_test_split(real, test_size=0.3, random_state=42)
    real_test = real_test_full.sample(n=min(5000, len(real_test_full)), random_state=42)

    all_results = []
    for seed in range(42, 42 + args.seeds):
        print(f"[Seed {seed}] Processing...")
        all_results.extend(
            run_benchmark_seed(seed, real_train, real_test, args, num_cols, cat_cols)
        )

        # Incremental save
        pd.DataFrame(all_results).to_csv(args.output, index=False)

    df = pd.DataFrame(all_results)
    summary = df.groupby("variant").agg(
        {"f1": ["mean", "std"], "acc": ["mean", "std"], "retention": "mean"}
    )

    print("\n" + "=" * 60)
    print(f"BENCHMARK SUMMARY: {args.dataset.upper()} ({args.generator})")
    print(summary)
    print("=" * 60)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
