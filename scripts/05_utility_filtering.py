"""
Experiment: Utility Filtering Audit (Target-Aware HIF Benchmark).

Evaluates how target-aware filtering of synthetic data by HIF scores and hard
logical constraints improves downstream machine learning utility (TSTR).

Flow:
  1. Load raw dataset and split into train/test holdouts
  2. Train tabular generator on real train data
  3. Generate synthetic data
  4. Perform target-aware HIF auditing (including target-feature dependencies)
  5. Compare downstream ML utility (classification/regression) across filtering variants
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tabular_polygraph.dataset import load_dataset  # noqa: E402
from tabular_polygraph.fidelity import hif_score  # noqa: E402
from tabular_polygraph.fidelity.logical import rule_violation_score  # noqa: E402
from tabular_polygraph.utils import numeric_columns  # noqa: E402


def prepare_utility_features(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target: str,
    num_cols: List[str],
    cat_cols: List[str],
    reference_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepares and aligns features for utility evaluation."""
    feat_cols = [c for c in num_cols + cat_cols if c != target]

    real_sub = real[feat_cols].copy()
    syn_sub = syn[feat_cols].copy()

    num_feats = [c for c in num_cols if c != target]
    cat_feats = [c for c in cat_cols if c != target]

    for col in num_feats:
        if col in real_sub.columns:
            med = real_sub[col].median()
            real_sub[col] = real_sub[col].fillna(med if not np.isnan(med) else 0.0)
            if col in syn_sub.columns:
                syn_sub[col] = syn_sub[col].fillna(med if not np.isnan(med) else 0.0)

    for col in cat_feats:
        if col in real_sub.columns:
            real_sub[col] = real_sub[col].astype(str).str.strip()
            if col in syn_sub.columns:
                syn_sub[col] = syn_sub[col].astype(str).str.strip()

    ohe_cols = [
        c for c in cat_feats if c in real_sub.columns and real_sub[c].nunique() <= 20
    ]

    n_real = len(real_sub)
    comb = pd.concat([real_sub, syn_sub], axis=0, ignore_index=True)
    comb_dummies = pd.get_dummies(comb, columns=ohe_cols)
    comb_num = comb_dummies.select_dtypes(include=[np.number])

    real_proc = comb_num.iloc[:n_real].copy().reset_index(drop=True)
    syn_proc = comb_num.iloc[n_real:].copy().reset_index(drop=True)

    if reference_df is not None:
        ref_cols = reference_df.columns
        for col in set(ref_cols) - set(syn_proc.columns):
            syn_proc[col] = 0.0
        syn_proc = syn_proc[ref_cols]

    y_real = real[target].to_numpy()
    y_syn = syn[target].to_numpy()

    return real_proc, syn_proc, y_real, y_syn


def evaluate_utility(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int,
) -> Tuple[float, float]:
    """Trains a classifier/regressor and returns metric pair (Score1, Score2).

    For classification: (f1_macro, accuracy)
    For regression: (r2_score, rmse)
    """
    if task == "classification":
        if len(np.unique(y_train)) < 2:
            return np.nan, np.nan
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        return float(f1_score(y_test, preds, average="macro")), float(
            accuracy_score(y_test, preds)
        )
    else:
        reg = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        r2 = float(r2_score(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        return r2, rmse


def run_benchmark_seed(
    seed: int,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    args: argparse.Namespace,
    num_cols: List[str],
    cat_cols: List[str],
) -> List[Dict]:
    """Runs the benchmark for a single seed with target-aware HIF auditing."""
    from tabular_polygraph.generators.ctgan import CTGANGenerator
    from tabular_polygraph.generators.gaussian_copula import (
        GaussianCopulaGenerator,
    )
    from tabular_polygraph.generators.tvae import TVAEGenerator

    gen_map = {
        "gaussian": GaussianCopulaGenerator,
        "ctgan": CTGANGenerator,
        "tvae": TVAEGenerator,
    }

    gen_class = gen_map.get(args.generator, GaussianCopulaGenerator)
    if args.generator in ("ctgan", "tvae"):
        gen = gen_class(epochs=args.epochs)
    else:
        gen = gen_class()
    gen.fit(real_train)

    syn_raw = gen.generate(args.rows, seed=seed).drop(
        columns=["syn_id"], errors="ignore"
    )

    # Infer task: classification vs regression
    n_unique_target = real_train[args.target].nunique()
    task = (
        "classification"
        if n_unique_target <= 10
        or not pd.api.types.is_numeric_dtype(real_train[args.target])
        else "regression"
    )

    # Base evaluation data (X_test/y_test from REAL test set)
    X_test_df, X_train_real_df, y_test, y_train_real = prepare_utility_features(
        real_test, real_train, args.target, num_cols, cat_cols
    )

    # TRTR Real Baseline
    score1_trtr, score2_trtr = evaluate_utility(
        X_train_real_df.values, y_train_real, X_test_df.values, y_test, task, seed
    )

    results: List[Dict] = []
    if not np.isnan(score1_trtr):
        results.append(
            {
                "variant": "TRTR Real Baseline",
                "seed": seed,
                "score1": score1_trtr,
                "score2": score2_trtr,
                "retention": 100.0,
                "task": task,
            }
        )

    # Target-Aware Audit logic: audit_cols includes ALL columns (features + target!)
    audit_cols = list(real_train.columns)

    # 1. Hard logical rule violations (including rules binding target to features!)
    rv = rule_violation_score(real_train, syn_raw, columns=audit_cols, max_rules=50)
    rule_mask = rv["row_violation_mask"]
    syn_rules_raw = syn_raw[rule_mask == 0].reset_index(drop=True)

    # 2. HIF Sentinels + NIC audit
    hif = hif_score(
        real_train,
        syn_raw,
        columns=audit_cols,
        hif_hubs=5,
        random_state=seed,
    )
    penalties = hif["row_penalties"]

    # HIF Clean: pass hard rules AND penalties <= 0.5 (true valid manifold rows)
    hif_clean_mask = (rule_mask == 0) & (penalties <= 0.5)
    syn_hif_clean = syn_raw[hif_clean_mask].reset_index(drop=True)

    # Top 80% cleanest rows
    sorted_indices = np.argsort(penalties)
    n_top = max(1, int(len(syn_raw) * 0.80))
    syn_hif_top80 = syn_raw.iloc[sorted_indices[:n_top]].reset_index(drop=True)

    variants: Dict[str, pd.DataFrame] = {
        "Full synthetic": syn_raw,
        "Rule-Filtered": syn_rules_raw,
        "HIF Clean (Threshold <= 0.5)": syn_hif_clean,
        "HIF Top-80% (Oracle)": syn_hif_top80,
    }

    for label, subset in variants.items():
        if subset is None or subset.empty:
            continue

        _, X_train_df, _, y_train = prepare_utility_features(
            real_test,
            subset,
            args.target,
            num_cols,
            cat_cols,
            reference_df=X_test_df,
        )
        score1, score2 = evaluate_utility(
            X_train_df.values, y_train, X_test_df.values, y_test, task, seed
        )

        if not np.isnan(score1):
            results.append(
                {
                    "variant": label,
                    "seed": seed,
                    "score1": score1,
                    "score2": score2,
                    "retention": (len(subset) / len(syn_raw)) * 100,
                    "task": task,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cleaned Target-Aware HIF Utility Filtering Benchmark"
    )
    parser.add_argument("--dataset", type=str, default="census_acs")
    parser.add_argument("--target", type=str, default="poverty_status")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--generator", type=str, default="tvae")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--output", type=str, default="outputs/table1_utility_filtering.csv"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    real = load_dataset(args.dataset)
    num_cols = numeric_columns(real)
    cat_cols = [c for c in real.columns if c not in num_cols]

    real_train, real_test_full = train_test_split(real, test_size=0.3, random_state=42)
    real_test = real_test_full.sample(n=min(5000, len(real_test_full)), random_state=42)

    all_results = []
    for seed in range(42, 42 + args.seeds):
        print(f"[Seed {seed}] Running target-aware audit benchmark...")
        all_results.extend(
            run_benchmark_seed(
                seed,
                real_train,
                real_test,
                args,
                num_cols,
                cat_cols,
            )
        )

        # Incremental save
        pd.DataFrame(all_results).to_csv(output_path, index=False)

    df = pd.DataFrame(all_results)
    task_name = df["task"].iloc[0] if not df.empty else "unknown"
    metric1_name = "f1" if task_name == "classification" else "r2"
    metric2_name = "acc" if task_name == "classification" else "rmse"

    summary = df.groupby("variant").agg(
        {"score1": ["mean", "std"], "score2": ["mean", "std"], "retention": "mean"}
    )
    summary.columns = [
        f"{metric1_name}_mean",
        f"{metric1_name}_std",
        f"{metric2_name}_mean",
        f"{metric2_name}_std",
        "retention_mean",
    ]

    print("\n" + "=" * 80)
    print(
        f"TARGET-AWARE UTILITY BENCHMARK: {args.dataset.upper()} ({args.generator}, target={args.target}, task={task_name})"
    )
    print("=" * 80)
    print(summary.to_string())
    print("=" * 80)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
