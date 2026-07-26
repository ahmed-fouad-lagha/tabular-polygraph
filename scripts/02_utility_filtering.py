"""
Experiment: Utility Filtering Audit-Downstream Utility Recovery (Target-Aware HIF Benchmark).

Q:"Does filtering out HIF-flagged rows improve downstream ML predictive performance?"

Evaluates how target-aware filtering of synthetic data by HIF scores and hard
logical constraints improves downstream machine learning utility (TSTR).

Flow:
  1. Load raw dataset and split into train/test holdouts
  2. Train tabular generator on real train data
  3. Generate synthetic data
  4. Perform target-aware HIF auditing (including target-feature dependencies)
  5. Compare downstream ML utility (classification/regression) across filtering variants

python scripts/02_utility_filtering.py --dataset census_acs --target poverty_status --generator tvae --epochs 200 --seeds 5
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
    is_numeric = pd.api.types.is_numeric_dtype(real_train[args.target])

    task = "classification" if n_unique_target <= 10 or not is_numeric else "regression"

    # Base evaluation data (X_test/y_test from REAL test set)
    X_test_df, X_train_real_df, y_test, y_train_real = prepare_utility_features(
        real_test, real_train, args.target, num_cols, cat_cols
    )

    # Discretize continuous targets into 5 equal-frequency quintiles for classification evaluation
    target_bins = None
    if is_numeric and n_unique_target > 10:
        task = "classification"
        target_bins = np.unique(np.quantile(y_train_real, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        target_bins[0] = -np.inf
        target_bins[-1] = np.inf

        y_train_real = pd.cut(
            y_train_real, bins=target_bins, labels=False, include_lowest=True
        )
        if isinstance(y_train_real, pd.Series):
            y_train_real = y_train_real.fillna(0).astype(int)
        else:
            y_train_real = np.nan_to_num(y_train_real, nan=0).astype(int)

        y_test = pd.cut(y_test, bins=target_bins, labels=False, include_lowest=True)
        if isinstance(y_test, pd.Series):
            y_test = y_test.fillna(0).astype(int)
        else:
            y_test = np.nan_to_num(y_test, nan=0).astype(int)

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

        if target_bins is not None:
            y_train = pd.cut(
                y_train, bins=target_bins, labels=False, include_lowest=True
            )
            if isinstance(y_train, pd.Series):
                y_train = y_train.fillna(0).astype(int)
            else:
                y_train = np.nan_to_num(y_train, nan=0).astype(int)

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
    parser.add_argument(
        "--dataset", type=str, default="all", help="Specific dataset to run, or 'all'"
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="all",
        help="Specific generator to run, or 'all'",
    )
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        datasets = ["census_acs", "online_purchases"]
    else:
        datasets = [args.dataset]

    if args.generator == "all":
        generators = ["gaussian", "ctgan", "tvae"]
    else:
        generators = [args.generator]

    target_map = {
        "census_acs": "poverty_status",
        "online_purchases": "item_subtotal",
        "credit": "default",
        "supermarket_sales": "Rating",
        "adult": "income",
    }

    global_summary_rows = []

    for ds in datasets:
        for gen in generators:
            args.dataset = ds
            args.generator = gen
            args.target = target_map.get(ds, "target")

            output_path = out_dir / f"utility_filtering_{ds}_{gen}.csv"

            print("\n" + "=" * 80)
            print(
                f"TARGET-AWARE UTILITY BENCHMARK: {ds.upper()} ({gen}, target={args.target})"
            )
            print("=" * 80)

            try:
                real = load_dataset(args.dataset)
                num_cols = numeric_columns(real)
                cat_cols = [c for c in real.columns if c not in num_cols]

                real_train, real_test_full = train_test_split(
                    real, test_size=0.3, random_state=42
                )
                real_test = real_test_full.sample(
                    n=min(5000, len(real_test_full)), random_state=42
                )
            except Exception as e:
                print(f"Skipping {ds} - Error loading: {e}")
                continue

            all_results = []
            for seed in range(42, 42 + args.seeds):
                print(f"  [Seed {seed}] Running target-aware audit benchmark...")
                try:
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
                except Exception as e:
                    print(f"    seed={seed} ERROR: {e}")

                # Incremental save
                if all_results:
                    pd.DataFrame(all_results).to_csv(output_path, index=False)

            if all_results:
                df = pd.DataFrame(all_results)
                task_name = df["task"].iloc[0] if not df.empty else "unknown"
                metric1_name = "f1" if task_name == "classification" else "r2"
                metric2_name = "acc" if task_name == "classification" else "rmse"

                summary = df.groupby("variant").agg(
                    {
                        "score1": ["mean", "std"],
                        "score2": ["mean", "std"],
                        "retention": "mean",
                    }
                )
                summary.columns = [
                    f"{metric1_name}_mean",
                    f"{metric1_name}_std",
                    f"{metric2_name}_mean",
                    f"{metric2_name}_std",
                    "retention_mean",
                ]

                print("\n" + "-" * 80)
                print(f"SUMMARY: {ds} | {gen}")
                print(summary.round(4))
                print("-" * 80)

                # Append to global summary
                for variant, row in summary.iterrows():
                    n_seeds = len(df[df["variant"] == variant])
                    sem_f1 = (
                        row[f"{metric1_name}_std"] / np.sqrt(n_seeds)
                        if n_seeds > 1
                        else 0.0
                    )
                    sem_acc = (
                        row[f"{metric2_name}_std"] / np.sqrt(n_seeds)
                        if n_seeds > 1
                        else 0.0
                    )

                    global_summary_rows.append(
                        {
                            "Dataset": ds,
                            "Generator": gen,
                            "Variant": variant,
                            "Retention": f"{row['retention_mean']:.1f}%",
                            "F1 (mean ± SEM)": f"{row[f'{metric1_name}_mean']:.4f} ± {sem_f1:.4f}",
                            "Acc (mean ± SEM)": f"{row[f'{metric2_name}_mean']:.4f} ± {sem_acc:.4f}",
                        }
                    )

    # Export markdown summary if we have results
    if global_summary_rows:
        md_path = out_dir / "utility_filtering_summary.md"
        with open(md_path, "w") as f:
            f.write("# Downstream Utility Filtering Summary\n\n")
            f.write(
                "| Dataset | Generator | Variant | Retention | F1 (mean ± SEM) | Acc (mean ± SEM) |\n"
            )
            f.write("|---|---|---|---|---|---|\n")
            for r in global_summary_rows:
                f.write(
                    f"| {r['Dataset']} | {r['Generator']} | {r['Variant']} | {r['Retention']} | {r['F1 (mean ± SEM)']} | {r['Acc (mean ± SEM)']} |\n"
                )
        print(f"\nGlobal markdown summary saved to: {md_path}")


if __name__ == "__main__":
    main()
