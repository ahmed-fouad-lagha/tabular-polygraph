"""
Experiment: Full Benchmark — HIF vs Standard Metrics Across Datasets & Generators.

Runs N datasets × M generators × S seeds, computing:
  1. Standard metrics via sdmetrics: BoundaryAdherence, KSComplement, CSTest,
     CorrelationSimilarity, ContingencySimilarity
  2. HIF metrics: hif_score, violation_rate, component scores
  3. Downstream utility: TSTR F1 (full vs HIF-filtered)
  4. Minor-class F1: F1 on minority class specifically

Produces tables suitable for the paper's main experimental results.

Run:
    python scripts/10_full_benchmark.py --rows 2000 --seeds 3 --epochs 100
"""

from __future__ import annotations

import argparse
import signal
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))

from tabular_polygraph.dataset import load_dataset  # noqa: E402
from tabular_polygraph.fidelity import hif_score  # noqa: E402
from tabular_polygraph.generators import (  # noqa: E402
    CTGANGenerator,
    GaussianCopulaGenerator,
    TVAEGenerator,
)
from tabular_polygraph.utils import numeric_columns  # noqa: E402

try:
    from tabular_polygraph.generators import VineCopulaGenerator

    HAS_VINE = True
except ImportError:
    HAS_VINE = False

try:
    from sdmetrics.single_table import (
        BoundaryAdherence,
        ContingencySimilarity,
        CorrelationSimilarity,
        CSTest,
        KSComplement,
        LogisticDetection,
        SVCDetection,
    )

    HAS_SDMETRICS = True
except ImportError:
    HAS_SDMETRICS = False


# ---------------------------------------------------------------------------
# Dataset configs: (dataset_id, target_col, task)
# ---------------------------------------------------------------------------

DATASETS = [
    ("adult", "income", "classification"),
    ("credit", "default_payment", "classification"),
    ("census_acs", "employment_status", "classification"),
    ("online_purchases", "item_total", "regression"),
    ("supermarket_sales", "total", "regression"),
]

GENERATORS = {
    "gaussian": lambda epochs=100: GaussianCopulaGenerator(),
    "ctgan": lambda epochs=100: CTGANGenerator(epochs=epochs),
    "tvae": lambda epochs=100: TVAEGenerator(epochs=epochs),
}

if HAS_VINE:
    GENERATORS["vine"] = lambda epochs=100: VineCopulaGenerator()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _compute_sdmetrics(real: pd.DataFrame, syn: pd.DataFrame) -> dict:
    """Compute sdmetrics standard metrics."""
    if not HAS_SDMETRICS:
        return {}

    results = {}
    metadata = {
        "columns": {},
        "primary_key": None,
    }
    for col in real.columns:
        if pd.api.types.is_numeric_dtype(real[col]):
            metadata["columns"][col] = {"sdtype": "numerical"}
        else:
            metadata["columns"][col] = {"sdtype": "categorical"}

    try:
        results["boundary_adherence"] = BoundaryAdherence.compute(
            real_data=real, synthetic_data=syn, metadata=metadata
        )
    except Exception:
        results["boundary_adherence"] = np.nan

    try:
        results["ks_complement"] = KSComplement.compute(
            real_data=real, synthetic_data=syn, metadata=metadata
        )
    except Exception:
        results["ks_complement"] = np.nan

    try:
        results["cs_test"] = CSTest.compute(
            real_data=real, synthetic_data=syn, metadata=metadata
        )
    except Exception:
        results["cs_test"] = np.nan

    try:
        results["correlation_similarity"] = CorrelationSimilarity.compute(
            real_data=real, synthetic_data=syn, metadata=metadata
        )
    except Exception:
        results["correlation_similarity"] = np.nan

    try:
        results["contingency_similarity"] = ContingencySimilarity.compute(
            real_data=real, synthetic_data=syn, metadata=metadata
        )
    except Exception:
        results["contingency_similarity"] = np.nan

    try:
        results["logistic_detection"] = LogisticDetection.compute(
            real_data=real, synthetic_data=syn, metadata=metadata
        )
    except Exception:
        results["logistic_detection"] = np.nan

    try:
        results["svc_detection"] = SVCDetection.compute(
            real_data=real, synthetic_data=syn, metadata=metadata
        )
    except Exception:
        results["svc_detection"] = np.nan

    return results


def _compute_hif_metrics(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    columns: list[str],
    seed: int,
) -> dict:
    """Compute HIF metrics."""
    result = hif_score(real, syn, columns=columns, random_state=seed, verbose=False)
    return {
        "hif_score": float(result["hif_score"]),
        "violation_rate": float(result["violation_rate"]),
        "lse_violation_rate": float(result["lse_violation_rate"]),
        "nic_violation_rate": float(result.get("nic_violation_rate", 0.0)),
        "rule_violation_rate": float(result.get("rule_violation_rate", 0.0)),
        "mean_penalty": float(result.get("mean_penalty", 0.0)),
    }


def _compute_utility(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target: str,
    seed: int,
    num_cols: list[str],
    cat_cols: list[str],
) -> dict:
    """Compute TSTR F1 macro."""
    real_util = real.copy()
    syn_util = syn.copy()
    encoded_cols = []

    for col in cat_cols:
        if col == target or col not in syn.columns:
            continue
        if real[col].nunique() > 50:
            continue
        dummies = pd.get_dummies(real[col], prefix=f"ohe__{col}").astype(float)
        real_util = pd.concat([real_util, dummies], axis=1)
        syn_dummies = pd.get_dummies(syn[col], prefix=f"ohe__{col}").astype(float)
        for d_col in dummies.columns:
            syn_util[d_col] = (
                syn_dummies[d_col] if d_col in syn_dummies.columns else 0.0
            )
        encoded_cols.extend(dummies.columns)

    feature_cols = [
        c for c in num_cols if c != target and c in syn.columns
    ] + encoded_cols
    if not feature_cols:
        return {"f1_macro": np.nan}

    u_real, u_syn = real_util.copy(), syn_util.copy()
    if pd.api.types.is_numeric_dtype(u_real[target]) and u_real[target].nunique() > 2:
        m = u_real[target].median()
        u_real[target] = (u_real[target] > m).astype(int)
        u_syn[target] = (u_syn[target] > m).astype(int)

    y_real = u_real[target].values
    y_syn = u_syn[target].values

    if len(np.unique(y_real)) < 2 or len(np.unique(y_syn)) < 2:
        return {"f1_macro": np.nan}

    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    X_syn = u_syn[feature_cols].fillna(0).values
    X_real = u_real[feature_cols].fillna(0).values
    clf.fit(X_syn, y_syn)
    preds = clf.predict(X_real)
    return {
        "f1_macro": float(f1_score(y_real, preds, average="macro", zero_division=0.0))
    }


def _compute_minority_f1(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target: str,
    seed: int,
    num_cols: list[str],
    cat_cols: list[str],
) -> dict:
    """Compute TSTR F1 on minority class specifically."""
    real_util = real.copy()
    syn_util = syn.copy()
    encoded_cols = []

    for col in cat_cols:
        if col == target or col not in syn.columns:
            continue
        if real[col].nunique() > 50:
            continue
        dummies = pd.get_dummies(real[col], prefix=f"ohe__{col}").astype(float)
        real_util = pd.concat([real_util, dummies], axis=1)
        syn_dummies = pd.get_dummies(syn[col], prefix=f"ohe__{col}").astype(float)
        for d_col in dummies.columns:
            syn_util[d_col] = (
                syn_dummies[d_col] if d_col in syn_dummies.columns else 0.0
            )
        encoded_cols.extend(dummies.columns)

    feature_cols = [
        c for c in num_cols if c != target and c in syn.columns
    ] + encoded_cols
    if not feature_cols:
        return {
            "minority_f1": np.nan,
            "minority_class": None,
            "minority_fraction": np.nan,
        }

    u_real, u_syn = real_util.copy(), syn_util.copy()
    if pd.api.types.is_numeric_dtype(u_real[target]) and u_real[target].nunique() > 2:
        m = u_real[target].median()
        u_real[target] = (u_real[target] > m).astype(int)
        u_syn[target] = (u_syn[target] > m).astype(int)

    y_real = u_real[target].values
    y_syn = u_syn[target].values

    if len(np.unique(y_real)) < 2 or len(np.unique(y_syn)) < 2:
        return {
            "minority_f1": np.nan,
            "minority_class": None,
            "minority_fraction": np.nan,
        }

    classes, counts = np.unique(y_real, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    minority_fraction = float(counts.min() / counts.sum())

    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    X_syn = u_syn[feature_cols].fillna(0).values
    X_real = u_real[feature_cols].fillna(0).values
    clf.fit(X_syn, y_syn)
    preds = clf.predict(X_real)

    try:
        f1 = float(
            f1_score(
                y_real,
                preds,
                labels=[minority_class],
                average="binary",
                zero_division=0.0,
            )
        )
    except ValueError:
        f1 = float(f1_score(y_real, preds, average="macro", zero_division=0.0))
    return {
        "minority_f1": f1,
        "minority_class": str(minority_class),
        "minority_fraction": minority_fraction,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Generator execution timed out")


def run_benchmark(
    datasets: list[tuple[str, str, str]],
    generator_names: list[str],
    n_seeds: int,
    n_rows: int,
    epochs: int,
    timeout_s: int,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    total = len(datasets) * len(generator_names) * n_seeds
    done = 0

    for ds_id, target, task in datasets:
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {ds_id} (target={target}, task={task})")
        print(f"{'=' * 70}")

        try:
            real_full = load_dataset(ds_id)
        except Exception as e:
            print(f"  SKIP (load error: {e})")
            continue

        drop_cols = [c for c in real_full.columns if real_full[c].isnull().mean() > 0.3]
        real_full = real_full.drop(columns=drop_cols, errors="ignore")

        num_cols = [c for c in numeric_columns(real_full) if c in real_full.columns]
        cat_cols = [c for c in real_full.columns if c not in num_cols]

        for gen_name in generator_names:
            if gen_name not in GENERATORS:
                print(f"\n  -- {gen_name} (skipped: not available) --")
                continue

            print(f"\n  -- {gen_name} --")

            for seed_i in range(n_seeds):
                seed = 42 + seed_i
                done += 1
                t0 = time.time()
                print(
                    f"    [{done}/{total}] {ds_id} | {gen_name} | seed={seed}: fitting...",
                    end="",
                    flush=True,
                )

                if timeout_s > 0 and hasattr(signal, "SIGALRM"):
                    signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(timeout_s)

                try:
                    real = real_full.sample(
                        min(n_rows, len(real_full)), random_state=seed
                    ).reset_index(drop=True)

                    gen = GENERATORS[gen_name](epochs=epochs)
                    gen.fit(real)
                    print(" sampling...", end="", flush=True)
                    syn = gen.generate(len(real), seed=seed).drop(
                        columns=["syn_id"], errors="ignore"
                    )
                    common_cols = [c for c in real.columns if c in syn.columns]
                    syn = syn[common_cols]

                    print(" metrics...", end="", flush=True)
                    sdmetrics_res = _compute_sdmetrics(real, syn)

                    hif_res = _compute_hif_metrics(real, syn, common_cols, seed)

                    util_full = _compute_utility(
                        real, syn, target, seed, num_cols, cat_cols
                    )

                    penalties = hif_score(
                        real, syn, columns=common_cols, random_state=seed, verbose=False
                    )["row_penalties"]
                    syn_filtered = syn[penalties < 0.5]
                    util_hif = (
                        _compute_utility(
                            real, syn_filtered, target, seed, num_cols, cat_cols
                        )
                        if len(syn_filtered) > 10
                        else {"f1_macro": np.nan}
                    )

                    min_full = _compute_minority_f1(
                        real, syn, target, seed, num_cols, cat_cols
                    )
                    min_hif = (
                        _compute_minority_f1(
                            real, syn_filtered, target, seed, num_cols, cat_cols
                        )
                        if len(syn_filtered) > 10
                        else {"minority_f1": np.nan}
                    )

                    if timeout_s > 0 and hasattr(signal, "SIGALRM"):
                        signal.alarm(0)

                    dt = time.time() - t0

                    row = {
                        "dataset": ds_id,
                        "target": target,
                        "task": task,
                        "generator": gen_name,
                        "seed": seed,
                        "n_rows": len(real),
                        "retention_pct": round(len(syn_filtered) / len(syn) * 100, 1),
                        "time_s": round(dt, 1),
                        **sdmetrics_res,
                        **hif_res,
                        "f1_full": util_full["f1_macro"],
                        "f1_hif": util_hif["f1_macro"],
                        "f1_delta": util_hif["f1_macro"] - util_full["f1_macro"]
                        if not np.isnan(util_hif["f1_macro"])
                        and not np.isnan(util_full["f1_macro"])
                        else np.nan,
                        "minority_f1_full": min_full["minority_f1"],
                        "minority_f1_hif": min_hif["minority_f1"],
                        "minority_class": min_full["minority_class"],
                        "minority_fraction": min_full["minority_fraction"],
                    }
                    all_rows.append(row)

                    print(
                        f" DONE ({dt:.1f}s) | HIF={hif_res['hif_score']:.3f} | "
                        f"F1_full={util_full['f1_macro']:.3f} | F1_hif={util_hif['f1_macro']:.3f}"
                    )

                except TimeoutError:
                    if timeout_s > 0 and hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                    dt = time.time() - t0
                    print(f" TIMEOUT ({dt:.1f}s >= {timeout_s}s)")
                    all_rows.append(
                        {
                            "dataset": ds_id,
                            "generator": gen_name,
                            "seed": seed,
                            "error": f"timeout ({timeout_s}s)",
                        }
                    )
                except Exception as e:
                    if timeout_s > 0 and hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                    dt = time.time() - t0
                    print(f" ERROR ({dt:.1f}s): {e}")
                    all_rows.append(
                        {
                            "dataset": ds_id,
                            "generator": gen_name,
                            "seed": seed,
                            "error": str(e),
                        }
                    )

    df = pd.DataFrame(all_rows)
    csv_path = output_dir / "full_benchmark.csv"
    df.to_csv(csv_path, index=False)

    _print_summary(df)
    _write_markdown(df, output_dir)

    return df


def _print_summary(df: pd.DataFrame):
    print(f"\n\n{'=' * 90}")
    print("  FULL BENCHMARK SUMMARY")
    print(f"{'=' * 90}")

    metric_cols = [
        "ks_complement",
        "boundary_adherence",
        "correlation_similarity",
        "hif_score",
        "violation_rate",
        "f1_full",
        "f1_hif",
        "f1_delta",
        "minority_f1_full",
        "minority_f1_hif",
    ]

    agg_dict = {}
    for m in metric_cols:
        if m in df.columns:
            agg_dict[m] = ["mean", "std"]

    if not agg_dict:
        return

    summary = df.groupby(["dataset", "generator"]).agg(agg_dict).round(4)
    print(summary.to_string())

    print("\n\n  Aggregate comparison (mean across all dataset×generator×seed):")
    for m in ["f1_full", "f1_hif", "f1_delta", "minority_f1_full", "minority_f1_hif"]:
        if m in df.columns:
            vals = df[m].dropna()
            if len(vals) > 0:
                print(f"    {m}: {vals.mean():.4f} ± {vals.std():.4f}")


def _write_markdown(df: pd.DataFrame, output_dir: Path):
    md_path = output_dir / "full_benchmark_summary.md"

    with open(md_path, "w") as f:
        f.write("# Full Benchmark Results\n\n")
        f.write(f"Datasets: {df['dataset'].nunique()} | ")
        f.write(f"Generators: {df['generator'].nunique()} | ")
        f.write(
            f"Seeds per combo: {df.groupby(['dataset', 'generator']).ngroups // df['dataset'].nunique()}\n\n"
        )

        f.write("## Per-Dataset × Generator Results (Mean ± SD)\n\n")
        f.write(
            "| Dataset | Generator | HIF Score | Viol Rate | F1 Full | F1 HIF | F1 Delta | MinF1 Full | MinF1 HIF |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|\n")

        for (ds, gen), group_df in df.groupby(["dataset", "generator"]):
            if "error" in group_df.columns and group_df["error"].notna().any():
                continue

            def fmt(col, sub_df=group_df):
                vals = sub_df[col].dropna()
                if len(vals) == 0:
                    return "N/A"
                if len(vals) == 1:
                    return f"{vals.iloc[0]:.3f}"
                return f"{vals.mean():.3f} ± {vals.std():.3f}"

            f.write(
                f"| {ds} | {gen} | {fmt('hif_score')} | {fmt('violation_rate')} "
                f"| {fmt('f1_full')} | {fmt('f1_hif')} | {fmt('f1_delta')} "
                f"| {fmt('minority_f1_full')} | {fmt('minority_f1_hif')} |\n"
            )

        f.write("\n## Statistical Significance\n\n")

        needed_cols = ["dataset", "generator", "f1_full", "f1_hif"]
        if not all(c in df.columns for c in needed_cols):
            f.write("Insufficient data for significance testing.\n")
        else:
            valid = df[needed_cols].dropna()
            if len(valid) >= 3:
                t_stat, p_val = sp_stats.ttest_rel(valid["f1_full"], valid["f1_hif"])
                f.write(
                    f"Paired t-test (F1_full vs F1_hif): t={t_stat:.3f}, p={p_val:.4f}\n"
                )

                diffs = valid["f1_hif"] - valid["f1_full"]
                ci_low, ci_high = (
                    sp_stats.t.interval(
                        0.95,
                        len(diffs) - 1,
                        loc=diffs.mean(),
                        scale=sp_stats.sem(diffs),
                    )
                    if len(diffs) > 1
                    else (np.nan, np.nan)
                )
                f.write(f"95% CI for F1 difference: [{ci_low:.4f}, {ci_high:.4f}]\n")
                f.write(f"Mean F1 difference: {diffs.mean():.4f}\n")

        f.write("\n## Aggregate Metrics\n\n")
        metric_cols = [
            "ks_complement",
            "boundary_adherence",
            "correlation_similarity",
            "hif_score",
        ]
        f.write("| Metric | Mean | SD |\n|---|---|---|\n")
        for m in metric_cols:
            if m in df.columns:
                vals = df[m].dropna()
                if len(vals) > 0:
                    f.write(f"| {m} | {vals.mean():.4f} | {vals.std():.4f} |\n")

    print(f"\n  Summary saved → {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full HIF Benchmark")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout in seconds per run (default: 180s)",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset IDs, or 'all'",
    )
    parser.add_argument(
        "--generators",
        default="all",
        help="Comma-separated generator names (gaussian,ctgan,tvae,vine), or 'all'",
    )
    args = parser.parse_args()

    if args.datasets == "all":
        datasets = DATASETS
    else:
        ds_names = [d.strip() for d in args.datasets.split(",")]
        datasets = [(d, t, task) for d, t, task in DATASETS if d in ds_names]

    if args.generators == "all":
        gen_names = list(GENERATORS.keys())
    else:
        gen_names = [g.strip() for g in args.generators.split(",")]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_benchmark(
        datasets=datasets,
        generator_names=gen_names,
        n_seeds=args.seeds,
        n_rows=args.rows,
        epochs=args.epochs,
        timeout_s=args.timeout,
        output_dir=out_dir,
    )
