"""
Experiment: Full Benchmark — HIF vs Standard Metrics Across Datasets & Generators
(ported to current API).

Runs N datasets × M generators × S seeds, computing KS, alpha-precision /
beta-recall (coverage), JCD, moment-matching, and HIF metrics plus downstream
utility (full vs HIF-filtered).

Run:
    python scripts/03_cross_architecture_benchmark.py --rows 2000 --seeds 3 \
      --generators gaussian,vine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ruff: noqa: E402
from _exp_utils import (
    aggregate_metrics,
    audit_hif,
    generate,
    load_real,
    utility_metrics,
)

DATASETS = [
    ("adult", "income", "classification"),
    ("credit", "default_payment", "classification"),
    ("census_acs", "household_income", "classification"),
    ("online_purchases", "item_total", "regression"),
    ("supermarket_sales", "total", "regression"),
]


def run_benchmark(
    datasets: list[tuple[str, str, str]],
    generator_names: list[str],
    n_seeds: int,
    n_rows: int,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    total = len(datasets) * len(generator_names) * n_seeds
    done = 0

    for ds_id, target, task in datasets:
        print(f"\n{'=' * 70}\n  Dataset: {ds_id} (target={target})\n{'=' * 70}")
        try:
            real_full = load_real(ds_id, n=n_rows)
        except Exception as e:
            print(f"  SKIP (load error: {e})")
            continue

        real = real_full.reset_index(drop=True)

        for gen_name in generator_names:
            print(f"\n  -- {gen_name} --")
            for seed_i in range(n_seeds):
                seed = 42 + seed_i
                done += 1
                print(
                    f"    [{done}/{total}] {ds_id} | {gen_name} | seed={seed}: ...",
                    end="",
                    flush=True,
                )

                try:
                    syn = generate(real, len(real), seed, gen_name)
                except Exception as e:
                    print(f" generate failed: {e}")
                    continue
                syn = syn[real.columns.intersection(syn.columns).tolist()]

                agg = aggregate_metrics(real, syn)
                hif_res = audit_hif(real, syn, seed=seed)

                util_full = utility_metrics(real, syn, target, seed)
                syn_filtered = syn[hif_res["row_penalties"] < 0.5]
                util_hif = (
                    utility_metrics(real, syn_filtered, target, seed)
                    if len(syn_filtered) > 10
                    else {"f1": np.nan, "accuracy": np.nan, "trr": np.nan}
                )
                retention = len(syn_filtered) / len(syn) * 100

                all_rows.append(
                    {
                        "dataset": ds_id,
                        "target": target,
                        "task": task,
                        "generator": gen_name,
                        "seed": seed,
                        "n_rows": len(real),
                        "retention_pct": round(retention, 1),
                        "ks": agg.get("ks", np.nan),
                        "jcd": agg.get("jcd", np.nan),
                        "mm": agg.get("mm", np.nan),
                        "tvd": agg.get("tvd", np.nan),
                        "alpha_precision": agg.get("alpha_precision", np.nan),
                        "beta_recall": agg.get("beta_recall", np.nan),
                        "hif_score": hif_res["hif_score"],
                        "violation_rate": hif_res["violation_rate"],
                        "lse_violation_rate": hif_res["lse_violation_rate"],
                        "nic_violation_rate": hif_res.get("nic_violation_rate", 0.0),
                        "rule_violation_rate": hif_res.get("rule_violation_rate", 0.0),
                        "f1_full": util_full["f1"],
                        "f1_hif": util_hif["f1"],
                    }
                )
                print(" done.")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "full_benchmark.csv", index=False)

    summary = (
        df.groupby(["dataset", "generator"])
        .agg(
            ks_mean=("ks", "mean"),
            ks_std=("ks", "std"),
            alpha_mean=("alpha_precision", "mean"),
            alpha_std=("alpha_precision", "std"),
            beta_mean=("beta_recall", "mean"),
            beta_std=("beta_recall", "std"),
            hif_mean=("hif_score", "mean"),
            hif_std=("hif_score", "std"),
            viol_mean=("violation_rate", "mean"),
            viol_std=("violation_rate", "std"),
            f1_full_mean=("f1_full", "mean"),
            f1_full_std=("f1_full", "std"),
            f1_hif_mean=("f1_hif", "mean"),
            f1_hif_std=("f1_hif", "std"),
            retention_mean=("retention_pct", "mean"),
        )
        .round(4)
        .reset_index()
    )
    summary.to_csv(output_dir / "full_benchmark_summary.csv", index=False)

    print("\n\n" + "=" * 100)
    print("Full Benchmark (Mean ± SD over seeds)")
    print("=" * 100)
    print(
        "| Dataset | Generator | KS (Mean ± SD) | alpha (Mean ± SD) | beta (Mean ± SD) | HIF (Mean ± SD) | Viol% (Mean ± SD) |"
    )
    print("|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        print(
            f"| {r['dataset']} | {r['generator']} | "
            f"{r['ks_mean']:.3f} ± {r['ks_std']:.3f} | "
            f"{r['alpha_mean']:.3f} ± {r['alpha_std']:.3f} | "
            f"{r['beta_mean']:.3f} ± {r['beta_std']:.3f} | "
            f"{r['hif_mean']:.3f} ± {r['hif_std']:.3f} | "
            f"{r['viol_mean'] * 100:.1f} ± {r['viol_std'] * 100:.1f}% |"
        )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full benchmark")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--generators",
        default="gaussian,vine",
        help="Comma-separated: gaussian,vine,ctgan,tvae",
    )
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    run_benchmark(
        datasets=DATASETS,
        generator_names=[g.strip() for g in args.generators.split(",") if g.strip()],
        n_seeds=args.seeds,
        n_rows=args.rows,
        output_dir=Path(args.output_dir),
    )
