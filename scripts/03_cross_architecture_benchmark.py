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
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _run_cell(
    ds_id: str, target: str, task: str, gen_name: str, seed: int, n_rows: int
) -> dict:
    """Score a single dataset-generator-seed cell (picklable for the pool)."""
    try:
        real = load_real(ds_id, n=n_rows, seed=seed).reset_index(drop=True)
    except Exception as e:
        return {
            "dataset": ds_id,
            "generator": gen_name,
            "seed": seed,
            "error": f"load: {e}",
        }

    try:
        syn = generate(real, len(real), seed, gen_name)
    except Exception as e:
        return {
            "dataset": ds_id,
            "generator": gen_name,
            "seed": seed,
            "error": f"generate: {e}",
        }
    syn = syn[real.columns.intersection(syn.columns).tolist()]

    agg = aggregate_metrics(real, syn, seed)
    hif_res = audit_hif(real, syn, seed=seed)

    util_full = utility_metrics(real, syn, target, seed)
    syn_filtered = syn[hif_res["row_penalties"] < 0.5]
    util_hif = (
        utility_metrics(real, syn_filtered, target, seed)
        if len(syn_filtered) > 10
        else {"f1": np.nan, "accuracy": np.nan, "trr": np.nan}
    )
    retention = len(syn_filtered) / len(syn) * 100

    return {
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


def _run_cell_tuple(cell: tuple) -> dict:
    return _run_cell(*cell)


def _done_cells(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return {
            (r.dataset, r.generator, int(r.seed))
            for r in df.itertuples()
            if "error" not in getattr(r, "_fields", [])
        }
    except Exception:
        return set()


def _write_row(csv_path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(
        csv_path, mode="a", header=not csv_path.exists(), index=False
    )


def run_benchmark(
    datasets: list[tuple[str, str, str]],
    generator_names: list[str],
    n_seeds: int,
    n_rows: int,
    output_dir: Path,
    workers: int = 1,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "full_benchmark.csv"
    cells = [
        (ds_id, target, task, gen_name, 42 + seed_i, n_rows)
        for ds_id, target, task in datasets
        for gen_name in generator_names
        for seed_i in range(n_seeds)
    ]
    already = _done_cells(csv_path)
    cells = [c for c in cells if (c[0], c[3], c[4]) not in already]
    if already:
        print(
            f"  resuming: {len(already)} cells already in {csv_path.name}", flush=True
        )

    if workers > 1 and len(cells) > 1:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_cell_tuple, c) for c in cells]
            for i, fut in enumerate(as_completed(futures), start=1):
                r = fut.result()
                results.append(r)
                _write_row(csv_path, r)
                print(
                    f"  [{i}/{len(cells)}] {r.get('dataset', '?')}/{r.get('generator', '?')} "
                    f"seed={r.get('seed', '?')} {'ERR' if 'error' in r else 'ok'}",
                    flush=True,
                )
    else:
        results = []
        for i, c in enumerate(cells, start=1):
            r = _run_cell(*c)
            results.append(r)
            _write_row(csv_path, r)
            print(
                f"  [{i}/{len(cells)}] {r.get('dataset', '?')}/{r.get('generator', '?')} "
                f"seed={r.get('seed', '?')} {'ERR' if 'error' in r else 'ok'}",
                flush=True,
            )

    errors = [r for r in results if "error" in r]
    for e in errors:
        print(f"  cell failed: {e}")
    all_rows = [r for r in results if "error" not in r]
    if not all_rows:
        print(f"  no new rows (cells remaining: {len(cells)})")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["dataset", "generator", "seed"])

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
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    run_benchmark(
        datasets=DATASETS,
        generator_names=[g.strip() for g in args.generators.split(",") if g.strip()],
        n_seeds=args.seeds,
        n_rows=args.rows,
        output_dir=Path(args.output_dir),
        workers=args.workers,
    )
