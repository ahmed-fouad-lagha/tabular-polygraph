"""Leakage-free TSTR evaluation for HIF filtering.

For every seed, the real cohort is split before generator or auditor fitting.
The generator and HIF see only the training partition; synthetic-trained
downstream models are evaluated only on the untouched real test partition.

Example:
    python scripts/19_leakage_free_utility.py \
        --dataset census_acs --target household_income \
        --generator gaussian_copula --rows 2000 --seeds 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ruff: noqa: E402
from _exp_utils import (
    audit_hif,
    generate,
    load_real,
    split_real_for_utility,
    utility_metrics_on_holdout,
)


def _paired_summary(raw: pd.DataFrame) -> pd.DataFrame:
    valid = raw.dropna(subset=["f1_full", "f1_hif"]).copy()
    valid["delta_f1"] = valid["f1_hif"] - valid["f1_full"]
    n = len(valid)
    if n >= 2:
        ci_low, ci_high = stats.t.interval(
            0.95,
            n - 1,
            loc=valid["delta_f1"].mean(),
            scale=stats.sem(valid["delta_f1"]),
        )
    else:
        ci_low, ci_high = np.nan, np.nan
    p_value = (
        stats.ttest_rel(valid["f1_full"], valid["f1_hif"]).pvalue if n >= 2 else np.nan
    )
    return pd.DataFrame(
        [
            {
                "n_valid_seeds": n,
                "f1_full_mean": valid["f1_full"].mean(),
                "f1_hif_mean": valid["f1_hif"].mean(),
                "delta_f1_mean": valid["delta_f1"].mean(),
                "retention_mean_pct": valid["retention_pct"].mean(),
                "paired_ttest_p": p_value,
                "delta_f1_ci_low": ci_low,
                "delta_f1_ci_high": ci_high,
            }
        ]
    )


def run_experiment(
    dataset_id: str,
    target: str,
    generator_type: str,
    n_rows: int,
    n_seeds: int,
    test_size: float,
    output_dir: Path,
    overwrite: bool = False,
    seed_start: int = 42,
) -> tuple[Path, Path]:
    """Run one generator--dataset leakage-free utility configuration."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"leakage_free_{dataset_id}_{generator_type}_{target}"
    raw_path = output_dir / f"{stem}_raw.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    if summary_path.exists() and not raw_path.exists():
        raise FileExistsError(
            f"Found summary without raw checkpoint; inspect before rerunning: {summary_path}"
        )
    if raw_path.exists() and not overwrite:
        existing = pd.read_csv(raw_path)
        required = {"dataset", "target", "generator", "seed", "test_size"}
        if not required.issubset(existing.columns):
            raise ValueError(f"Raw checkpoint has an incompatible schema: {raw_path}")
        # Earlier checkpoints predate this field.  They are compatible only
        # with the current requested cap and are upgraded in place below.
        if "requested_n_rows" not in existing.columns:
            existing["requested_n_rows"] = n_rows
        if not (
            (existing["dataset"] == dataset_id).all()
            and (existing["target"] == target).all()
            and (existing["generator"] == generator_type).all()
            and (existing["requested_n_rows"] == n_rows).all()
            and (existing["test_size"] == test_size).all()
        ):
            raise ValueError(
                f"Raw checkpoint belongs to another experiment: {raw_path}"
            )
        rows: list[dict[str, float | int | str]] = existing.to_dict("records")
    else:
        rows = []
    completed_seeds = {int(row["seed"]) for row in rows}

    for seed_offset in range(n_seeds):
        seed = seed_start + seed_offset
        if seed in completed_seeds:
            print(f"seed={seed} already checkpointed; skipping", flush=True)
            continue
        real = load_real(dataset_id, n=n_rows, seed=seed).reset_index(drop=True)
        if target not in real.columns:
            raise ValueError(
                f"Target '{target}' is not present in dataset '{dataset_id}'"
            )

        train_real, test_real = split_real_for_utility(real, target, seed, test_size)

        # Keep the synthetic cohort size equal to the sampled real cohort.  This
        # is 2,000 for capped benchmarks and the full size for smaller tables.
        synthetic = generate(train_real, len(real), seed, generator_type)
        columns = train_real.columns.intersection(synthetic.columns).tolist()
        synthetic = synthetic[columns].copy()
        if target not in synthetic.columns:
            raise ValueError(f"Generator output is missing target '{target}'")

        hif_result = audit_hif(train_real, synthetic, seed=seed)
        keep_mask = hif_result["row_penalties"] < 0.5
        synthetic_hif = synthetic.loc[keep_mask].reset_index(drop=True)

        full = utility_metrics_on_holdout(
            train_real, synthetic, test_real, target, seed
        )
        filtered = (
            utility_metrics_on_holdout(
                train_real, synthetic_hif, test_real, target, seed
            )
            if len(synthetic_hif) >= 10
            else {"f1": np.nan, "accuracy": np.nan}
        )
        retention = len(synthetic_hif) / len(synthetic) * 100
        rows.append(
            {
                "dataset": dataset_id,
                "target": target,
                "generator": generator_type,
                "seed": seed,
                "requested_n_rows": n_rows,
                "n_real_total": len(real),
                "n_real_train": len(train_real),
                "n_real_test": len(test_real),
                "n_synthetic": len(synthetic),
                "test_size": test_size,
                "retention_pct": retention,
                "hif_score": hif_result["hif_score"],
                "violation_rate": hif_result["violation_rate"],
                "f1_full": full["f1"],
                "accuracy_full": full["accuracy"],
                "f1_hif": filtered["f1"],
                "accuracy_hif": filtered["accuracy"],
            }
        )
        pd.DataFrame(rows).to_csv(raw_path, index=False)
        print(
            f"seed={seed} train={len(train_real)} test={len(test_real)} "
            f"retention={retention:.1f}% full_f1={full['f1']:.3f} "
            f"hif_f1={filtered['f1']:.3f}",
            flush=True,
        )

    raw = pd.DataFrame(rows)
    _paired_summary(raw).to_csv(summary_path, index=False)
    return raw_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leakage-free HIF TSTR evaluation")
    parser.add_argument("--dataset", default="census_acs")
    parser.add_argument("--target", default="household_income")
    parser.add_argument(
        "--generator",
        default="gaussian_copula",
        choices=["gaussian_copula", "ctgan", "tvae", "vine"],
    )
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/leakage_free")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_output, summary_output = run_experiment(
        dataset_id=args.dataset,
        target=args.target,
        generator_type=args.generator,
        n_rows=args.rows,
        n_seeds=args.seeds,
        test_size=args.test_size,
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
        seed_start=args.seed_start,
    )
    print(f"Raw results: {raw_output}")
    print(f"Summary: {summary_output}")
