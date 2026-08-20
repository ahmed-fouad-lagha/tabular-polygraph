"""
Experiment: Does integrity filtering recover utility for NON-HUB targets?

Q: "Is the utility recovery structural, or does it require the auditor to
    condition on the downstream label?"

Companion to 12_target_leakage_control.py.  Script 12 showed that removing the
target from the hub set destroys the recovery.  This script asks the converse
question on fixed data: holding dataset, generator and synthetic cohort
constant, does recovery appear when the target is a column the auditor did
*not* select as a hub?

Design note -- under the standard protocol the auditor sees every column and
its output does not depend on the target at all.  So one generator fit and one
audit per (dataset, generator, seed) serves every candidate target, and the
comparison across targets is exactly paired: identical synthetic rows,
identical HIF penalties, identical retained subset.  Only the label changes.

If Delta-F1 is positive for hub targets and null for non-hub targets on the
same retained cohort, the recovery is label-conditioned rather than structural.

Following the canonical protocol of scripts 03--04, each seed draws its own
real-data sample (seed-specific `load_real`); within a seed, one generator fit
and one audit serve every candidate target on identical synthetic rows.

Run:
    python scripts/13_nonhub_target_recovery.py --seeds 10
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
from _exp_utils import generate, load_real, utility_metrics
from tabular_polygraph._config import HIFConfig
from tabular_polygraph.fidelity.hif.auditor import HIFAuditor

# (dataset_id, n_rows, candidate target columns).  Each list mixes the auditor's
# selected hubs with non-hub columns; hub membership is recorded per run rather
# than hard-coded, so the split stays correct if hub selection changes.
CONFIGS = [
    (
        "census_acs",
        2000,
        [
            # hubs under the default configuration
            "household_income",
            "housing_cost",
            "poverty_status",
            "education",
            "tenure",
            # non-hubs
            "cost_burden_pct",
            "employment_status",
            "household_size",
            "age_group",
        ],
    ),
    (
        "credit",
        2000,
        [
            # hubs
            "pay_2",
            "pay_3",
            "pay_5",
            "pay_4",
            "pay_6",
            # non-hubs
            "default_payment",
            "limit_bal",
            "age",
            "pay_0",
            "bill_amt1",
            "pay_amt1",
        ],
    ),
]

GENERATORS = ("ctgan", "vine")


def paired_stats(full: np.ndarray, filtered: np.ndarray) -> dict:
    """Paired t-test, Wilcoxon and t-based 95% CI on the filtered-minus-full diff."""
    valid = ~(np.isnan(full) | np.isnan(filtered))
    base = {
        "n_valid": int(valid.sum()),
        "f1_full": float(np.nanmean(full)) if valid.any() else np.nan,
        "f1_filtered": float(np.nanmean(filtered)) if valid.any() else np.nan,
    }
    if valid.sum() < 3:
        return {
            **base,
            "delta_f1": np.nan,
            "p_ttest": np.nan,
            "p_wilcoxon": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    diffs = filtered[valid] - full[valid]
    _, p_t = stats.ttest_rel(filtered[valid], full[valid])
    try:
        _, p_w = stats.wilcoxon(filtered[valid], full[valid])
    except ValueError:
        p_w = np.nan
    if np.allclose(diffs, 0):
        ci_low = ci_high = 0.0
    else:
        ci_low, ci_high = stats.t.interval(
            0.95, len(diffs) - 1, loc=diffs.mean(), scale=stats.sem(diffs)
        )
    return {
        **base,
        "delta_f1": float(diffs.mean()),
        "p_ttest": float(p_t),
        "p_wilcoxon": float(p_w),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _save_checkpoint(output_dir: Path, rows: list[dict]) -> pd.DataFrame:
    """Write the raw CSV (overwrite) and derive the summary."""
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "nonhub_target_recovery_raw.csv", index=False)

    summary = []
    for (ds_id, gen_name, tgt), grp in raw.groupby(
        ["dataset", "generator", "target"], sort=False
    ):
        summary.append(
            {
                "dataset": ds_id,
                "generator": gen_name,
                "target": tgt,
                "target_is_hub": bool(grp["target_is_hub"].iloc[0]),
                "retention": float(grp["retention"].mean()),
                **paired_stats(
                    grp["f1_full"].to_numpy(dtype=float),
                    grp["f1_filtered"].to_numpy(dtype=float),
                ),
            }
        )
    summ = pd.DataFrame(summary)
    summ.to_csv(output_dir / "nonhub_target_recovery_summary.csv", index=False)
    return summ


def _load_existing_rows(output_dir: Path) -> list[dict]:
    """Load rows written by a previous process so resumed runs accumulate.

    Checkpoints overwrite the raw CSV on every save; without this reload, any
    combo skipped via `_completed_combos` would be dropped from the file on the
    first save of the new process.
    """
    raw_path = output_dir / "nonhub_target_recovery_raw.csv"
    if not raw_path.exists():
        return []
    return pd.read_csv(raw_path).to_dict("records")


def _completed_combos(output_dir: Path) -> set[tuple]:
    """Return the set of (dataset, generator, seed) already in the raw CSV."""
    raw_path = output_dir / "nonhub_target_recovery_raw.csv"
    if not raw_path.exists():
        return set()
    df = pd.read_csv(raw_path)
    return set(zip(df["dataset"], df["generator"], df["seed"].astype(int), strict=True))


def run(n_seeds: int, output_dir: Path) -> None:
    rows: list[dict] = _load_existing_rows(output_dir)
    done = _completed_combos(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_id, n_rows, targets in CONFIGS:
        for gen_name in GENERATORS:
            print(f"\n{'=' * 70}\n  {ds_id} | {gen_name}\n{'=' * 70}", flush=True)

            for seed_i in range(n_seeds):
                seed = 42 + seed_i
                if (ds_id, gen_name, seed) in done:
                    print(f"    seed={seed} -- skipped (checkpoint)", flush=True)
                    continue

                real = load_real(ds_id, n=n_rows, seed=seed).reset_index(drop=True)
                usable = [
                    t for t in targets if t in real.columns and real[t].nunique() >= 2
                ]
                syn = generate(real, len(real), seed, gen_name)
                syn = syn[real.columns.intersection(syn.columns).tolist()]

                # One audit serves every target: the auditor never sees the label.
                auditor = HIFAuditor(HIFConfig())
                auditor.fit(real)
                result = auditor.score(syn)
                hubs = list(auditor.oracle.hubs) if auditor.oracle else []
                mask = np.asarray(result["row_penalties"]) < 0.5
                retention = float(mask.mean() * 100)

                if mask.sum() <= 10:
                    print(f"    seed={seed} retention={retention:.1f}% -- too few rows")
                    continue

                syn_kept = syn[mask]
                for tgt in usable:
                    full = utility_metrics(real, syn, tgt, seed)
                    filt = utility_metrics(real, syn_kept, tgt, seed)
                    rows.append(
                        {
                            "dataset": ds_id,
                            "generator": gen_name,
                            "target": tgt,
                            "target_is_hub": tgt in hubs,
                            "seed": seed,
                            "retention": retention,
                            "violation_rate": result["violation_rate"],
                            "f1_full": full["f1"],
                            "f1_filtered": filt["f1"],
                        }
                    )
                # checkpoint after each seed
                _save_checkpoint(output_dir, rows)
                print(
                    f"    seed={seed} retention={retention:5.1f}% "
                    f"({len(usable)} targets scored)",
                    flush=True,
                )

    summ = _save_checkpoint(output_dir, rows)

    pd.set_option("display.width", 240)
    print(f"\n{'=' * 70}\n  PER-TARGET RESULTS\n{'=' * 70}")
    print(
        summ[
            [
                "dataset",
                "generator",
                "target",
                "target_is_hub",
                "retention",
                "f1_full",
                "f1_filtered",
                "delta_f1",
                "p_ttest",
                "ci_low",
                "ci_high",
            ]
        ].to_string(index=False)
    )

    print(f"\n{'=' * 70}\n  HUB vs NON-HUB TARGETS\n{'=' * 70}")
    agg = (
        summ.groupby(["dataset", "generator", "target_is_hub"])
        .agg(
            n_targets=("target", "count"),
            mean_delta_f1=("delta_f1", "mean"),
            n_significant_positive=(
                "p_ttest",
                lambda s: int(((s < 0.05) & (summ.loc[s.index, "delta_f1"] > 0)).sum()),
            ),
        )
        .reset_index()
    )
    print(agg.to_string(index=False))
    print(f"\nWrote {output_dir / 'nonhub_target_recovery_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    run(n_seeds=args.seeds, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
