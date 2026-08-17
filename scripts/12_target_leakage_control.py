"""
Experiment: Target-Leakage Control for HIF Integrity Filtering.

Q: "Is the downstream utility recovery structural, or is it supervised
    label-noise cleaning in disguise?"

In both configurations that show significant utility recovery, the downstream
target is itself a top-5 LSE hub (census_acs / household_income;
online_purchases / item_total).  The HIF filter therefore conditions on the
label: dropping low-H rows may mechanically drop rows whose target is
improbable given the features, which a supervised label-cleaning step would
also do.  A gain obtained that way says nothing about structural integrity.

This script re-runs the utility-filtering protocol under three auditor
configurations, holding the synthetic cohort fixed within each seed so the
three arms differ only in what the auditor is allowed to see:

    A  published  -- auditor sees every column; target eligible as a hub
    B  no-hub     -- auditor sees every column, but the target is removed from
                     the hub candidate set, so no sentinel predicts the target
                     (it may still act as a predictor for other hubs)
    C  blind      -- auditor never sees the target column at all

Arm C is the decisive control: if the Delta-F1 gain survives when the auditor
is blind to the label, the remediation claim is structural.

Following the canonical protocol of scripts 03--04, each seed draws its own
real-data sample (seed-specific `load_real`), so within each seed the auditor,
the synthetic cohort, and the retained subset match the published per-seed
configuration; arm A therefore reproduces the retention and Delta-F1 of the
corresponding row in Table 3.

Run:
    python scripts/12_target_leakage_control.py --seeds 10
"""

from __future__ import annotations

import argparse
import contextlib
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
from tabular_polygraph.fidelity.hif.sentinel import LogicalSentinelEnsemble

# (dataset_id, target, generator, n_rows) -- the four configurations the
# manuscript reports as significant utility recoveries.
CONFIGS = [
    ("census_acs", "household_income", "ctgan", 2000),
    ("census_acs", "household_income", "vine", 2000),
    ("online_purchases", "item_total", "ctgan", 664),
    ("online_purchases", "item_total", "vine", 664),
]

ARMS = ("A_published", "B_no_hub", "C_blind")


@contextlib.contextmanager
def target_excluded_from_hubs(target: str):
    """Drop ``target`` from the LSE hub candidate set for the duration."""
    original = LogicalSentinelEnsemble._discover_hubs

    def patched(self, df, x_encoded, potential_hubs=None):
        cols = potential_hubs if potential_hubs is not None else list(df.columns)
        return original(self, df, x_encoded, [c for c in cols if c != target])

    LogicalSentinelEnsemble._discover_hubs = patched
    try:
        yield
    finally:
        LogicalSentinelEnsemble._discover_hubs = original


def audit_arm(
    arm: str,
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target: str,
) -> tuple[np.ndarray, list[str]]:
    """Return (row_penalties, selected_hubs) for one auditor configuration.

    Uses ``HIFConfig()``'s default ``random_state`` (42) for every arm.  The
    published runs called ``hif_score(..., random_state=seed)``, which silently
    discards the argument, so they too always audited at 42 -- arm A therefore
    reproduces the published baseline exactly.
    """
    columns = real.columns.intersection(syn.columns).tolist()

    if arm == "C_blind":
        columns = [c for c in columns if c != target]

    def _run() -> tuple[np.ndarray, list[str]]:
        auditor = HIFAuditor(HIFConfig())
        auditor.fit(real, columns=columns)
        result = auditor.score(syn)
        hubs = auditor.oracle.hubs if auditor.oracle is not None else []
        return np.asarray(result["row_penalties"]), list(hubs)

    if arm == "B_no_hub":
        with target_excluded_from_hubs(target):
            return _run()
    return _run()


def paired_stats(full: np.ndarray, filtered: np.ndarray) -> dict:
    """Paired t-test, Wilcoxon and t-based 95% CI on the filtered-minus-full diff."""
    valid = ~(np.isnan(full) | np.isnan(filtered))
    out: dict[str, float] = {
        "n_valid": int(valid.sum()),
        "f1_full": float(np.nanmean(full)),
        "f1_filtered": float(np.nanmean(filtered)),
    }
    if valid.sum() < 3:
        return {
            **out,
            "delta_f1": np.nan,
            "p_ttest": np.nan,
            "p_wilcoxon": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    diffs = filtered[valid] - full[valid]
    _, p_ttest = stats.ttest_rel(filtered[valid], full[valid])
    try:
        _, p_wilcoxon = stats.wilcoxon(filtered[valid], full[valid])
    except ValueError:
        p_wilcoxon = np.nan

    ci_low, ci_high = stats.t.interval(
        0.95, len(diffs) - 1, loc=diffs.mean(), scale=stats.sem(diffs)
    )
    return {
        **out,
        "delta_f1": float(diffs.mean()),
        "p_ttest": float(p_ttest),
        "p_wilcoxon": float(p_wilcoxon),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def run(n_seeds: int, output_dir: Path) -> None:
    rows: list[dict] = []

    for ds_id, target, gen_name, n_rows in CONFIGS:
        print(f"\n{'=' * 70}\n  {ds_id} | {gen_name} | target={target}\n{'=' * 70}")

        for seed_i in range(n_seeds):
            seed = 42 + seed_i
            real = load_real(ds_id, n=n_rows, seed=seed).reset_index(drop=True)
            syn = generate(real, len(real), seed, gen_name)
            syn = syn[real.columns.intersection(syn.columns).tolist()]

            util_full = utility_metrics(real, syn, target, seed)

            for arm in ARMS:
                penalties, hubs = audit_arm(arm, real, syn, target)
                mask = penalties < 0.5
                retention = float(mask.mean() * 100)
                util = (
                    utility_metrics(real, syn[mask], target, seed)
                    if mask.sum() > 10
                    else {"f1": np.nan, "accuracy": np.nan}
                )
                rows.append(
                    {
                        "dataset": ds_id,
                        "generator": gen_name,
                        "target": target,
                        "arm": arm,
                        "seed": seed,
                        "target_is_hub": target in hubs,
                        "hubs": "|".join(hubs),
                        "retention": retention,
                        "f1_full": util_full["f1"],
                        "f1_filtered": util["f1"],
                    }
                )
                print(
                    f"    seed={seed} {arm:12s} retention={retention:5.1f}% "
                    f"F1 {util_full['f1']:.3f} -> {util['f1']:.3f} "
                    f"target_is_hub={target in hubs}"
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "target_leakage_control_raw.csv", index=False)

    summary_rows = []
    for (ds_id, gen_name, arm), grp in raw.groupby(
        ["dataset", "generator", "arm"], sort=False
    ):
        st = paired_stats(
            grp["f1_full"].to_numpy(dtype=float),
            grp["f1_filtered"].to_numpy(dtype=float),
        )
        summary_rows.append(
            {
                "dataset": ds_id,
                "generator": gen_name,
                "arm": arm,
                "target_is_hub": bool(grp["target_is_hub"].iloc[0]),
                "retention": float(grp["retention"].mean()),
                **st,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "target_leakage_control_summary.csv", index=False)

    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    print(
        summary[
            [
                "dataset",
                "generator",
                "arm",
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
    print(f"\nWrote {output_dir / 'target_leakage_control_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    run(n_seeds=args.seeds, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
