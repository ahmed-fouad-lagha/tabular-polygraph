"""Leakage-free target-withholding control for HIF filtering.

For each seed, the real cohort is split before fitting either the generator or
auditor.  All three auditor arms train on the same real training partition and
their synthetic-trained classifiers are evaluated on the untouched test
partition.  Thus target blindness is tested without reintroducing TSTR
evaluation leakage.
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
from _exp_utils import (
    generate,
    load_real,
    split_real_for_utility,
    utility_metrics_on_holdout,
)
from tabular_polygraph._config import HIFConfig
from tabular_polygraph.fidelity.hif.auditor import HIFAuditor
from tabular_polygraph.fidelity.hif.sentinel import LogicalSentinelEnsemble

CONFIGS = (
    ("census_acs", "household_income", "ctgan", 2000),
    ("census_acs", "household_income", "vine", 2000),
    ("online_purchases", "item_total", "ctgan", 664),
    ("online_purchases", "item_total", "vine", 664),
)
ARMS = ("A_published", "B_no_hub", "C_blind")


@contextlib.contextmanager
def target_excluded_from_hubs(target: str):
    """Temporarily prevent the target from being selected as an LSE hub."""
    original = LogicalSentinelEnsemble._discover_hubs

    def patched(self, df, x_encoded, potential_hubs=None):
        candidates = potential_hubs if potential_hubs is not None else list(df.columns)
        return original(self, df, x_encoded, [c for c in candidates if c != target])

    LogicalSentinelEnsemble._discover_hubs = patched
    try:
        yield
    finally:
        LogicalSentinelEnsemble._discover_hubs = original


def audit_arm(
    arm: str, train_real: pd.DataFrame, synthetic: pd.DataFrame, target: str, seed: int
) -> tuple[np.ndarray, list[str]]:
    """Fit one arm exclusively on ``train_real`` and score ``synthetic``."""
    columns = train_real.columns.intersection(synthetic.columns).tolist()
    if arm == "C_blind":
        columns = [column for column in columns if column != target]

    def run_auditor() -> tuple[np.ndarray, list[str]]:
        auditor = HIFAuditor(HIFConfig(random_state=seed))
        auditor.fit(train_real, columns=columns)
        result = auditor.score(synthetic)
        hubs = auditor.oracle.hubs if auditor.oracle is not None else []
        return np.asarray(result["row_penalties"]), list(hubs)

    if arm == "B_no_hub":
        with target_excluded_from_hubs(target):
            return run_auditor()
    return run_auditor()


def paired_stats(group: pd.DataFrame) -> dict[str, float | int]:
    valid = group.dropna(subset=["f1_full", "f1_filtered"]).copy()
    diffs = valid["f1_filtered"] - valid["f1_full"]
    n = len(diffs)
    if n < 2:
        return {
            "n_valid_seeds": n,
            "delta_f1_mean": np.nan,
            "paired_ttest_p": np.nan,
            "delta_f1_ci_low": np.nan,
            "delta_f1_ci_high": np.nan,
        }
    ci_low, ci_high = stats.t.interval(
        0.95, n - 1, loc=diffs.mean(), scale=stats.sem(diffs)
    )
    return {
        "n_valid_seeds": n,
        "delta_f1_mean": float(diffs.mean()),
        "paired_ttest_p": float(
            stats.ttest_rel(valid["f1_filtered"], valid["f1_full"]).pvalue
        ),
        "delta_f1_ci_low": float(ci_low),
        "delta_f1_ci_high": float(ci_high),
    }


def run(
    n_seeds: int, output_dir: Path, test_size: float = 0.30, seed_start: int = 42
) -> tuple[Path, Path]:
    """Run all configurations, checkpointing every completed seed/arm."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "leakage_free_target_withholding_raw.csv"
    summary_path = output_dir / "leakage_free_target_withholding_summary.csv"
    rows = pd.read_csv(raw_path).to_dict("records") if raw_path.exists() else []
    completed = {(r["dataset"], r["generator"], int(r["seed"]), r["arm"]) for r in rows}

    for dataset, target, generator, n_rows in CONFIGS:
        for offset in range(n_seeds):
            seed = seed_start + offset
            needed = [(dataset, generator, seed, arm) for arm in ARMS]
            if all(key in completed for key in needed):
                print(
                    f"{dataset}/{generator} seed={seed} already checkpointed",
                    flush=True,
                )
                continue
            real = load_real(dataset, n=n_rows, seed=seed).reset_index(drop=True)
            train_real, test_real = split_real_for_utility(
                real, target, seed, test_size
            )
            synthetic = generate(train_real, len(real), seed, generator)
            synthetic = synthetic[
                train_real.columns.intersection(synthetic.columns)
            ].copy()
            full = utility_metrics_on_holdout(
                train_real, synthetic, test_real, target, seed
            )
            for arm in ARMS:
                key = (dataset, generator, seed, arm)
                if key in completed:
                    continue
                penalties, hubs = audit_arm(arm, train_real, synthetic, target, seed)
                keep = penalties < 0.5
                filtered = synthetic.loc[keep].reset_index(drop=True)
                utility = (
                    utility_metrics_on_holdout(
                        train_real, filtered, test_real, target, seed
                    )
                    if len(filtered) >= 10
                    else {"f1": np.nan, "accuracy": np.nan}
                )
                row = {
                    "dataset": dataset,
                    "generator": generator,
                    "target": target,
                    "seed": seed,
                    "arm": arm,
                    "n_real_total": len(real),
                    "n_real_train": len(train_real),
                    "n_real_test": len(test_real),
                    "n_synthetic": len(synthetic),
                    "test_size": test_size,
                    "target_is_hub": target in hubs,
                    "hubs": "|".join(hubs),
                    "retention_pct": float(keep.mean() * 100),
                    "f1_full": full["f1"],
                    "accuracy_full": full["accuracy"],
                    "f1_filtered": utility["f1"],
                    "accuracy_filtered": utility["accuracy"],
                }
                rows.append(row)
                completed.add(key)
                pd.DataFrame(rows).to_csv(raw_path, index=False)
                print(
                    f"{dataset}/{generator} seed={seed} {arm}: retention={row['retention_pct']:.1f}% F1 {full['f1']:.3f}->{utility['f1']:.3f}",
                    flush=True,
                )

    raw = pd.DataFrame(rows)
    summary_rows = []
    for (dataset, generator, arm), group in raw.groupby(
        ["dataset", "generator", "arm"], sort=False
    ):
        summary_rows.append(
            {
                "dataset": dataset,
                "generator": generator,
                "arm": arm,
                "target_is_hub": bool(group["target_is_hub"].iloc[0]),
                "retention_mean_pct": group["retention_pct"].mean(),
                "f1_full_mean": group["f1_full"].mean(),
                "f1_filtered_mean": group["f1_filtered"].mean(),
                **paired_stats(group),
            }
        )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return raw_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/leakage_free"
    )
    args = parser.parse_args()
    raw, summary = run(args.seeds, args.output_dir, args.test_size, args.seed_start)
    print(f"Raw results: {raw}\nSummary: {summary}")
