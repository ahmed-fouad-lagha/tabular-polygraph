"""Leakage-free target sweep on Census ACS.

One pre-generator train/test split, generator fit, and HIF audit are shared by
all targets within each seed.  Thus the retained synthetic cohort is fixed and
only the downstream label changes.
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
    generate,
    load_real,
    split_real_for_utility,
    utility_metrics_on_holdout,
)

TARGETS = [
    "household_income",
    "housing_cost",
    "poverty_status",
    "education",
    "tenure",
    "cost_burden_pct",
    "employment_status",
    "household_size",
    "age_group",
]


def _summary(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (generator, target), group in raw.groupby(["generator", "target"], sort=False):
        delta = group["f1_hif"] - group["f1_full"]
        valid = delta.dropna()
        ci_low, ci_high = (np.nan, np.nan)
        p_value = np.nan
        if len(valid) >= 2:
            ci_low, ci_high = stats.t.interval(
                0.95, len(valid) - 1, loc=valid.mean(), scale=stats.sem(valid)
            )
            p_value = stats.ttest_1samp(valid, 0.0).pvalue
        records.append(
            {
                "generator": generator,
                "target": target,
                "target_is_hub": bool(group["target_is_hub"].iloc[0]),
                "n_valid_seeds": len(valid),
                "retention_mean_pct": group["retention_pct"].mean(),
                "delta_f1_mean": valid.mean(),
                "paired_ttest_p": p_value,
                "delta_f1_ci_low": ci_low,
                "delta_f1_ci_high": ci_high,
            }
        )
    return pd.DataFrame(records)


def run(n_seeds: int, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "leakage_free_target_sweep_raw.csv"
    summary_path = output_dir / "leakage_free_target_sweep_summary.csv"
    rows = pd.read_csv(raw_path).to_dict("records") if raw_path.exists() else []
    completed = {(row["generator"], int(row["seed"])) for row in rows}

    for generator in ("ctgan", "vine"):
        for seed in range(42, 42 + n_seeds):
            if (generator, seed) in completed:
                print(f"{generator} seed={seed} already checkpointed", flush=True)
                continue
            real = load_real("census_acs", n=2000, seed=seed)
            train, test = split_real_for_utility(real, "household_income", seed)
            synthetic = generate(train, len(real), seed, generator)
            synthetic = synthetic[
                train.columns.intersection(synthetic.columns).tolist()
            ]
            from tabular_polygraph._config import HIFConfig
            from tabular_polygraph.fidelity.hif.auditor import HIFAuditor

            auditor = HIFAuditor(HIFConfig(random_state=seed))
            auditor.fit(train)
            hif = auditor.score(synthetic)
            keep = hif["row_penalties"] < 0.5
            retained = synthetic.loc[keep].reset_index(drop=True)
            hubs = set(auditor.oracle.hubs if auditor.oracle else [])

            for target in TARGETS:
                full = utility_metrics_on_holdout(train, synthetic, test, target, seed)
                filtered = utility_metrics_on_holdout(
                    train, retained, test, target, seed
                )
                rows.append(
                    {
                        "generator": generator,
                        "seed": seed,
                        "target": target,
                        "target_is_hub": target in hubs,
                        "retention_pct": keep.mean() * 100,
                        "f1_full": full["f1"],
                        "f1_hif": filtered["f1"],
                    }
                )
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(
                f"{generator} seed={seed} retention={keep.mean() * 100:.1f}%",
                flush=True,
            )

    raw = pd.DataFrame(rows)
    _summary(raw).to_csv(summary_path, index=False)
    return raw_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/leakage_free"
    )
    args = parser.parse_args()
    raw, summary = run(args.seeds, args.output_dir)
    print(f"Raw results: {raw}")
    print(f"Summary: {summary}")
