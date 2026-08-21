"""
Experiment: Representation Drift Under Integrity Filtering.

Q: Does HIF filtering disproportionately remove records from specific
demographic groups? For each sensitive attribute we measure, per seed:

  - flag rate per group (disparate flagging),
  - group share before vs. after filtering (percentage-point shift),
  - TVD between the pre- and post-filter group distributions.

Run:
    python scripts/18_representation_drift.py --seeds 10
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
from _exp_utils import audit_hif, generate, load_real

CONFIGS = [
    {
        "dataset": "adult",
        "n": 2000,
        "attributes": ["sex", "race"],
    },
    {
        "dataset": "credit",
        "n": 2000,
        "attributes": ["sex", "marriage", "age_quartile"],
    },
]

GENERATORS = ["gaussian", "vine", "ctgan", "tvae"]


def _prepare(real: pd.DataFrame) -> pd.DataFrame:
    """Add derived sensitive columns (age quartiles fit on real data)."""
    out = real.copy()
    if "age" in out.columns:
        try:
            out["age_quartile"] = pd.qcut(out["age"], q=4, duplicates="drop").astype(
                str
            )
        except ValueError:
            pass
    return out


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def run_config(cfg: dict, gen: str, n_seeds: int) -> list[dict]:
    rows: list[dict] = []
    for seed_i in range(n_seeds):
        seed = 42 + seed_i
        real = _prepare(load_real(cfg["dataset"], n=cfg["n"], seed=seed))
        syn = generate(real, cfg["n"], seed, gen)
        syn = syn[real.columns.intersection(syn.columns).tolist()].reset_index(
            drop=True
        )

        hif = audit_hif(real, syn, seed=seed)
        retained = hif["row_penalties"] <= 0.5

        for attr in cfg["attributes"]:
            if attr not in syn.columns:
                continue
            groups = sorted(syn[attr].astype(str).unique())
            flags = hif["row_penalties"] > 0.5
            share_before = np.array(
                [(syn[attr].astype(str) == g).mean() for g in groups]
            )
            share_after = np.array(
                [
                    ((syn[attr].astype(str) == g) & retained).sum()
                    / max(retained.sum(), 1)
                    for g in groups
                ]
            )
            flag_rates = [
                float(flags[syn[attr].astype(str) == g].mean()) for g in groups
            ]
            rows.append(
                {
                    "dataset": cfg["dataset"],
                    "generator": gen,
                    "attribute": attr,
                    "seed": seed,
                    "violation_rate": float(flags.mean()),
                    "retention": float(retained.mean()),
                    "max_flag_gap": float(max(flag_rates) - min(flag_rates)),
                    "max_share_shift_pp": float(
                        np.abs(share_after - share_before).max() * 100
                    ),
                    "tvd": _tvd(share_before, share_after),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for cfg in CONFIGS:
        for gen in GENERATORS:
            print(f"\n=== {cfg['dataset']} x {gen} ===")
            all_rows.extend(run_config(cfg, gen, args.seeds))

    raw = pd.DataFrame(all_rows)
    raw.to_csv(output_dir / "representation_drift_raw.csv", index=False)

    summary = (
        raw.groupby(["dataset", "generator", "attribute"])
        .agg(
            viol_mean=("violation_rate", "mean"),
            flag_gap_mean=("max_flag_gap", "mean"),
            flag_gap_sd=("max_flag_gap", "std"),
            shift_pp_mean=("max_share_shift_pp", "mean"),
            shift_pp_sd=("max_share_shift_pp", "std"),
            tvd_mean=("tvd", "mean"),
            tvd_sd=("tvd", "std"),
        )
        .round(4)
        .reset_index()
    )
    summary.to_csv(output_dir / "representation_drift_summary.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
