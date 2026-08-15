"""
Experiment: Downstream-Utility Threshold Sensitivity (Q3).

Q: "How sensitive are the utility-recovery conclusions to the H < 0.5
    threshold choice (the appendix shows score-level insensitivity on
    Census ACS — is that also true for downstream F1)?"

Re-runs the integrity-filtered downstream F1 protocol on the significant
utility-recovery case (Census ACS, CTGAN, N=10 seeds).  Filtering keeps rows
with ``row_penalty <= threshold``, and ``row_penalty == 1 - H``, so the sweep
``threshold in {0.3, 0.5, 0.7}`` corresponds to the retention frontiers
``H >= {0.7, 0.5, 0.3}``.  The ``threshold`` column of the output CSVs is the
penalty threshold; ``retention_frontier`` is the equivalent H bound and is what
should be quoted in prose.  Larger frontier => stricter filter => lower
retention.

Run:
    python scripts/10_threshold_utility_sensitivity.py --seeds 10
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
from _exp_utils import audit_hif, generate, load_real, rule_mask, utility_metrics

THRESHOLDS = [0.3, 0.5, 0.7]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for thr in THRESHOLDS:
        sub = df[df["threshold"] == thr].dropna(subset=["filtered_f1"])
        full = df[df["threshold"] == thr].dropna(subset=["full_f1"])
        diffs = sub["filtered_f1"].values - full["full_f1"].values
        if len(diffs) >= 3:
            _, p_ttest = stats.ttest_rel(
                full["full_f1"].values, sub["filtered_f1"].values
            )
            try:
                _, p_wilcoxon = stats.wilcoxon(
                    full["full_f1"].values, sub["filtered_f1"].values
                )
            except ValueError:
                p_wilcoxon = np.nan
            ci_low, ci_high = stats.t.interval(
                0.95, len(diffs) - 1, loc=diffs.mean(), scale=stats.sem(diffs)
            )
        else:
            p_ttest, p_wilcoxon, ci_low, ci_high = np.nan, np.nan, np.nan, np.nan
        summary_rows.append(
            {
                "threshold": thr,
                "retention_frontier": round(1.0 - thr, 2),
                "filtered_f1": sub["filtered_f1"].mean(),
                "retention": sub["retention"].mean(),
                "delta_f1": diffs.mean() if len(diffs) else np.nan,
                "p_ttest": p_ttest,
                "p_wilcoxon": p_wilcoxon,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(summary_rows).round(4)


def run_threshold_sensitivity(
    dataset_id: str,
    target: str,
    n_rows: int,
    n_seeds: int,
    generator_type: str,
    output_dir: Path,
):
    print(f"Dataset: {dataset_id} | Generator: {generator_type} | Rows: {n_rows}")

    rows: list[dict] = []

    for seed_i in range(n_seeds):
        seed = 42 + seed_i
        real = load_real(dataset_id, n=n_rows, seed=seed).reset_index(drop=True)
        syn = generate(real, n_rows, seed, generator_type)
        syn = syn[real.columns.intersection(syn.columns).tolist()]

        util_full = utility_metrics(real, syn, target, seed)
        rmask = rule_mask(real, syn, seed)
        util_rule = utility_metrics(real, syn[rmask < 0.5], target, seed)

        hif_result = audit_hif(real, syn, seed=seed)
        pen = hif_result["row_penalties"]

        for thr in THRESHOLDS:
            # NOTE: ``thr`` is a *penalty* threshold. ``pen == 1 - H``
            # (auditor.py), so ``pen <= thr`` retains rows with H >= 1 - thr.
            # thr=0.3 is therefore the STRICTEST arm (retains H >= 0.7) and
            # thr=0.7 the loosest (retains H >= 0.3).  Report results against
            # the retention frontier ``1 - thr``, not against ``thr``.
            mask = pen <= thr
            retention = float(mask.mean())
            util = (
                utility_metrics(real, syn[mask], target, seed)
                if mask.sum() > 10
                else {"f1": np.nan, "accuracy": np.nan}
            )
            rows.append(
                {
                    "seed": seed,
                    "threshold": thr,
                    "retention_frontier": round(1.0 - thr, 2),
                    "full_f1": util_full["f1"],
                    "rule_f1": util_rule["f1"],
                    "filtered_f1": util["f1"],
                    "retention": round(retention * 100, 2),
                }
            )
        print(
            f"  seed {seed}: full_f1={util_full['f1']:.3f} "
            + " ".join(
                f"| H>={round(1 - thr, 2)}: f1={next(r['filtered_f1'] for r in rows if r['seed'] == seed and r['threshold'] == thr):.3f} "
                f"ret={next(r['retention'] for r in rows if r['seed'] == seed and r['threshold'] == thr):.0f}%"
                for thr in THRESHOLDS
            )
        )

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "threshold_utility_sensitivity_raw.csv", index=False)

    summary = summarize(df)
    summary.to_csv(
        output_dir / "threshold_utility_sensitivity_summary.csv", index=False
    )

    print("\n" + "=" * 90)
    print("Downstream-F1 Threshold Sensitivity (mean over seeds)")
    print("=" * 90)
    print(
        f"{'H >=':>10} | {'F1':>6} | {'retention':>9} | "
        f"{'delta_F1':>8} | {'p(ttest)':>8} | {'p(wilcox)':>9} | {'95% CI':>16}"
    )
    for _, r in summary.iterrows():
        ci = (
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
            if not np.isnan(r["ci_low"])
            else "N/A"
        )
        print(
            f"{r['retention_frontier']:>10} | {r['filtered_f1']:>6.3f} | {r['retention']:>8.1f}% | "
            f"{r['delta_f1']:>+8.3f} | {r['p_ttest']:>8.4f} | {r['p_wilcoxon']:>9.4f} | {ci:>16}"
        )
    full_f1 = df["full_f1"].dropna().mean()
    print(f"  full cohort F1 (reference): {full_f1:.3f}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downstream-F1 sensitivity to HIF threshold (Q3)"
    )
    parser.add_argument("--dataset", default="census_acs")
    parser.add_argument("--target", default="household_income")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--generator", default="ctgan")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    run_threshold_sensitivity(
        dataset_id=args.dataset,
        target=args.target,
        n_rows=args.rows,
        n_seeds=args.seeds,
        generator_type=args.generator,
        output_dir=Path(args.output_dir),
    )
