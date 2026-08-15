"""
Experiment: 16-Configuration Utility Battery + Multiple-Comparison Correction.

Computes, for every dataset--generator configuration in the cross-architecture
benchmark, the paired F1 difference (HIF-filtered minus full synthetic) with
paired t-test, Wilcoxon signed-rank, 95% CI, and paired Cohen's d, then applies
a Bonferroni correction over all configurations with estimable deltas.

All statistics are recomputed from the committed raw per-seed rows in
`outputs/full_benchmark.csv` (the same rows script 03 writes), so every p-value
quoted in Section~\ref{sec:utility} reproduces from a committed artifact.

Run:
    python scripts/15_multiple_comparisons.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
BENCHMARK_CSV = PROJECT_ROOT / "outputs" / "full_benchmark.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "multiple_comparisons.csv"


def paired_statistics(full: np.ndarray, hif: np.ndarray) -> dict:
    """Paired statistics on the HIF-minus-full F1 difference (n >= 3)."""
    valid = ~(np.isnan(full) | np.isnan(hif))
    full_v, hif_v = full[valid], hif[valid]
    n = len(full_v)
    if n < 3:
        return {
            "n": n,
            "delta": np.nan,
            "p_ttest": np.nan,
            "p_wilcoxon": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "cohens_d": np.nan,
        }

    diffs = hif_v - full_v
    t_stat, p_t = stats.ttest_rel(full_v, hif_v)
    try:
        _, p_w = stats.wilcoxon(full_v, hif_v)
    except ValueError:
        p_w = np.nan
    ci_low, ci_high = stats.t.interval(
        0.95, len(diffs) - 1, loc=diffs.mean(), scale=stats.sem(diffs)
    )
    d = diffs.mean() / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else np.nan
    return {
        "n": n,
        "delta": diffs.mean(),
        "p_ttest": p_t,
        "p_wilcoxon": p_w,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohens_d": d,
    }


def main() -> None:
    df = pd.read_csv(BENCHMARK_CSV)

    rows: list[dict] = []
    for (ds, gen), g in df.groupby(["dataset", "generator"]):
        full = g["f1_full"].to_numpy(dtype=float)
        hif = g["f1_hif"].to_numpy(dtype=float)
        stats_ = paired_statistics(full, hif)
        rows.append(
            {
                "dataset": ds,
                "generator": gen,
                "n_seeds": len(g),
                **stats_,
            }
        )

    out = pd.DataFrame(rows)

    # Configurations with estimable deltas define the correction family.
    estimable = out["n"] >= 3
    m = int(estimable.sum())
    alpha_bonf = ALPHA / m
    out.loc[:, "bonf_threshold"] = alpha_bonf
    out.loc[:, "sig_ttest"] = out["p_ttest"] < ALPHA
    out.loc[:, "sig_bonf"] = out["p_ttest"] < alpha_bonf

    out = out.round(
        {
            "delta": 3,
            "p_ttest": 4,
            "p_wilcoxon": 4,
            "ci_low": 3,
            "ci_high": 3,
            "cohens_d": 2,
            "bonf_threshold": 4,
        }
    )
    out.to_csv(OUTPUT_CSV, index=False)

    print(
        f"Configurations with estimable paired deltas: {m} (Bonferroni alpha = {alpha_bonf:.4f})"
    )
    print()
    print(
        "| dataset | generator | n | ΔF1 | p (t) | p (Wilcoxon) | CI | d | sig@0.05 | sig@Bonf |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in out.sort_values(["dataset", "generator"]).iterrows():
        print(
            f"| {r['dataset']} | {r['generator']} | {r['n']} | "
            f"{r['delta']:+.3f} | {r['p_ttest']:.4f} | {r['p_wilcoxon']:.4f} | "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['cohens_d']:.2f} | "
            f"{bool(r['sig_ttest'])} | {bool(r['sig_bonf'])} |"
        )
    print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
