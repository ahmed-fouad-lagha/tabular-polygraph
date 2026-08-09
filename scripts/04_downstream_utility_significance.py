"""
Experiment: Statistical Significance Tests (ported to current API).

Q: "Are the observed utility gains statistically significant (p < 0.05)?"

Re-runs utility filtering with N seeds and computes paired statistical tests.

Run:
    python scripts/03_statistical_significance.py \
      --dataset census_acs --rows 2000 --seeds 10 --generator gaussian_copula
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


def run_significance_test(
    dataset_id: str,
    target: str,
    n_rows: int,
    n_seeds: int,
    generator_type: str,
    output_dir: Path,
):
    print(f"Loading dataset: {dataset_id}")
    real = load_real(dataset_id, n=n_rows)
    print(f"Fitting generator: {generator_type}")
    print(f"Real rows: {len(real)}")

    full_f1s: list[float] = []
    rule_f1s: list[float] = []
    hif_f1s: list[float] = []
    full_accs: list[float] = []
    rule_accs: list[float] = []
    hif_accs: list[float] = []
    retentions_rule: list[float] = []
    retentions_hif: list[float] = []
    hif_scores_list: list[float] = []

    for seed_i in range(n_seeds):
        seed = 42 + seed_i
        print(f"\nSeed {seed} ({seed_i + 1}/{n_seeds})")

        syn = generate(real, n_rows, seed, generator_type)
        syn = syn[real.columns.intersection(syn.columns).tolist()]

        util_full = utility_metrics(real, syn, target, seed)
        full_f1s.append(util_full["f1"])
        full_accs.append(util_full["accuracy"])

        rmask = rule_mask(real, syn, seed)
        syn_rule = syn[rmask < 0.5]
        retentions_rule.append(len(syn_rule) / len(syn) * 100)
        util_rule = (
            utility_metrics(real, syn_rule, target, seed)
            if len(syn_rule) > 10
            else {"f1": np.nan, "accuracy": np.nan, "trr": np.nan}
        )
        rule_f1s.append(util_rule["f1"])
        rule_accs.append(util_rule["accuracy"])

        hif_result = audit_hif(real, syn, seed=seed)
        hif_scores_list.append(hif_result["hif_score"])
        syn_hif = syn[hif_result["row_penalties"] < 0.5]
        retentions_hif.append(len(syn_hif) / len(syn) * 100)
        util_hif = (
            utility_metrics(real, syn_hif, target, seed)
            if len(syn_hif) > 10
            else {"f1": np.nan, "accuracy": np.nan, "trr": np.nan}
        )
        hif_f1s.append(util_hif["f1"])
        hif_accs.append(util_hif["accuracy"])

        print(
            f"  Full F1={util_full['f1']:.3f} | Rule F1={util_rule['f1']:.3f} "
            f"| HIF F1={util_hif['f1']:.3f} | retention={retentions_hif[-1]:.1f}%"
        )

    full_f1s_arr = np.array(full_f1s, dtype=float)
    rule_f1s_arr = np.array(rule_f1s, dtype=float)
    hif_f1s_arr = np.array(hif_f1s, dtype=float)

    valid_mask = ~(np.isnan(full_f1s_arr) | np.isnan(hif_f1s_arr))

    if valid_mask.sum() >= 3:
        t_stat, p_value_ttest = stats.ttest_rel(
            full_f1s_arr[valid_mask], hif_f1s_arr[valid_mask]
        )
        try:
            w_stat, p_value_wilcoxon = stats.wilcoxon(
                full_f1s_arr[valid_mask], hif_f1s_arr[valid_mask]
            )
        except ValueError:
            w_stat, p_value_wilcoxon = np.nan, np.nan
    else:
        t_stat, p_value_ttest = np.nan, np.nan
        w_stat, p_value_wilcoxon = np.nan, np.nan

    diffs = hif_f1s_arr[valid_mask] - full_f1s_arr[valid_mask]
    ci_low, ci_high = (
        stats.t.interval(0.95, len(diffs) - 1, loc=diffs.mean(), scale=stats.sem(diffs))
        if len(diffs) > 1
        else (np.nan, np.nan)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(
        {
            "seed": [42 + i for i in range(n_seeds)],
            "generator": generator_type,
            "full_f1": full_f1s_arr,
            "rule_f1": rule_f1s_arr,
            "hif_f1": hif_f1s_arr,
            "full_acc": full_accs,
            "rule_acc": rule_accs,
            "hif_acc": hif_accs,
            "retention_rule": retentions_rule,
            "retention_hif": retentions_hif,
            "hif_score": hif_scores_list,
        }
    )
    raw_df.to_csv(
        output_dir / f"significance_{dataset_id}_{generator_type}_raw.csv", index=False
    )

    def fmt(v: float) -> str:
        return "N/A" if np.isnan(v) else f"{v:.4f}"

    print("\n" + "=" * 80)
    print(f"Statistical Significance ({dataset_id}, {generator_type})")
    print("=" * 80)
    print()
    print("| Variant | Retention% | F1 (mean +/- SEM) | Accuracy (mean +/- SEM) |")
    print("|---|---|---|---|")
    print(
        f"| Full synthetic | 100.0% | {np.nanmean(full_f1s_arr):.3f} ± "
        f"{stats.sem(full_f1s_arr[~np.isnan(full_f1s_arr)]):.3f} | "
        f"{np.nanmean(full_accs):.3f} ± "
        f"{stats.sem(np.array(full_accs)[~np.isnan(full_accs)]):.3f} |"
    )
    print(
        f"| Rule-only | {np.nanmean(retentions_rule):.1f}% | {np.nanmean(rule_f1s_arr):.3f} ± "
        f"{stats.sem(rule_f1s_arr[~np.isnan(rule_f1s_arr)]):.3f} | "
        f"{np.nanmean(rule_accs):.3f} ± "
        f"{stats.sem(np.array(rule_accs)[~np.isnan(rule_accs)]):.3f} |"
    )
    print(
        f"| **HIF Oracle** | **{np.nanmean(retentions_hif):.1f}%** | "
        f"**{np.nanmean(hif_f1s_arr):.3f} ± "
        f"{stats.sem(hif_f1s_arr[~np.isnan(hif_f1s_arr)]):.3f}** | "
        f"**{np.nanmean(hif_accs):.3f} ± "
        f"{stats.sem(np.array(hif_accs)[~np.isnan(hif_accs)]):.3f}** |"
    )
    print()
    print(f"Paired t-test (Full vs HIF): t={t_stat:.3f}, p={fmt(p_value_ttest)}")
    if not np.isnan(p_value_wilcoxon):
        print(f"Wilcoxon signed-rank: W={w_stat}, p={fmt(p_value_wilcoxon)}")
    if not np.isnan(ci_low):
        print(f"95% CI for F1 difference (HIF - Full): [{ci_low:.4f}, {ci_high:.4f}]")
    if len(diffs) > 0:
        print(f"Mean F1 improvement: {diffs.mean():.4f}")

    with open(
        output_dir / f"significance_{dataset_id}_{generator_type}_summary.md", "w"
    ) as f:
        f.write(f"# Statistical Significance: {dataset_id}\n\n")
        f.write(
            f"Generator: {generator_type} | N seeds: {n_seeds} | Rows: {n_rows}\n\n"
        )
        f.write("## Utility Filtering Results\n\n")
        f.write("| Variant | Retention% | F1 (mean ± SEM) | Accuracy (mean ± SEM) |\n")
        f.write("|---|---|---|---|\n")
        f.write(
            f"| Full synthetic | 100.0% | {np.nanmean(full_f1s_arr):.3f} ± "
            f"{stats.sem(full_f1s_arr[~np.isnan(full_f1s_arr)]):.3f} | "
            f"{np.nanmean(full_accs):.3f} ± "
            f"{stats.sem(np.array(full_accs)[~np.isnan(full_accs)]):.3f} |\n"
        )
        f.write(
            f"| Rule-only | {np.nanmean(retentions_rule):.1f}% | {np.nanmean(rule_f1s_arr):.3f} ± "
            f"{stats.sem(rule_f1s_arr[~np.isnan(rule_f1s_arr)]):.3f} | "
            f"{np.nanmean(rule_accs):.3f} ± "
            f"{stats.sem(np.array(rule_accs)[~np.isnan(rule_accs)]):.3f} |\n"
        )
        f.write(
            f"| **HIF Oracle** | **{np.nanmean(retentions_hif):.1f}%** | "
            f"**{np.nanmean(hif_f1s_arr):.3f} ± "
            f"{stats.sem(hif_f1s_arr[~np.isnan(hif_f1s_arr)]):.3f}** | "
            f"**{np.nanmean(hif_accs):.3f} ± "
            f"{stats.sem(np.array(hif_accs)[~np.isnan(hif_accs)]):.3f}** |\n"
        )
        f.write("\n## Statistical Tests\n\n")
        f.write(
            f"- **Paired t-test** (Full vs HIF): t={t_stat:.3f}, p={fmt(p_value_ttest)}\n"
        )
        if not np.isnan(p_value_wilcoxon):
            f.write(
                f"- **Wilcoxon signed-rank**: W={w_stat}, p={fmt(p_value_wilcoxon)}\n"
            )
        if not np.isnan(ci_low):
            f.write(
                f"- **95% CI** for F1 difference (HIF - Full): [{ci_low:.4f}, {ci_high:.4f}]\n"
            )
        if len(diffs) > 0:
            f.write(f"- **Mean F1 improvement**: {diffs.mean():.4f}\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Significance Tests")
    parser.add_argument("--dataset", default="census_acs")
    parser.add_argument("--target", default="household_income")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--generator",
        default="gaussian_copula",
        choices=["gaussian_copula", "ctgan", "tvae", "vine"],
    )
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    run_significance_test(
        dataset_id=args.dataset,
        target=args.target,
        n_rows=args.rows,
        n_seeds=args.seeds,
        generator_type=args.generator,
        output_dir=Path(args.output_dir),
    )
