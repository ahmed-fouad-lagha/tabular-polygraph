"""
Experiment: Statistical Significance Tests.

Q:"Are the observed utility gains statistically significant ($p < 0.05$)?"

Re-runs utility filtering with 10 seeds and computes paired statistical tests.

Addresses:
  - Q1: "Are any of the F1 differences statistically significant?"
  - Q2: "Please report mean ± error (SEM/SD) for the N=3 seed results"

Run:
    python scripts/07_statistical_significance.py --dataset census_acs --rows 2000 --seeds 10
"""

from __future__ import annotations

import argparse

# ruff: noqa: E402
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tabular_polygraph.dataset import load_dataset
from tabular_polygraph.fidelity import hif_score
from tabular_polygraph.fidelity.downstream import tstr_score
from tabular_polygraph.fidelity.logical import rule_violation_score
from tabular_polygraph.generators import CTGANGenerator, GaussianCopulaGenerator


def _compute_utility(real, syn, target, seed, num_cols, cat_cols):
    """Compute TSTR metrics for a given real/synthetic pair."""
    real_util = real.copy()
    syn_util = syn.copy()
    encoded_cols = []

    for col in cat_cols:
        if col == target or col not in syn.columns:
            continue
        if real[col].nunique() > 50:
            continue
        dummies = pd.get_dummies(real[col], prefix=f"ohe__{col}").astype(float)
        real_util = pd.concat([real_util, dummies], axis=1)
        syn_dummies = pd.get_dummies(syn[col], prefix=f"ohe__{col}").astype(float)
        for d_col in dummies.columns:
            syn_util[d_col] = (
                syn_dummies[d_col] if d_col in syn_dummies.columns else 0.0
            )
        encoded_cols.extend(dummies.columns)

    feature_cols = [
        c for c in num_cols if c != target and c in syn.columns
    ] + encoded_cols
    if not feature_cols:
        return {"f1": np.nan, "accuracy": np.nan}

    u_real, u_syn = real_util.copy(), syn_util.copy()
    if pd.api.types.is_numeric_dtype(u_real[target]) and u_real[target].nunique() > 2:
        m = u_real[target].median()
        u_real[target] = (u_real[target] > m).astype(int)
        u_syn[target] = (u_syn[target] > m).astype(int)

    try:
        result = tstr_score(
            u_real,
            u_syn,
            target_col=target,
            feature_cols=feature_cols,
            task="classification",
            seed=seed,
        )
        if "error" in result:
            return {"f1": np.nan, "accuracy": np.nan}
        return {"f1": float(result.get("tstr_score", np.nan)), "accuracy": np.nan}
    except Exception:
        return {"f1": np.nan, "accuracy": np.nan}


def run_significance_test(
    dataset_id: str,
    target: str,
    n_rows: int,
    n_seeds: int,
    generator_type: str,
    output_dir: Path,
):
    print(f"Loading dataset: {dataset_id}")
    real = load_dataset(dataset_id, n=n_rows)

    print(f"Fitting generator: {generator_type}")
    if generator_type == "ctgan":
        gen = CTGANGenerator()
    else:
        gen = GaussianCopulaGenerator()
    gen.fit(real)

    all_cols = real.columns.tolist()
    num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(real[c])]
    cat_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(real[c])]

    full_f1s = []
    rule_f1s = []
    hif_f1s = []
    full_accs = []
    rule_accs = []
    hif_accs = []
    retentions_rule = []
    retentions_hif = []
    hif_scores_list = []

    for seed_i in range(n_seeds):
        seed = 42 + seed_i
        print(f"\nSeed {seed} ({seed_i + 1}/{n_seeds})")

        syn = gen.generate(n_rows, seed=seed)
        syn = syn.drop(columns=["syn_id"], errors="ignore")
        common_cols = [c for c in all_cols if c in syn.columns]
        syn = syn[common_cols]

        # Full synthetic
        util_full = _compute_utility(real, syn, target, seed, num_cols, cat_cols)
        full_f1s.append(util_full["f1"])
        full_accs.append(util_full["accuracy"])

        # Rule-only filtering
        rule_result = rule_violation_score(
            real,
            syn,
            columns=common_cols,
            min_confidence=0.95,
            min_support=0.005,
            max_rules=25,
            min_lift=1.0,
            max_antecedents=2,
        )
        rule_mask = rule_result.get("row_violation_mask", np.zeros(len(syn)))
        syn_rule = syn[rule_mask < 0.5]
        retentions_rule.append(len(syn_rule) / len(syn) * 100)
        util_rule = (
            _compute_utility(real, syn_rule, target, seed, num_cols, cat_cols)
            if len(syn_rule) > 10
            else {"f1": np.nan, "accuracy": np.nan}
        )
        rule_f1s.append(util_rule["f1"])
        rule_accs.append(util_rule["accuracy"])

        # HIF Oracle filtering
        hif_result = hif_score(
            real, syn, columns=common_cols, random_state=seed, verbose=False
        )
        hif_scores_list.append(hif_result["hif_score"])
        syn_hif = syn[hif_result["row_penalties"] < 0.5]
        retentions_hif.append(len(syn_hif) / len(syn) * 100)
        util_hif = (
            _compute_utility(real, syn_hif, target, seed, num_cols, cat_cols)
            if len(syn_hif) > 10
            else {"f1": np.nan, "accuracy": np.nan}
        )
        hif_f1s.append(util_hif["f1"])
        hif_accs.append(util_hif["accuracy"])

        print(
            f"  Full F1={util_full['f1']:.3f} | Rule F1={util_rule['f1']:.3f} | HIF F1={util_hif['f1']:.3f}"
        )

    # Convert to arrays
    full_f1s = np.array(full_f1s)
    rule_f1s = np.array(rule_f1s)
    hif_f1s = np.array(hif_f1s)

    # Remove NaN pairs for paired tests
    valid_mask = ~(np.isnan(full_f1s) | np.isnan(hif_f1s))

    # Paired t-test: Full vs HIF
    if valid_mask.sum() >= 3:
        t_stat, p_value_ttest = stats.ttest_rel(
            full_f1s[valid_mask], hif_f1s[valid_mask]
        )
        # Wilcoxon signed-rank (non-parametric alternative)
        try:
            w_stat, p_value_wilcoxon = stats.wilcoxon(
                full_f1s[valid_mask], hif_f1s[valid_mask]
            )
        except ValueError:
            w_stat, p_value_wilcoxon = np.nan, np.nan
    else:
        t_stat, p_value_ttest = np.nan, np.nan
        w_stat, p_value_wilcoxon = np.nan, np.nan

    # 95% CI for the difference
    diffs = hif_f1s[valid_mask] - full_f1s[valid_mask]
    ci_low, ci_high = (
        stats.t.interval(0.95, len(diffs) - 1, loc=diffs.mean(), scale=stats.sem(diffs))
        if len(diffs) > 1
        else (np.nan, np.nan)
    )

    # Save raw results
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(
        {
            "seed": [42 + i for i in range(n_seeds)],
            "full_f1": full_f1s,
            "rule_f1": rule_f1s,
            "hif_f1": hif_f1s,
            "full_acc": full_accs,
            "rule_acc": rule_accs,
            "hif_acc": hif_accs,
            "retention_rule": retentions_rule,
            "retention_hif": retentions_hif,
            "hif_score": hif_scores_list,
        }
    )
    raw_df.to_csv(output_dir / f"significance_{dataset_id}_raw.csv", index=False)

    # Print results
    print("\n" + "=" * 80)
    print(f"Statistical Significance ({dataset_id})")
    print("=" * 80)
    print()
    print("| Variant | Retention% | F1 (mean ± SEM) | Accuracy (mean ± SEM) |")
    print("|---|---|---|---|")
    print(
        f"| Full synthetic | 100.0% | {np.nanmean(full_f1s):.3f} ± {stats.sem(full_f1s[~np.isnan(full_f1s)]):.3f} | {np.nanmean(full_accs):.3f} ± {stats.sem(np.array(full_accs)[~np.isnan(full_accs)]):.3f} |"
    )
    print(
        f"| Rule-only | {np.nanmean(retentions_rule):.1f}% | {np.nanmean(rule_f1s):.3f} ± {stats.sem(rule_f1s[~np.isnan(rule_f1s)]):.3f} | {np.nanmean(rule_accs):.3f} ± {stats.sem(np.array(rule_accs)[~np.isnan(rule_accs)]):.3f} |"
    )
    print(
        f"| **HIF Oracle** | **{np.nanmean(retentions_hif):.1f}%** | **{np.nanmean(hif_f1s):.3f} ± {stats.sem(hif_f1s[~np.isnan(hif_f1s)]):.3f}** | **{np.nanmean(hif_accs):.3f} ± {stats.sem(np.array(hif_accs)[~np.isnan(hif_accs)]):.3f}** |"
    )
    print()
    print(f"Paired t-test (Full vs HIF): t={t_stat:.3f}, p={p_value_ttest:.4f}")
    print(
        f"Wilcoxon signed-rank: W={w_stat}, p={p_value_wilcoxon:.4f}"
        if not np.isnan(p_value_wilcoxon)
        else "Wilcoxon: N/A"
    )
    print(
        f"95% CI for F1 difference (HIF - Full): [{ci_low:.4f}, {ci_high:.4f}]"
        if not np.isnan(ci_low)
        else "CI: N/A"
    )
    print(f"Mean F1 improvement: {diffs.mean():.4f}" if len(diffs) > 0 else "")

    # Significance stars
    if not np.isnan(p_value_ttest):
        if p_value_ttest < 0.001:
            sig = "***"
        elif p_value_ttest < 0.01:
            sig = "**"
        elif p_value_ttest < 0.05:
            sig = "*"
        else:
            sig = "n.s."
    else:
        sig = "N/A"
    print(f"Significance: {sig}")

    # Write markdown summary
    with open(output_dir / f"significance_{dataset_id}_summary.md", "w") as f:
        f.write(f"# Statistical Significance: {dataset_id}\n\n")
        f.write(
            f"Generator: {generator_type} | N seeds: {n_seeds} | Rows: {n_rows}\n\n"
        )
        f.write("## Utility Filtering Results\n\n")
        f.write("| Variant | Retention% | F1 (mean ± SEM) | Accuracy (mean ± SEM) |\n")
        f.write("|---|---|---|---|\n")
        f.write(
            f"| Full synthetic | 100.0% | {np.nanmean(full_f1s):.3f} ± {stats.sem(full_f1s[~np.isnan(full_f1s)]):.3f} | {np.nanmean(full_accs):.3f} ± {stats.sem(np.array(full_accs)[~np.isnan(full_accs)]):.3f} |\n"
        )
        f.write(
            f"| Rule-only | {np.nanmean(retentions_rule):.1f}% | {np.nanmean(rule_f1s):.3f} ± {stats.sem(rule_f1s[~np.isnan(rule_f1s)]):.3f} | {np.nanmean(rule_accs):.3f} ± {stats.sem(np.array(rule_accs)[~np.isnan(rule_accs)]):.3f} |\n"
        )
        f.write(
            f"| **HIF Oracle** | **{np.nanmean(retentions_hif):.1f}%** | **{np.nanmean(hif_f1s):.3f} ± {stats.sem(hif_f1s[~np.isnan(hif_f1s)]):.3f}** | **{np.nanmean(hif_accs):.3f} ± {stats.sem(np.array(hif_accs)[~np.isnan(hif_accs)]):.3f}** |\n"
        )
        f.write("\n## Statistical Tests\n\n")
        f.write(
            f"- **Paired t-test** (Full vs HIF): t={t_stat:.3f}, p={p_value_ttest:.4f} {sig}\n"
        )
        if not np.isnan(p_value_wilcoxon):
            f.write(
                f"- **Wilcoxon signed-rank**: W={w_stat}, p={p_value_wilcoxon:.4f}\n"
            )
        if not np.isnan(ci_low):
            f.write(
                f"- **95% CI** for F1 difference (HIF - Full): [{ci_low:.4f}, {ci_high:.4f}]\n"
            )
        f.write(
            f"- **Mean F1 improvement**: {diffs.mean():.4f}\n" if len(diffs) > 0 else ""
        )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Significance Tests")
    parser.add_argument("--dataset", default="census_acs")
    parser.add_argument("--target", default="household_income")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--generator", default="gaussian_copula", choices=["gaussian_copula", "ctgan"]
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
