"""
Example 4: LCV metric validation for research reporting.

This script runs five checks to decide whether the current LCV design is
scientifically useful:
1) Corruption monotonicity
2) External validity
3) Seed stability
4) Feature dominance
5) Practical separability vs distributional metrics

Run:
    python examples/04_lcv_validation.py \
      --dataset census_acs \
      --rows 2000 \
      --seeds 42,43,44,45,46 \
      --corruption-levels 0,0.1,0.2,0.4,0.6 \
      --target household_income
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ruff: noqa: E402
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.catalog import load_dataset
from src.catalog.downloader import load_cached
from src.generators import GaussianCopulaGenerator
from src.fidelity import (
    correlation_distance_score,
    lcv_score,
    mean_moment_matching_score,
    moment_matching_scores,
)
from src.fidelity.downstream import tstr_score
from src.fidelity.logical import rule_violation_score
from src.utils import numeric_columns


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_str_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _load_real(dataset_id: str, fit_rows: int | None) -> pd.DataFrame:
    if fit_rows is None:
        cached = load_cached(dataset_id)
        if cached is not None and len(cached) > 0:
            return cached.reset_index(drop=True)
    return load_dataset(dataset_id, n=fit_rows or 2000)


def _drop_cols(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    if not drop_cols:
        return df
    present = [c for c in drop_cols if c in df.columns]
    if not present:
        return df
    return df.drop(columns=present)


def _generate_synthetic(real: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    gen = GaussianCopulaGenerator()
    gen.fit(real)
    syn = gen.generate(rows, seed=seed)
    return syn.drop(columns=["syn_id"], errors="ignore")


def _corrupt_categorical(
    syn: pd.DataFrame,
    real: pd.DataFrame,
    cat_cols: list[str],
    corruption_level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if corruption_level <= 0.0 or not cat_cols:
        return syn.copy()

    out = syn.copy()
    for col in cat_cols:
        if col not in out.columns or col not in real.columns:
            continue
        pool = real[col].dropna().astype(str).to_numpy()
        if len(pool) == 0:
            continue

        mask = rng.random(len(out)) < corruption_level
        n = int(mask.sum())
        if n == 0:
            continue

        out.loc[mask, col] = rng.choice(pool, size=n, replace=True)
    return out


def _antecedent_features_from_rule(rule: dict) -> set[str]:
    features: set[str] = set()

    ant_feature = rule.get("antecedent_feature")
    if ant_feature:
        features.add(str(ant_feature))

    ant_repr = rule.get("antecedent_repr")
    if ant_repr:
        for clause in str(ant_repr).split(" AND "):
            if "=" in clause:
                features.add(clause.split("=", 1)[0].strip())

    return features


def _rule_involved_features(rule: dict) -> set[str]:
    features = _antecedent_features_from_rule(rule)
    cons_feature = rule.get("consequent_feature")
    if cons_feature:
        features.add(str(cons_feature))
    return features or {"unknown"}


def _feature_dominance_share(rule_result: dict) -> float:
    total_violations = max(int(rule_result.get("num_rule_violations", 0)), 1)
    counts: dict[str, float] = {}

    for rule in rule_result.get("top_violated_rules", []):
        violation_count = int(rule.get("violation_count", 0))
        if violation_count <= 0:
            continue

        features = _rule_involved_features(rule)

        weight_per_feature = float(violation_count) / float(len(features))
        for feature in features:
            counts[feature] = counts.get(feature, 0.0) + weight_per_feature

    if not counts or total_violations <= 0:
        return 0.0
    return float(max(counts.values()) / float(total_violations))


def _utility_feature_columns(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target: str,
    num_cols: list[str],
    cat_cols: list[str],
    feature_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build utility features aligned to evaluation intent.

    categorical_target_encoded:
      Map categorical values to real-data target means (cheap and corruption-sensitive).
    hybrid:
      Use both numeric columns and target-encoded categorical columns.
    numeric:
      Use numeric columns only (legacy behavior).
    """
    if feature_mode == "numeric":
        return (
            real,
            syn,
            [
                c
                for c in num_cols
                if c != target and c in real.columns and c in syn.columns
            ],
        )

    real_util = real.copy()
    syn_util = syn.copy()

    target_mean = float(pd.to_numeric(real[target], errors="coerce").mean())
    encoded_cols: list[str] = []
    for col in cat_cols:
        if col not in real.columns or col not in syn.columns or col == target:
            continue

        mapping_frame = real[[col, target]].dropna()
        if mapping_frame.empty:
            continue

        means = (
            mapping_frame.assign(
                **{target: pd.to_numeric(mapping_frame[target], errors="coerce")}
            )
            .dropna(subset=[target])
            .groupby(col)[target]
            .mean()
        )
        if means.empty:
            continue

        out_col = f"te__{col}"
        real_util[out_col] = (
            real[col].astype(str).map(means).fillna(target_mean).astype(float)
        )
        syn_util[out_col] = (
            syn[col].astype(str).map(means).fillna(target_mean).astype(float)
        )
        encoded_cols.append(out_col)

    if feature_mode == "categorical_target_encoded":
        return real_util, syn_util, encoded_cols
    if feature_mode == "hybrid":
        numeric = [
            c
            for c in num_cols
            if c != target and c in real.columns and c in syn.columns
        ]
        return real_util, syn_util, numeric + encoded_cols

    raise ValueError(
        "Unknown utility_feature_mode: "
        f"{feature_mode}. Use 'numeric', 'categorical_target_encoded', or 'hybrid'."
    )


def _evaluate_once(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    target: str | None,
    utility_feature_mode: str,
    lcv_epochs: int,
    seed: int,
    lcv_weighting: str,
    compare_uniform: bool,
) -> dict:
    lcv = lcv_score(
        real,
        syn,
        columns=cat_cols,
        epochs=lcv_epochs,
        random_state=seed,
        feature_weighting=lcv_weighting,
        verbose=False,
    )

    lcv_uniform_score = np.nan
    lcv_uniform_vr = np.nan
    if compare_uniform:
        lcv_uniform = lcv_score(
            real,
            syn,
            columns=cat_cols,
            epochs=lcv_epochs,
            random_state=seed,
            feature_weighting="uniform",
            verbose=False,
        )
        lcv_uniform_score = float(lcv_uniform["lcv_score"])
        lcv_uniform_vr = float(lcv_uniform["violation_rate"])

    rules = rule_violation_score(
        real,
        syn,
        columns=cat_cols,
        min_confidence=0.95,
        min_support=0.005,
        max_rules=25,
        min_lift=1.0,
        max_antecedents=1,
    )

    mm = mean_moment_matching_score(moment_matching_scores(real, syn, num_cols))
    joint = correlation_distance_score(real, syn, num_cols)

    utility_ratio = np.nan
    if target and target in real.columns and target in syn.columns:
        real_util, syn_util, util_features = _utility_feature_columns(
            real=real,
            syn=syn,
            target=target,
            num_cols=num_cols,
            cat_cols=cat_cols,
            feature_mode=utility_feature_mode,
        )
        if util_features:
            util = tstr_score(
                real_util,
                syn_util,
                target_col=target,
                feature_cols=util_features,
                seed=seed,
            )
            if "error" not in util:
                utility_ratio = float(util.get("ratio", np.nan))

    return {
        "lcv_score": float(lcv["lcv_score"]),
        "lcv_violation_rate": float(lcv["violation_rate"]),
        "lcv_score_uniform": lcv_uniform_score,
        "lcv_violation_rate_uniform": lcv_uniform_vr,
        "rule_violation_rate": float(rules["rule_violation_rate"]),
        "num_rule_violations": int(rules["num_rule_violations"]),
        "num_rules_mined": int(rules["num_rules_mined"]),
        "moment_matching_score": float(mm),
        "joint_score": float(joint),
        "utility_ratio": utility_ratio,
        "dominant_feature_share": float(_feature_dominance_share(rules)),
    }


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    res = spearmanr(x, y)
    rho = float(np.asarray(getattr(res, "statistic", np.nan)).reshape(-1)[0])
    p = float(np.asarray(getattr(res, "pvalue", 1.0)).reshape(-1)[0])
    if np.isnan(rho):
        return 0.0, 1.0
    return rho, p


def _monotonicity_by_seed(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for seed, sub in df.groupby("seed"):
        sub = sub.sort_values("corruption_level")
        rho, p = _safe_spearman(
            sub["corruption_level"].to_numpy(), sub[metric].to_numpy()
        )
        rows.append({"seed": seed, "rho": rho, "pvalue": p})
    return pd.DataFrame(rows)


def _weighted_monotonicity_equivalent(
    weighted_strength: float,
    uniform_strength: float,
    has_uniform: bool,
    tolerance: float = 0.02,
) -> bool:
    if not has_uniform or np.isnan(uniform_strength):
        return True
    return weighted_strength + tolerance >= uniform_strength


def _compute_summary(df: pd.DataFrame, has_utility: bool) -> dict:
    mono_lcv = _monotonicity_by_seed(df, "lcv_score")
    mono_rule = _monotonicity_by_seed(df, "rule_violation_rate")
    has_uniform = bool(
        "lcv_score_uniform" in df.columns and df["lcv_score_uniform"].notna().any()
    )
    mono_lcv_uniform = (
        _monotonicity_by_seed(
            df.dropna(subset=["lcv_score_uniform"]), "lcv_score_uniform"
        )
        if has_uniform
        else pd.DataFrame()
    )

    lcv_rho_mean = float(mono_lcv["rho"].mean()) if not mono_lcv.empty else 0.0
    rule_rho_mean = float(mono_rule["rho"].mean()) if not mono_rule.empty else 0.0
    lcv_uniform_rho_mean = (
        float(mono_lcv_uniform["rho"].mean()) if not mono_lcv_uniform.empty else np.nan
    )
    weighted_monotonicity_strength = abs(min(lcv_rho_mean, 0.0))
    uniform_monotonicity_strength = abs(min(float(lcv_uniform_rho_mean), 0.0))
    weighted_vs_uniform_gap = (
        weighted_monotonicity_strength - uniform_monotonicity_strength
        if has_uniform
        else np.nan
    )

    ext_lcv_vs_rule = _safe_spearman(
        df["lcv_score"].to_numpy(), df["rule_violation_rate"].to_numpy()
    )
    ext_lcv_uniform_vs_rule = (np.nan, np.nan)
    if has_uniform:
        ext_lcv_uniform_vs_rule = _safe_spearman(
            df["lcv_score_uniform"].to_numpy(), df["rule_violation_rate"].to_numpy()
        )
    ext_lcv_vs_util = (np.nan, np.nan)
    if has_utility:
        valid = df.dropna(subset=["utility_ratio"])
        if len(valid) > 2:
            ext_lcv_vs_util = _safe_spearman(
                valid["lcv_score"].to_numpy(), valid["utility_ratio"].to_numpy()
            )

    grouped = df.groupby("corruption_level")["lcv_score"]
    lcv_std_by_level = grouped.std(ddof=0).fillna(0.0)
    mean_lcv_std = float(lcv_std_by_level.mean()) if len(lcv_std_by_level) else 0.0
    weighted_minus_uniform_mean = (
        float((df["lcv_score"] - df["lcv_score_uniform"]).mean())
        if has_uniform
        else np.nan
    )

    dominance_mean = float(df["dominant_feature_share"].mean())
    dominance_max = float(df["dominant_feature_share"].max())

    # Practical separability: LCV should track corruption at least as strongly
    # as moment matching and joint score in absolute rank-correlation magnitude.
    better_count = 0
    total = 0
    for _, sub in df.groupby("seed"):
        sub = sub.sort_values("corruption_level")
        x = sub["corruption_level"].to_numpy()
        rho_lcv, _ = _safe_spearman(x, sub["lcv_score"].to_numpy())
        rho_mm, _ = _safe_spearman(x, sub["moment_matching_score"].to_numpy())
        rho_joint, _ = _safe_spearman(x, sub["joint_score"].to_numpy())

        # For these metrics, stronger degradation means more negative rho.
        lcv_strength = abs(min(rho_lcv, 0.0))
        mm_strength = abs(min(rho_mm, 0.0))
        joint_strength = abs(min(rho_joint, 0.0))

        better_count += int(lcv_strength > max(mm_strength, joint_strength))
        total += 1

    separability_rate = float(better_count / max(total, 1))

    checks = {
        "monotonicity_lcv": lcv_rho_mean <= -0.8,
        "monotonicity_rule": rule_rho_mean >= 0.8,
        "external_validity_rules": ext_lcv_vs_rule[0] <= -0.6,
        "external_validity_utility": (not has_utility)
        or (not np.isnan(ext_lcv_vs_util[0]) and ext_lcv_vs_util[0] >= 0.4),
        "seed_stability": mean_lcv_std <= 0.05,
        "feature_dominance": dominance_max <= 0.5,
        "practical_separability": separability_rate >= 0.6,
        "weighted_vs_uniform_monotonicity": _weighted_monotonicity_equivalent(
            weighted_monotonicity_strength,
            uniform_monotonicity_strength,
            has_uniform,
        ),
    }

    return {
        "checks": checks,
        "check_pass_rate": float(sum(checks.values()) / len(checks)),
        "stats": {
            "lcv_monotonicity_rho_mean": lcv_rho_mean,
            "lcv_uniform_monotonicity_rho_mean": lcv_uniform_rho_mean,
            "weighted_vs_uniform_monotonicity_gap": weighted_vs_uniform_gap,
            "rule_monotonicity_rho_mean": rule_rho_mean,
            "lcv_vs_rule_rho": ext_lcv_vs_rule[0],
            "lcv_vs_rule_pvalue": ext_lcv_vs_rule[1],
            "lcv_uniform_vs_rule_rho": ext_lcv_uniform_vs_rule[0],
            "lcv_uniform_vs_rule_pvalue": ext_lcv_uniform_vs_rule[1],
            "lcv_vs_utility_rho": ext_lcv_vs_util[0],
            "lcv_vs_utility_pvalue": ext_lcv_vs_util[1],
            "mean_lcv_std_across_corruption_levels": mean_lcv_std,
            "weighted_minus_uniform_mean": weighted_minus_uniform_mean,
            "dominant_feature_share_mean": dominance_mean,
            "dominant_feature_share_max": dominance_max,
            "separability_rate": separability_rate,
        },
    }


def _write_markdown_summary(path: Path, summary: dict) -> None:
    lines = ["# LCV Validation Summary", ""]
    lines.append("## Check Outcomes")
    for k, v in summary["checks"].items():
        status = "PASS" if v else "FAIL"
        lines.append(f"- {k}: {status}")

    lines.append("")
    lines.append("## Key Statistics")
    for k, v in summary["stats"].items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
        else:
            lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append(f"Overall pass rate: {summary['check_pass_rate']:.2%}")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LCV metric validation checks.")
    parser.add_argument("--dataset", type=str, default="census_acs")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--fit-rows", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--corruption-levels", type=str, default="0,0.1,0.2,0.4,0.6")
    parser.add_argument("--target", type=str, default="household_income")
    parser.add_argument(
        "--utility-feature-mode",
        type=str,
        default="categorical_target_encoded",
        choices=["numeric", "categorical_target_encoded", "hybrid"],
        help=(
            "Predictor set for utility check. "
            "Use categorical_target_encoded for LCV-aligned external validity."
        ),
    )
    parser.add_argument("--lcv-epochs", type=int, default=10)
    parser.add_argument(
        "--lcv-weighting",
        type=str,
        default="inverse_log_cardinality",
        choices=["inverse_log_cardinality", "uniform"],
    )
    parser.add_argument(
        "--no-uniform-baseline",
        action="store_true",
        help="Skip side-by-side uniform-weight LCV baseline.",
    )
    parser.add_argument("--drop-cols", type=str, default="tract_id")
    parser.add_argument("--output-dir", type=str, default="examples")
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    levels = _parse_float_list(args.corruption_levels)
    drop_cols = _parse_str_list(args.drop_cols)
    compare_uniform = not args.no_uniform_baseline

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("LCV Validation")
    print("=" * 72)
    print(f"Dataset: {args.dataset}")
    print(f"Seeds: {seeds}")
    print(f"Corruption levels: {levels}")

    real = _load_real(args.dataset, args.fit_rows)
    real = _drop_cols(real, drop_cols)

    num_cols = [c for c in numeric_columns(real) if c in real.columns]
    cat_cols = [c for c in real.columns if c not in num_cols]

    print(
        f"Real rows: {len(real):,} | Numeric cols: {len(num_cols)} | Categorical cols: {len(cat_cols)}"
    )

    rows: list[dict] = []
    for seed in seeds:
        print(f"\n[seed={seed}] fitting + generating base synthetic...", flush=True)
        base_syn = _generate_synthetic(real, args.rows, seed)

        for level in levels:
            rng = np.random.default_rng(seed * 1000 + int(level * 1000))
            syn = _corrupt_categorical(base_syn, real, cat_cols, level, rng)
            metrics = _evaluate_once(
                real=real,
                syn=syn,
                cat_cols=cat_cols,
                num_cols=num_cols,
                target=args.target,
                utility_feature_mode=args.utility_feature_mode,
                lcv_epochs=args.lcv_epochs,
                seed=seed,
                lcv_weighting=args.lcv_weighting,
                compare_uniform=compare_uniform,
            )
            rows.append(
                {
                    "dataset": args.dataset,
                    "seed": seed,
                    "rows": int(args.rows),
                    "corruption_level": float(level),
                    **metrics,
                }
            )
            print(
                f"  level={level:>4.2f} | lcv_w={metrics['lcv_score']:.4f} | "
                f"lcv_u={metrics['lcv_score_uniform']:.4f} | "
                f"rule_vr={metrics['rule_violation_rate']:.4f} | "
                f"mm={metrics['moment_matching_score']:.2f} | "
                f"joint={metrics['joint_score']:.2f}",
                flush=True,
            )

    results = pd.DataFrame(rows).sort_values(["seed", "corruption_level"])
    has_utility = bool(
        bool(args.target) and bool(results["utility_ratio"].notna().any())
    )
    summary = _compute_summary(results, has_utility=has_utility)

    csv_path = out_dir / "lcv_validation_results.csv"
    json_path = out_dir / "lcv_validation_summary.json"
    md_path = out_dir / "lcv_validation_summary.md"

    results.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2))
    _write_markdown_summary(md_path, summary)

    print("\n" + "-" * 72)
    print("Summary checks")
    for name, ok in summary["checks"].items():
        print(f"  {name:<28} {'PASS' if ok else 'FAIL'}")
    print(f"Overall pass rate: {summary['check_pass_rate']:.2%}")
    print("-" * 72)

    print(f"Saved results : {csv_path}")
    print(f"Saved summary : {json_path}")
    print(f"Saved report  : {md_path}")


if __name__ == "__main__":
    main()
