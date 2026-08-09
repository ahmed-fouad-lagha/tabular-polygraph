"""
Experiment: HIF Calibration & Differential Diagnostic (ported to current API).

Two protocols:
  1. ``permutation`` — the protocol claimed in the manuscript: shuffle values of
     a column across rows at corruption rate eta. Preserves column marginals but
     necessarily breaks pairwise correlations.
  2. ``conditional_swap`` — targeted row-level conditional violations: replace
     the most-correlated numeric feature of a fraction of rows with a value drawn
     from the *opposite* half of the conditioning feature. Each value is still
     in-domain, so marginals survive, while the row-level joint is broken.

Records HIF, JCD, moment-matching, KS, and rule-violation rate per level.

Run:
    python scripts/02_hif_calibration.py --dataset census_acs --rows 2000 \
      --seeds 42,43,44 --levels 0,0.1,0.2,0.4,0.6 --strategy permutation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ruff: noqa: E402
from _exp_utils import aggregate_metrics, audit_hif, generate, load_real


def corrupt_permutation(
    syn: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Permute values within each column for a fraction of rows.

    Preserves each column's marginal distribution exactly while breaking the
    joint distribution (and, unavoidably, pairwise correlations).
    """
    if level <= 0.0:
        return syn.copy()
    out = syn.copy()
    for col in cat_cols + num_cols:
        if col not in out.columns:
            continue
        mask = rng.random(len(out)) < level
        n = int(mask.sum())
        if n <= 1:
            continue
        vals = out.loc[mask, col].values
        out.loc[mask, col] = rng.permutation(vals)
    return out


def corrupt_conditional_swap(
    syn: pd.DataFrame,
    real: pd.DataFrame,
    level: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Inject marginal-preserving row-level conditional violations.

    Finds the two most strongly correlated numeric features (c1, c2) in the
    real data. For a fraction ``level`` of rows, replaces c2 with a value drawn
    from real rows in the *opposite* half of c1. Every injected value is
    in-domain (marginals intact) but the joint conditional is broken.
    Returns (corrupted, true_labels).
    """
    out = syn.copy()
    n_rows = len(out)
    n_corrupt = max(1, int(n_rows * level))
    row_idx = rng.choice(n_rows, size=n_corrupt, replace=False)
    labels = np.zeros(n_rows, dtype=bool)
    labels[row_idx] = True

    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if len(num_cols) < 2:
        return out, labels

    corr_matrix = real[num_cols].corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    c1, c2 = corr_matrix.stack().idxmax()
    c1, c2 = str(c1), str(c2)
    median_c1 = float(real[c1].median())

    pool_low_c2 = real[real[c1] < median_c1][c2].dropna().values
    pool_high_c2 = real[real[c1] >= median_c1][c2].dropna().values
    fallback = real[c2].dropna().values

    for idx in row_idx:
        row_id = out.index[idx]
        current_c1 = float(out.at[row_id, c1])
        if current_c1 >= median_c1:
            pool = pool_low_c2 if len(pool_low_c2) > 0 else fallback
        else:
            pool = pool_high_c2 if len(pool_high_c2) > 0 else fallback
        new_c2 = rng.choice(pool)
        if pd.api.types.is_integer_dtype(out[c2]):
            new_c2 = int(np.round(float(new_c2)))
        out.at[row_id, c2] = new_c2

    return out, labels


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    res = spearmanr(x, y)
    rho = float(np.asarray(getattr(res, "statistic", np.nan)).reshape(-1)[0])
    p = float(np.asarray(getattr(res, "pvalue", 1.0)).reshape(-1)[0])
    if np.isnan(rho):
        return 0.0, 1.0
    return rho, p


def main() -> None:
    parser = argparse.ArgumentParser(description="HIF calibration experiment")
    parser.add_argument("--dataset", type=str, default="census_acs")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--levels", type=str, default="0,0.1,0.2,0.4,0.6")
    parser.add_argument(
        "--strategy",
        type=str,
        default="permutation",
        choices=["permutation", "conditional_swap"],
    )
    parser.add_argument("--generator", type=str, default="gaussian_copula")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    levels = [float(x.strip()) for x in args.levels.split(",") if x.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"HIF CALIBRATION: strategy={args.strategy} dataset={args.dataset}")
    print("=" * 72)

    real = load_real(args.dataset, n=args.rows)
    cat_cols = [c for c in real.columns if not pd.api.types.is_numeric_dtype(real[c])]
    num_cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]
    print(f"Real rows: {len(real)} | num={num_cols} | cat={cat_cols}")

    rows: list[dict] = []
    for seed in seeds:
        print(f"\n[seed={seed}] generating base synthetic...", flush=True)
        base_syn = generate(real, args.rows, seed, args.generator)
        base_syn = base_syn[real.columns.intersection(base_syn.columns).tolist()]

        for level in levels:
            rng = np.random.default_rng(seed * 1000 + int(round(level * 1000)))

            if args.strategy == "conditional_swap":
                syn, labels = corrupt_conditional_swap(base_syn, real, level, rng)
            else:
                syn = corrupt_permutation(base_syn, cat_cols, num_cols, level, rng)
                labels = np.zeros(len(syn), dtype=bool)

            hif = audit_hif(real, syn, seed=seed)
            agg = aggregate_metrics(real, syn)

            recall = (
                float((hif["row_penalties"] > 0.5)[labels].mean())
                if labels.sum()
                else np.nan
            )
            jcd_clean = aggregate_metrics(real, base_syn)["jcd"]

            rows.append(
                {
                    "dataset": args.dataset,
                    "seed": seed,
                    "corruption_level": float(level),
                    "hif_score": hif["hif_score"],
                    "hif_violation_rate": hif["violation_rate"],
                    "jcd": agg["jcd"],
                    "mm": agg["mm"],
                    "ks": agg["ks"],
                    "rule_violation_rate": hif["rule_violation_rate"],
                    "recall": recall,
                    "jcd_delta": round(float(agg["jcd"] - jcd_clean), 2),
                }
            )
            print(
                f"  level={level:>4.2f} | hif={hif['hif_score']:.4f} | "
                f"jcd={agg['jcd']:.2f} (d={float(agg['jcd'] - jcd_clean):+.2f}) | "
                f"mm={agg['mm']:.2f} | ks={agg['ks']:.2f}",
                flush=True,
            )

    results = pd.DataFrame(rows).sort_values(["seed", "corruption_level"])

    # Monotonicity per metric (Spearman with corruption level)
    metric_cols = ["hif_score", "jcd", "mm", "ks"]
    mono: dict[str, list[float]] = {}
    for metric in metric_cols:
        rhos = []
        for seed in seeds:
            sub = results[results["seed"] == seed].sort_values("corruption_level")
            rho, _ = _safe_spearman(
                sub["corruption_level"].to_numpy(), sub[metric].to_numpy()
            )
            rhos.append(rho)
        mono[metric] = rhos

    print("\n" + "-" * 72)
    print("Monotonicity (Spearman rho with corruption level, mean over seeds)")
    for metric in metric_cols:
        print(f"  {metric:<10} rho = {np.mean(mono[metric]):+.3f}")
    print("-" * 72)

    csv_path = out_dir / f"hif_calibration_{args.strategy}_{args.dataset}.csv"
    results.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
