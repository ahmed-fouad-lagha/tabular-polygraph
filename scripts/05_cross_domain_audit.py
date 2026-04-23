"""
Example 5: Cross-Architecture & Multi-Domain Integrity Audit.

This script evaluates the Hybrid Integrity Framework (HIF) across diverse
architectures and datasets to produce Table 2 and Table 3 for the manuscript.
- Datasets: census_acs, world_bank
- Generators: GaussianCopula, VineCopula, CTGAN

Runs N_SEEDS independent experiments and reports mean ± std.

Usage:
    python scripts/05_cross_domain_audit.py --rows 500 --seeds 3 --epochs 150
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# ruff: noqa: E402
PROJECT_ROOT = Path(__file__).resolve().parents[1]


from tabular_polygraph.catalog.downloader import load_cached
from tabular_polygraph.generators import GaussianCopulaGenerator, VineCopulaGenerator, CTGANGenerator
from tabular_polygraph.fidelity import fidelity_report
from tabular_polygraph.utils import DEFAULT_DROP_LIST


def _get_generator(gen_type: str, epochs: int = 150):
    if gen_type == "gaussian":
        return GaussianCopulaGenerator()
    elif gen_type == "vine":
        return VineCopulaGenerator()
    elif gen_type == "ctgan":
        return CTGANGenerator(epochs=epochs, batch_size=500)
    else:
        raise ValueError(f"Unknown generator type: {gen_type}")


def load_real_data(dataset_id: str) -> pd.DataFrame:
    df = load_cached(dataset_id)
    if df is None:
        from tabular_polygraph.catalog import load_dataset
        df = load_dataset(dataset_id, n=50000)
    drop = [c for c in DEFAULT_DROP_LIST if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    return df.dropna().reset_index(drop=True)


def run_single(real_df, gen_type, rows, seed, epochs):
    """Run a single generate + evaluate cycle. Returns a result dict."""
    np.random.seed(seed)

    gen = _get_generator(gen_type, epochs=epochs)
    gen.fit(real_df)
    syn = gen.generate(rows, seed=seed)
    syn_body = syn.drop(columns=["syn_id"], errors="ignore")

    report = fidelity_report(
        real_df,
        syn_body,
        dataset_type="cross_sectional",
        include_downstream=False,
        random_state=seed,
        rule_min_confidence=0.95,
        rule_min_support=0.005,
        rule_max_rules=25,
        rule_min_lift=1.0,
        rule_max_antecedents=2,
    )

    s = report["summary"]
    lg = report.get("logical", {})

    return {
        "ks_score": report["distribution_fit"]["mean_score"],
        "mm_score": report["moment_matching"]["mean_score"],
        "joint_score": s.get("joint_score", 0),
        "hif_score_pct": lg.get("hif_score_pct", 0),
        "violation_rate_pct": lg.get("hif_violation_rate_pct", 0),
        "rule_violation_rate_pct": lg.get("rule_violation_rate_pct", 0),
        "nic_violation_rate_pct": lg.get("nic_violation_rate_pct", 0),
        "rules_mined": lg.get("num_rules_mined", 0),
        "num_violations": lg.get("num_hif_violations", 0),
        "fidelity_pillar": s["pillars"]["fidelity"],
        "logic_pillar": s["pillars"]["logic"],
        "privacy_pillar": s["pillars"]["privacy"],
        "hybrid_integrity": s["hybrid_integrity"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Architecture Maturity Audit for NeurIPS manuscript tables."
    )
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["census_acs"]
    generators = ["gaussian", "vine", "ctgan"]

    print("=" * 70)
    print("  CROSS-ARCHITECTURE MATURITY AUDIT")
    print("=" * 70)
    print(f"  Rows per experiment: {args.rows}")
    print(f"  Seeds: {args.seeds}")
    print(f"  CTGAN epochs: {args.epochs}")
    print()

    all_results = []

    for dataset_id in datasets:
        print(f"\n{'━' * 70}")
        print(f"  Dataset: {dataset_id}")
        print(f"{'━' * 70}")

        real_df = load_real_data(dataset_id)
        print(f"  Real data: {len(real_df)} rows × {len(real_df.columns)} cols")

        for gen_type in generators:
            gen_label = {
                "gaussian": "Gaussian Copula",
                "vine": "Vine Copula",
                "ctgan": "CTGAN (Neural)",
            }[gen_type]

            print(f"\n  ── {gen_label} ──")

            for seed in range(42, 42 + args.seeds):
                print(f"    Seed {seed}...", end="", flush=True)
                t0 = time.time()
                try:
                    result = run_single(real_df, gen_type, args.rows, seed, args.epochs)
                    result["dataset"] = dataset_id
                    result["generator"] = gen_label
                    result["seed"] = seed
                    all_results.append(result)
                    elapsed = time.time() - t0
                    print(
                        f" KS={result['ks_score']:.1f}%  "
                        f"HIF={result['hif_score_pct']:.2f}%  "
                        f"ViolRate={result['violation_rate_pct']:.1f}%  "
                        f"Rules={result['rules_mined']}  "
                        f"({elapsed:.0f}s)"
                    )
                except Exception as e:
                    print(f" ERROR: {e}")

    # ── Summary Tables ────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)

    print("\n\n" + "=" * 70)
    print("  TABLE 2: Cross-Architecture Maturity Audit (for manuscript)")
    print("=" * 70)

    header = f"  {'Architecture':<22} {'KS (↑)':<16} {'HIF Score (↑)':<20} {'Halluc. Rate':<16}"
    print(header)
    print("  " + "-" * 70)

    for gen_name in df["generator"].unique():
        g = df[df["generator"] == gen_name]
        ks_m, ks_s = g["ks_score"].mean() / 100, g["ks_score"].std() / 100
        hif_m, hif_s = g["hif_score_pct"].mean() / 100, g["hif_score_pct"].std() / 100
        vr_m, vr_s = g["violation_rate_pct"].mean(), g["violation_rate_pct"].std()

        print(
            f"  {gen_name:<22} "
            f"{ks_m:.3f}±{ks_s:.3f}    "
            f"{hif_m:.4f}±{hif_s:.4f}      "
            f"{vr_m:.1f}±{vr_s:.1f}%"
        )

    print()
    print("  4-Pillar Breakdown (mean across seeds):")
    print(f"  {'Architecture':<22} {'Fidelity':<12} {'Logic':<12} {'Privacy':<12} {'Hybrid':<12}")
    print("  " + "-" * 60)
    for gen_name in df["generator"].unique():
        g = df[df["generator"] == gen_name]
        print(
            f"  {gen_name:<22} "
            f"{g['fidelity_pillar'].mean():.2f}%    "
            f"{g['logic_pillar'].mean():.2f}%    "
            f"{g['privacy_pillar'].mean():.2f}%    "
            f"{g['hybrid_integrity'].mean():.2f}%"
        )

    # Save raw results
    out_path = out_dir / "architecture_audit.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    csv_path = out_dir / "architecture_audit.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n  Raw results saved → {out_path}")
    print(f"  CSV results saved → {csv_path}")


if __name__ == "__main__":
    main()
