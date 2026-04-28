"""
Example 5: Cross-Architecture Integrity Audit.
Collects "Real Numbers" for Table 2 comparing statistical and neural generators.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from tabular_polygraph.catalog import load_dataset
from tabular_polygraph.fidelity import (
    correlation_distance_score,
    hif_score,
)
from tabular_polygraph.generators import (
    CTGANGenerator,
    ForestDiffusionGenerator,
    GaussianCopulaGenerator,
    VineCopulaGenerator,
)
from tabular_polygraph.utils import numeric_columns


def _compute_ks(real: pd.DataFrame, syn: pd.DataFrame, cols: list[str]) -> float:
    scores = []
    for col in cols:
        s, _ = ks_2samp(real[col].dropna(), syn[col].dropna())
        scores.append(1.0 - s)
    return float(np.mean(scores)) if scores else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--output", type=str, default="results/architecture_audit.json")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    real = load_dataset(args.dataset, n=args.rows)
    num_cols = numeric_columns(real)
    cat_cols = [c for c in real.columns if c not in num_cols]

    generators = {
        "Gaussian Copula": GaussianCopulaGenerator(),
        "Vine Copula": VineCopulaGenerator(),
        "CTGAN": CTGANGenerator(),
        "ForestDiffusion": ForestDiffusionGenerator(),
    }

    results = []
    for name, gen in generators.items():
        print(f"\nAuditing {name}...", flush=True)
        for seed in seeds:
            print(f"  seed={seed}", flush=True)
            try:
                gen.fit(real)
                syn = gen.generate(args.rows, seed=seed)
                syn = syn.drop(columns=["syn_id"], errors="ignore")

                hif = hif_score(
                    real,
                    syn,
                    columns=cat_cols + num_cols,
                    random_state=seed,
                    verbose=False,
                )
                ks = _compute_ks(real, syn, num_cols)
                jcd = correlation_distance_score(real, syn, num_cols)

                results.append(
                    {
                        "architecture": name,
                        "seed": seed,
                        "hif": float(hif["hif_score"]),
                        "violation_rate": float(hif["violation_rate"]),
                        "ks_fidelity": ks,
                        "jcd_fidelity": float(jcd),
                    }
                )
            except Exception as e:
                print(f"  FAILED: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        summary = df.groupby("architecture").agg(["mean", "std"])
        # Flatten MultiIndex columns for JSON serialization
        summary.columns = [f"{c[0]}_{c[1]}" for c in summary.columns]
        summary_dict = summary.to_dict(orient="index")
    else:
        summary_dict = {}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary_dict, f, indent=2)

    print("\n" + "=" * 50)
    print("ARCHITECTURE AUDIT SUMMARY (Real Numbers)")
    print("=" * 50)
    if not df.empty:
        pivot = df.groupby("architecture")[
            ["ks_fidelity", "jcd_fidelity", "hif", "violation_rate"]
        ].mean()
        print(pivot)
    else:
        print("No results to display.")
    print("=" * 50)


if __name__ == "__main__":
    main()
