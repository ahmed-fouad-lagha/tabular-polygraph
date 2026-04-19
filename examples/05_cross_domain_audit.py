"""
Example 5: Cross-Architecture Integrity Audit for research reporting.

This script evaluates HIF across multiple datasets and generators:
- Datasets: census_acs, bls (state-level subset)
- Generators: GaussianCopula, VineCopula, CTGAN

Goal: Confirm that HIF reliably identifies hallucinations across diverse architectures.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ruff: noqa: E402
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.catalog import load_dataset
from src.generators import GaussianCopulaGenerator, VineCopulaGenerator, CTGANGenerator
from src.fidelity import hif_score
from src.utils import numeric_columns

def _get_generator(gen_type: str, epochs: int = 300):
    if gen_type == "gaussian":
        return GaussianCopulaGenerator()
    elif gen_type == "vine":
        return VineCopulaGenerator()
    elif gen_type == "ctgan":
        return CTGANGenerator(epochs=epochs)
    else:
        raise ValueError(f"Unknown generator type: {gen_type}")

def run_audit(dataset_id: str, gen_type: str, rows: int = 2000, seed: int = 42, epochs: int = 100):
    print(f"\n[Audit] Dataset: {dataset_id} | Generator: {gen_type}")
    
    # 1. Load Real Data
    real_df = load_dataset(dataset_id, n=rows * 2)
    
    # 2. Fit and Generate Synthetic
    print(f"  Fitting {gen_type} generator...")
    gen = _get_generator(gen_type, epochs=epochs)
    gen.fit(real_df)
    
    print(f"  Generating {rows} synthetic records...")
    syn_df = gen.generate(rows, seed=seed).drop(columns=["syn_id"], errors="ignore")
    
    # 3. Evaluate Integrity (Base)
    print("  Evaluating HIF base integrity...")
    base_res = hif_score(real_df, syn_df, verbose=False)
    base_hif = base_res["hif_score"]
    
    # 4. Corruption Sensitivity Check
    # Swap 30% of categorical values to force hallucinations
    corrupt_syn = syn_df.copy()
    cat_cols = [c for c in syn_df.columns if c not in numeric_columns(syn_df)]
    for col in cat_cols:
        mask = np.random.rand(len(corrupt_syn)) < 0.3
        corrupt_syn.loc[mask, col] = np.random.choice(real_df[col].unique(), size=mask.sum())
        
    print("  Evaluating HIF corrupted sensitivity...")
    corr_res = hif_score(real_df, corrupt_syn, verbose=False)
    corr_hif = corr_res["hif_score"]
    
    delta = base_hif - corr_hif
    print(f"  Results: Base={base_hif:.4f} | Corrupted={corr_hif:.4f} | Delta={delta:.4f}")
    
    return {
        "dataset": dataset_id,
        "generator": gen_type,
        "base_hif": base_hif,
        "corrupted_hif": corr_hif,
        "delta": delta,
        "status": "PASS" if delta > 0.01 else "FAIL"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50) # Faster for testing
    args = parser.parse_args()
    
    datasets = ["census_acs", "bls"]
    generators = ["gaussian", "vine", "ctgan"]
    
    results = []
    for ds in datasets:
        for gen in generators:
            try:
                res = run_audit(ds, gen, rows=args.rows, epochs=args.epochs)
                results.append(res)
            except Exception as e:
                print(f"  Error auditing {ds} with {gen}: {e}")
                
    df_res = pd.DataFrame(results)
    print("\n" + "="*60)
    print("CROSS-ARCHITECTURE MATURITY AUDIT SUMMARY")
    print("="*60)
    print(df_res)
    print("="*60)
    
    df_res.to_csv("results/cross_architecture_audit.csv", index=False)
    print("\nResults saved to results/cross_architecture_audit.csv")
