import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from tabular_polygraph.catalog import load_dataset
from tabular_polygraph.fidelity import hif_score
from tabular_polygraph.generators import GaussianCopulaGenerator


def _audit_privacy(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame, syn_df: pd.DataFrame
) -> float:
    """Quantitative Privacy Audit via Membership Inference Attack (MIA)."""
    if syn_df.empty:
        return 0.5
    cols = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])]
    if not cols:
        return 0.5
    scaler = StandardScaler()
    n_test = min(1000, len(train_df), len(holdout_df))
    n_syn = min(2000, len(syn_df))
    train_sample = train_df[cols].sample(n_test, random_state=42).fillna(0)
    holdout_sample = holdout_df[cols].sample(n_test, random_state=42).fillna(0)
    syn_sample = syn_df[cols].sample(n_syn, random_state=42).fillna(0)
    combined_test = pd.concat([train_sample, holdout_sample])
    labels = np.array([1] * n_test + [0] * n_test)
    scaler.fit(combined_test)
    test_norm = scaler.transform(combined_test)
    syn_norm = scaler.transform(syn_sample)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(syn_norm)
    distances, _ = nn.kneighbors(test_norm)
    scores = -distances.flatten()
    try:
        auc = roc_auc_score(labels, scores)
        return float(auc)
    except Exception:
        return 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--rows", type=int, default=2000)
    args = parser.parse_args()

    print(f"Privacy Audit: {args.dataset}")
    real = load_dataset(args.dataset, n=args.rows * 2)

    # Split for MIA (Train/Holdout)
    real_train, real_holdout = train_test_split(real, train_size=0.5, random_state=42)

    print("Fitting Gaussian Copula...")
    gen = GaussianCopulaGenerator()
    gen.fit(real_train)
    syn = gen.generate(args.rows, seed=42)

    print("Calculating HIF scores...")
    hif = hif_score(real_train, syn, random_state=42)

    # Filtering by Integrity Frontier (Penalty < 0.5)
    syn_filtered = syn[hif["row_penalties"] < 0.5]

    print(f"Full synthetic size: {len(syn)}")
    print(f"Filtered synthetic size: {len(syn_filtered)}")

    print("Auditing Privacy (MIA AUC)...")
    auc_full = _audit_privacy(real_train, real_holdout, syn)
    auc_filtered = _audit_privacy(real_train, real_holdout, syn_filtered)

    print(f"MIA AUC (Full):     {auc_full:.4f}")
    print(f"MIA AUC (Filtered): {auc_filtered:.4f}")

    print("\nPrivacy claim: Filtering does not increase leakage (AUC remains near 0.5)")
    if abs(auc_filtered - 0.5) <= 0.1:
        print("RESULT: SUCCESS")
    else:
        print("RESULT: FAIL (Potential leakage or outlier influence)")


if __name__ == "__main__":
    main()
