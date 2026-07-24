"""
Experiment: Held-Out Error Types & Outlier Baseline Comparison.

Tests whether HIF detects error types it was NOT designed for, and compares
against standard outlier detectors (Isolation Forest, LOF).

Addresses:
  - Reviewer mUq7 Q1: "Does the method detect errors other than the dependency
    violations it was tuned for?"
  - Reviewer mUq7 Q2: "How does HIF compare against standard outlier detectors?"

Run:
    python scripts/08_heldout_errors.py --dataset census_acs --rows 2000 --seeds 5
"""

from __future__ import annotations

import argparse

# ruff: noqa: E402
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabular_polygraph.dataset import load_dataset
from tabular_polygraph.fidelity import hif_score
from tabular_polygraph.generators import GaussianCopulaGenerator

# ---------------------------------------------------------------------------
# Held-out corruption strategies (NOT what HIF was designed for)
# ---------------------------------------------------------------------------


def corrupt_random_injection(
    syn: pd.DataFrame, real: pd.DataFrame, level: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Replace random cells with random values from the column's domain.
    Breaks marginals AND dependencies — a generic, non-targeted corruption."""
    out = syn.copy()
    n_rows, n_cols = out.shape
    labels = np.zeros(n_rows, dtype=bool)

    n_corrupt = max(1, int(n_rows * level))
    row_idx = rng.choice(n_rows, size=n_corrupt, replace=False)
    labels[row_idx] = True

    for col in out.columns:
        pool = real[col].dropna().values
        if len(pool) == 0:
            continue
        replacements = rng.choice(pool, size=n_corrupt, replace=True)
        out.iloc[row_idx, out.columns.get_loc(col)] = replacements

    return out, labels


def corrupt_row_duplication(
    syn: pd.DataFrame, level: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Duplicate random rows with small Gaussian noise on numerics.
    Tests near-duplicate / memorization detection."""
    n_corrupt = max(1, int(len(syn) * level))
    source_idx = rng.choice(len(syn), size=n_corrupt, replace=True)
    duplicated = syn.iloc[source_idx].copy().reset_index(drop=True)

    # Add small noise to numeric columns
    num_cols = [
        c for c in duplicated.columns if pd.api.types.is_numeric_dtype(duplicated[c])
    ]
    for col in num_cols:
        std = syn[col].std()
        if std > 0:
            noise = rng.normal(0, std * 0.01, size=n_corrupt)
            duplicated[col] = duplicated[col].values + noise

    # Replace tail of synthetic with duplicated rows
    out = syn.copy()
    replace_idx = rng.choice(len(out), size=n_corrupt, replace=False)
    labels = np.zeros(len(out), dtype=bool)
    labels[replace_idx] = True

    for i, idx in enumerate(replace_idx):
        out.iloc[idx] = duplicated.iloc[i]

    return out, labels


def corrupt_feature_dropout(
    syn: pd.DataFrame, real: pd.DataFrame, level: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Set random features to column mode/median, simulating missing data.
    Tests whether HIF catches records with suspiciously uniform values."""
    out = syn.copy()
    n_rows = len(out)
    n_corrupt = max(1, int(n_rows * level))
    row_idx = rng.choice(n_rows, size=n_corrupt, replace=False)
    labels = np.zeros(n_rows, dtype=bool)
    labels[row_idx] = True

    # For each corrupted row, blank out ~50% of its features
    for idx in row_idx:
        cols_to_blank = rng.choice(
            out.columns.tolist(),
            size=max(1, len(out.columns) // 2),
            replace=False,
        )
        for col in cols_to_blank:
            if pd.api.types.is_numeric_dtype(out[col]):
                out.iloc[idx, out.columns.get_loc(col)] = real[col].median()
            else:
                out.iloc[idx, out.columns.get_loc(col)] = (
                    real[col].mode().iloc[0]
                    if not real[col].mode().empty
                    else out.iloc[idx][col]
                )

    return out, labels


def corrupt_covariate_shift(
    syn: pd.DataFrame, real: pd.DataFrame, level: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Replace fraction of rows with samples from a biased subset of real data.
    Tests distribution shift detection (not row-level logic)."""
    out = syn.copy()
    n_corrupt = max(1, int(len(out) * level))
    row_idx = rng.choice(len(out), size=n_corrupt, replace=False)
    labels = np.zeros(len(out), dtype=bool)
    labels[row_idx] = True

    # Create biased subset: take only top-quartile of the first numeric column
    num_cols_list = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]
    if num_cols_list:
        bias_col = num_cols_list[0]
        threshold = real[bias_col].quantile(0.75)
        biased = real[real[bias_col] >= threshold]
        if len(biased) < n_corrupt:
            biased = real  # fallback
    else:
        biased = real

    replacements = biased.sample(
        n=n_corrupt, replace=True, random_state=int(rng.integers(1e6))
    )
    for i, idx in enumerate(row_idx):
        out.iloc[idx] = replacements.iloc[i]

    return out, labels


# ---------------------------------------------------------------------------
# Baseline outlier detectors
# ---------------------------------------------------------------------------


def _encode_for_outlier_detection(
    real: pd.DataFrame, syn: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """One-hot encode categorical + scale numeric for IF/LOF."""
    cat_cols = [c for c in real.columns if not pd.api.types.is_numeric_dtype(real[c])]
    num_cols_list = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]

    parts_real, parts_syn = [], []

    if num_cols_list:
        scaler = StandardScaler()
        parts_real.append(scaler.fit_transform(real[num_cols_list].fillna(0)))
        parts_syn.append(scaler.transform(syn[num_cols_list].fillna(0)))

    if cat_cols:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        parts_real.append(enc.fit_transform(real[cat_cols].astype(str)))
        parts_syn.append(enc.transform(syn[cat_cols].astype(str)))

    X_real = np.hstack(parts_real) if parts_real else np.zeros((len(real), 1))
    X_syn = np.hstack(parts_syn) if parts_syn else np.zeros((len(syn), 1))
    return X_real, X_syn


def detect_isolation_forest(
    real: pd.DataFrame, syn: pd.DataFrame, contamination: float = 0.2
) -> np.ndarray:
    """Return binary anomaly labels (1=anomaly) using Isolation Forest."""
    X_real, X_syn = _encode_for_outlier_detection(real, syn)
    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    clf.fit(X_real)
    preds = clf.predict(X_syn)
    return (preds == -1).astype(int)


def detect_lof(
    real: pd.DataFrame, syn: pd.DataFrame, contamination: float = 0.2
) -> np.ndarray:
    """Return binary anomaly labels using Local Outlier Factor."""
    X_real, X_syn = _encode_for_outlier_detection(real, syn)
    # LOF novelty detection: fit on real, predict on synthetic
    clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    clf.fit(X_real)
    preds = clf.predict(X_syn)
    return (preds == -1).astype(int)


def detect_hif(real: pd.DataFrame, syn: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Return binary anomaly labels using HIF (H(x) < 0.5 = anomaly)."""
    cols = real.columns.intersection(syn.columns).tolist()
    result = hif_score(real, syn, columns=cols, random_state=seed, verbose=False)
    return (result["row_penalties"] > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    dataset_id: str,
    n_rows: int,
    n_seeds: int,
    corruption_levels: list[float],
    output_dir: Path,
):
    print(f"Loading dataset: {dataset_id}")
    real = load_dataset(dataset_id, n=n_rows)

    # Generate clean synthetic baseline
    print("Generating clean synthetic baseline (Gaussian Copula)...")
    gen = GaussianCopulaGenerator()
    gen.fit(real)

    corruption_strategies = {
        "random_injection": corrupt_random_injection,
        "row_duplication": corrupt_row_duplication,
        "feature_dropout": corrupt_feature_dropout,
        "covariate_shift": corrupt_covariate_shift,
    }

    all_results = []

    for seed_i in range(n_seeds):
        seed = 42 + seed_i
        rng = np.random.default_rng(seed)
        print(f"\n{'=' * 60}")
        print(f"Seed {seed} ({seed_i + 1}/{n_seeds})")
        print(f"{'=' * 60}")

        syn_clean = gen.generate(n_rows, seed=seed)
        syn_clean = syn_clean.drop(columns=["syn_id"], errors="ignore")
        # Align columns
        common_cols = real.columns.intersection(syn_clean.columns).tolist()
        syn_clean = syn_clean[common_cols]

        for strategy_name, corrupt_fn in corruption_strategies.items():
            for level in corruption_levels:
                print(
                    f"  [{strategy_name}] corruption={level:.1f}, seed={seed}...",
                    end="",
                    flush=True,
                )

                # Apply corruption
                if strategy_name == "row_duplication":
                    corrupted, true_labels = corrupt_fn(syn_clean, level, rng)
                elif strategy_name in (
                    "random_injection",
                    "feature_dropout",
                    "covariate_shift",
                ):
                    corrupted, true_labels = corrupt_fn(syn_clean, real, level, rng)
                else:
                    corrupted, true_labels = corrupt_fn(syn_clean, real, level, rng)

                # Detect with each method
                try:
                    hif_preds = detect_hif(real, corrupted, seed=seed)
                except Exception as e:
                    print(f" HIF failed: {e}")
                    hif_preds = np.zeros(len(corrupted), dtype=int)

                try:
                    if_preds = detect_isolation_forest(
                        real, corrupted, contamination=level
                    )
                except Exception as e:
                    print(f" IF failed: {e}")
                    if_preds = np.zeros(len(corrupted), dtype=int)

                try:
                    lof_preds = detect_lof(real, corrupted, contamination=level)
                except Exception as e:
                    print(f" LOF failed: {e}")
                    lof_preds = np.zeros(len(corrupted), dtype=int)

                # Compute metrics
                for method_name, preds in [
                    ("HIF", hif_preds),
                    ("IsolationForest", if_preds),
                    ("LOF", lof_preds),
                ]:
                    if true_labels.sum() == 0:
                        f1 = prec = rec = 0.0
                    else:
                        f1 = f1_score(true_labels, preds, zero_division=0.0)
                        prec = precision_score(true_labels, preds, zero_division=0.0)
                        rec = recall_score(true_labels, preds, zero_division=0.0)

                    all_results.append(
                        {
                            "seed": seed,
                            "error_type": strategy_name,
                            "corruption_level": level,
                            "method": method_name,
                            "f1": f1,
                            "precision": prec,
                            "recall": rec,
                            "n_flagged": int(preds.sum()),
                            "n_true_errors": int(true_labels.sum()),
                        }
                    )

                print(" Done.")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "heldout_errors_raw.csv", index=False)

    # Generate summary table
    summary = (
        df.groupby(["error_type", "corruption_level", "method"])
        .agg(
            f1_mean=("f1", "mean"),
            f1_sem=("f1", sem),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
        )
        .round(3)
        .reset_index()
    )
    summary.to_csv(output_dir / "heldout_errors_summary.csv", index=False)

    # Print markdown table for rebuttal
    print("\n\n" + "=" * 80)
    print("REBUTTAL TABLE: Held-Out Error Detection (F1)")
    print("=" * 80)

    for level in corruption_levels:
        print(f"\n### Corruption Level = {level}")
        print("| Error Type | HIF F1 | IF F1 | LOF F1 |")
        print("|---|---|---|---|")
        sub = summary[summary["corruption_level"] == level]
        for etype in corruption_strategies:
            row_data = sub[sub["error_type"] == etype]
            hif_f1 = row_data[row_data["method"] == "HIF"]["f1_mean"].values
            if_f1 = row_data[row_data["method"] == "IsolationForest"]["f1_mean"].values
            lof_f1 = row_data[row_data["method"] == "LOF"]["f1_mean"].values
            hif_s = f"{hif_f1[0]:.3f}" if len(hif_f1) else "—"
            if_s = f"{if_f1[0]:.3f}" if len(if_f1) else "—"
            lof_s = f"{lof_f1[0]:.3f}" if len(lof_f1) else "—"
            print(f"| {etype} | {hif_s} | {if_s} | {lof_s} |")

    # Write markdown summary
    with open(output_dir / "heldout_errors_summary.md", "w") as f:
        f.write("# Held-Out Error Type Detection\n\n")
        f.write("Tests HIF on error types it was NOT designed for.\n\n")
        for level in corruption_levels:
            f.write(f"\n## Corruption Level = {level}\n\n")
            f.write("| Error Type | HIF F1 | IF F1 | LOF F1 |\n")
            f.write("|---|---|---|---|\n")
            sub = summary[summary["corruption_level"] == level]
            for etype in corruption_strategies:
                row_data = sub[sub["error_type"] == etype]
                hif_f1 = row_data[row_data["method"] == "HIF"]["f1_mean"].values
                if_f1 = row_data[row_data["method"] == "IsolationForest"][
                    "f1_mean"
                ].values
                lof_f1 = row_data[row_data["method"] == "LOF"]["f1_mean"].values
                hif_s = f"{hif_f1[0]:.3f}" if len(hif_f1) else "—"
                if_s = f"{if_f1[0]:.3f}" if len(if_f1) else "—"
                lof_s = f"{lof_f1[0]:.3f}" if len(lof_f1) else "—"
                f.write(f"| {etype} | {hif_s} | {if_s} | {lof_s} |\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Held-out error type experiment")
    parser.add_argument("--dataset", default="census_acs")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--corruption-levels", default="0.1,0.2,0.4")
    parser.add_argument("--output-dir", default="results/rebuttal")
    args = parser.parse_args()

    levels = [float(x) for x in args.corruption_levels.split(",")]
    run_experiment(
        dataset_id=args.dataset,
        n_rows=args.rows,
        n_seeds=args.seeds,
        corruption_levels=levels,
        output_dir=Path(args.output_dir),
    )
