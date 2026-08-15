"""
Experiment: Held-Out Error Types & Outlier Baseline Comparison (current API).

Tests whether HIF detects error types it was NOT designed for, and compares
against standard outlier detectors (Isolation Forest, LOF).

Run:
    python scripts/05_heldout_error_baselines.py --dataset census_acs --rows 2000 \
      --seeds 10 --corruption-levels 0.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ruff: noqa: E402
from _exp_utils import audit_hif, generate, load_real


def corrupt_random_injection(
    syn: pd.DataFrame, real: pd.DataFrame, level: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Replace random cells with random values from the column's domain."""
    out = syn.copy()
    n_rows = len(out)
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
    """Duplicate random rows with small Gaussian noise on numerics."""
    n_corrupt = max(1, int(len(syn) * level))
    source_idx = rng.choice(len(syn), size=n_corrupt, replace=True)
    duplicated = syn.iloc[source_idx].copy().reset_index(drop=True)
    num_cols = [
        c for c in duplicated.columns if pd.api.types.is_numeric_dtype(duplicated[c])
    ]
    for col in num_cols:
        std = float(syn[col].std())
        if std > 0:
            noise = rng.normal(0, std * 0.01, size=n_corrupt)
            if pd.api.types.is_integer_dtype(duplicated[col]):
                duplicated[col] = (duplicated[col] + np.round(noise)).astype(
                    duplicated[col].dtype
                )
            else:
                duplicated[col] = duplicated[col] + noise
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
    """Set random features to column mode/median, simulating missing data."""
    out = syn.copy()
    n_rows = len(out)
    n_corrupt = max(1, int(n_rows * level))
    row_idx = rng.choice(n_rows, size=n_corrupt, replace=False)
    labels = np.zeros(n_rows, dtype=bool)
    labels[row_idx] = True
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
    """Replace a fraction of rows with samples from a biased real subset."""
    out = syn.copy()
    n_corrupt = max(1, int(len(out) * level))
    row_idx = rng.choice(len(out), size=n_corrupt, replace=False)
    labels = np.zeros(len(out), dtype=bool)
    labels[row_idx] = True
    num_cols_list = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]
    if num_cols_list:
        bias_col = num_cols_list[0]
        threshold = real[bias_col].quantile(0.75)
        biased = real[real[bias_col] >= threshold]
        if len(biased) < n_corrupt:
            biased = real
    else:
        biased = real
    replacements = biased.sample(
        n=n_corrupt, replace=True, random_state=int(rng.integers(1000000))
    )
    for i, idx in enumerate(row_idx):
        out.iloc[idx] = replacements.iloc[i]
    return out, labels


def corrupt_semantic_hallucination(
    syn: pd.DataFrame, real: pd.DataFrame, level: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Inject dependency violations that preserve marginals.

    Replaces the second-most-correlated numeric feature of a fraction of rows
    with a value drawn from real rows in the *opposite* half of the
    most-correlated feature.
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


def _encode_for_outlier_detection(
    real: pd.DataFrame, syn: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray]:
    X_real, X_syn = _encode_for_outlier_detection(real, syn)
    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    clf.fit(X_real)
    scores = -clf.score_samples(X_syn)
    preds = (clf.predict(X_syn) == -1).astype(int)
    return preds, scores


def detect_lof(
    real: pd.DataFrame, syn: pd.DataFrame, contamination: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    X_real, X_syn = _encode_for_outlier_detection(real, syn)
    clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    clf.fit(X_real)
    scores = -clf.score_samples(X_syn)
    preds = (clf.predict(X_syn) == -1).astype(int)
    return preds, scores


def detect_hif(
    real: pd.DataFrame, syn: pd.DataFrame, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    result = audit_hif(real, syn, seed=seed)
    scores = result["row_penalties"]
    preds = (scores > 0.5).astype(int)
    return preds, scores


def fit_gmm(real: pd.DataFrame, seed: int = 42) -> GaussianMixture:
    """Fit a Gaussian Mixture density model on the encoded real data.

    Component count is selected by BIC over ``k in [1, 6]``; ``reg_covar``
    guards against singular covariance on sparse one-hot encodings.
    """
    X_real, _ = _encode_for_outlier_detection(real, real)
    best_bic, best = np.inf, None
    for k in range(1, 7):
        m = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=seed,
            reg_covar=1e-4,
            max_iter=300,
            n_init=1,
        )
        m.fit(X_real)
        bic = m.bic(X_real)
        if bic < best_bic:
            best_bic, best = bic, m
    assert best is not None
    return best


def detect_gmm(
    gmm: GaussianMixture,
    real: pd.DataFrame,
    syn: pd.DataFrame,
    contamination: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Learned-density baseline: score records by negative log-likelihood.

    A synthetic record inconsistent with the joint distribution learned from
    the real data receives low density, hence a high score. The F1 threshold is
    the ``(1 - contamination)`` quantile of the *real* data's scores, mirroring
    the oracle ``contamination`` calibration granted to IF and LOF.
    """
    X_real, X_syn = _encode_for_outlier_detection(real, syn)
    scores = -gmm.score_samples(X_syn)
    thr = np.quantile(-gmm.score_samples(X_real), 1.0 - contamination)
    preds = (scores >= thr).astype(int)
    return preds, scores


def run_experiment(
    dataset_id: str,
    n_rows: int,
    n_seeds: int,
    corruption_levels: list[float],
    output_dir: Path,
):
    print(f"Loading dataset: {dataset_id}")

    print("Generating clean synthetic baseline (Gaussian Copula)...")
    corruption_strategies = {
        "random_injection": corrupt_random_injection,
        "semantic_hallucination": corrupt_semantic_hallucination,
        "row_duplication": corrupt_row_duplication,
        "feature_dropout": corrupt_feature_dropout,
        "covariate_shift": corrupt_covariate_shift,
    }

    all_results = []
    for seed_i in range(n_seeds):
        seed = 42 + seed_i
        rng = np.random.default_rng(seed)
        print(f"\n{'=' * 60}\nSeed {seed} ({seed_i + 1}/{n_seeds})\n{'=' * 60}")

        real = load_real(dataset_id, n=n_rows, seed=seed).reset_index(drop=True)
        syn_clean = generate(real, n_rows, seed, "gaussian_copula")
        syn_clean = syn_clean[real.columns.intersection(syn_clean.columns).tolist()]

        print("  Fitting GMM density baseline on real data...", flush=True)
        gmm = fit_gmm(real, seed=seed)

        for strategy_name, corrupt_fn in corruption_strategies.items():
            for level in corruption_levels:
                print(
                    f"  [{strategy_name}] corruption={level:.1f}, seed={seed}...",
                    end="",
                    flush=True,
                )
                if strategy_name == "row_duplication":
                    corrupted, true_labels = corrupt_fn(syn_clean, level, rng)
                else:
                    corrupted, true_labels = corrupt_fn(syn_clean, real, level, rng)

                try:
                    hif_preds, hif_scores = detect_hif(real, corrupted, seed=seed)
                except Exception as e:
                    print(f" HIF failed: {e}")
                    hif_preds = np.zeros(len(corrupted), dtype=int)
                    hif_scores = np.zeros(len(corrupted))

                try:
                    if_preds, if_scores = detect_isolation_forest(
                        real, corrupted, contamination=level
                    )
                except Exception as e:
                    print(f" IF failed: {e}")
                    if_preds = np.zeros(len(corrupted), dtype=int)
                    if_scores = np.zeros(len(corrupted))

                try:
                    lof_preds, lof_scores = detect_lof(
                        real, corrupted, contamination=level
                    )
                except Exception as e:
                    print(f" LOF failed: {e}")
                    lof_preds = np.zeros(len(corrupted), dtype=int)
                    lof_scores = np.zeros(len(corrupted))

                try:
                    gmm_preds, gmm_scores = detect_gmm(
                        gmm, real, corrupted, contamination=level
                    )
                except Exception as e:
                    print(f" GMM failed: {e}")
                    gmm_preds = np.zeros(len(corrupted), dtype=int)
                    gmm_scores = np.zeros(len(corrupted))

                for method_name, preds, scores in [
                    ("HIF", hif_preds, hif_scores),
                    ("IsolationForest", if_preds, if_scores),
                    ("LOF", lof_preds, lof_scores),
                    ("GMM", gmm_preds, gmm_scores),
                ]:
                    if true_labels.sum() == 0:
                        f1 = prec = rec = roc_auc = pr_auc = 0.0
                    else:
                        f1 = f1_score(true_labels, preds, zero_division=0.0)
                        prec = precision_score(true_labels, preds, zero_division=0.0)
                        rec = recall_score(true_labels, preds, zero_division=0.0)
                        roc_auc = (
                            float(roc_auc_score(true_labels, scores))
                            if len(np.unique(true_labels)) > 1
                            else np.nan
                        )
                        pr_auc = (
                            float(average_precision_score(true_labels, scores))
                            if len(np.unique(true_labels)) > 1
                            else np.nan
                        )
                    all_results.append(
                        {
                            "seed": seed,
                            "error_type": strategy_name,
                            "corruption_level": level,
                            "method": method_name,
                            "f1": f1,
                            "precision": prec,
                            "recall": rec,
                            "roc_auc": roc_auc,
                            "pr_auc": pr_auc,
                            "n_flagged": int(preds.sum()),
                            "n_true_errors": int(true_labels.sum()),
                        }
                    )
                print(" Done.")

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "heldout_errors_raw.csv", index=False)

    summary = (
        df.groupby(["error_type", "corruption_level", "method"])
        .agg(
            f1_mean=("f1", "mean"),
            f1_sem=("f1", sem),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
        )
        .round(3)
        .reset_index()
    )
    summary.to_csv(output_dir / "heldout_errors_summary.csv", index=False)

    print("\n\n" + "=" * 80)
    print("Held-Out Error Detection (F1 / ROC-AUC / PR-AUC)")
    print("=" * 80)
    for level in corruption_levels:
        print(f"\n### Corruption Level = {level}")
        print(
            "| Error Type | HIF F1 (ROC-AUC / PR-AUC) | IF F1 (ROC / PR) | "
            "LOF F1 (ROC / PR) | GMM F1 (ROC / PR) |"
        )
        print("|---|---|---|---|---|")
        sub = summary[summary["corruption_level"] == level]
        for etype in corruption_strategies:
            row_data = sub[sub["error_type"] == etype]

            def get_str(m_name: str, r_df: pd.DataFrame) -> str:
                d = r_df[r_df["method"] == m_name]
                if d.empty:
                    return "—"
                return (
                    f"{d['f1_mean'].values[0]:.3f} "
                    f"({d['roc_auc_mean'].values[0]:.3f} / {d['pr_auc_mean'].values[0]:.3f})"
                )

            print(
                f"| {etype} | {get_str('HIF', row_data)} | "
                f"{get_str('IsolationForest', row_data)} | {get_str('LOF', row_data)} | "
                f"{get_str('GMM', row_data)} |"
            )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Held-out error type experiment")
    parser.add_argument("--dataset", default="census_acs")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--corruption-levels", default="0.4")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    levels = [float(x) for x in args.corruption_levels.split(",")]
    run_experiment(
        dataset_id=args.dataset,
        n_rows=args.rows,
        n_seeds=args.seeds,
        corruption_levels=levels,
        output_dir=Path(args.output_dir),
    )
