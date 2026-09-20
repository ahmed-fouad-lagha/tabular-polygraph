"""Leakage-free comparison of HIF filtering and conditional label curation.

For each seed, the real data are split before the generator, HIF, or label
model is fitted.  The label model and its OOB confidence floor use only the
real training partition; TSTR is evaluated only on the untouched test
partition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ruff: noqa: E402
from _exp_utils import (
    _utility_feature_frame,
    audit_hif,
    generate,
    load_real,
    split_real_for_utility,
    utility_metrics_on_holdout,
)

CONFIGS = (
    ("census_acs", "household_income", "ctgan", 2000),
    ("census_acs", "household_income", "vine", 2000),
    ("online_purchases", "item_total", "ctgan", 664),
    ("online_purchases", "item_total", "vine", 664),
)


def _encode_target(
    train_real: pd.DataFrame, frame: pd.DataFrame, target: str
) -> pd.Series:
    """Use only training-real labels to define the downstream target encoding."""
    if (
        pd.api.types.is_numeric_dtype(train_real[target])
        and train_real[target].nunique() > 2
    ):
        return (frame[target] > train_real[target].median()).astype(int)
    if not pd.api.types.is_numeric_dtype(train_real[target]):
        categories = train_real[target].astype("category").cat.categories
        mapping = {value: index for index, value in enumerate(categories)}
        return frame[target].map(mapping).fillna(-1).astype(int)
    return frame[target].astype(int)


def label_model_mask(
    train_real: pd.DataFrame, synthetic: pd.DataFrame, target: str, seed: int
) -> tuple[np.ndarray, float]:
    """Return a label-model curation mask and its training-only OOB floor."""
    real_util, syn_util, features = _utility_feature_frame(
        train_real, synthetic, target
    )
    if not features:
        return np.ones(len(synthetic), dtype=bool), np.nan
    y_real = _encode_target(train_real, real_util, target)
    y_syn = _encode_target(train_real, syn_util, target)
    if y_real.nunique() < 2:
        return np.ones(len(synthetic), dtype=bool), np.nan

    model = RandomForestClassifier(
        n_estimators=100, random_state=seed, oob_score=True, n_jobs=-1
    )
    model.fit(real_util[features], y_real)
    classes = list(model.classes_)
    class_index = {label: index for index, label in enumerate(classes)}
    true_indices = np.array([class_index[label] for label in y_real])
    p_true = model.oob_decision_function_[np.arange(len(y_real)), true_indices]
    floor = max(float(np.quantile(p_true, 0.05)), 0.01)

    probabilities = model.predict_proba(syn_util[features])
    observed_indices = y_syn.map(class_index).fillna(-1).astype(int).to_numpy()
    p_observed = np.zeros(len(synthetic), dtype=float)
    known = observed_indices >= 0
    p_observed[known] = probabilities[
        np.arange(len(synthetic))[known], observed_indices[known]
    ]
    return p_observed >= floor, floor


def paired_summary(group: pd.DataFrame, metric: str) -> dict[str, float | int]:
    valid = group.dropna(subset=["f1_full", metric])
    differences = valid[metric] - valid["f1_full"]
    n = len(differences)
    if n < 2:
        return {
            "n_valid_seeds": n,
            "delta_f1_mean": np.nan,
            "paired_ttest_p": np.nan,
            "delta_f1_ci_low": np.nan,
            "delta_f1_ci_high": np.nan,
        }
    interval = stats.t.interval(
        0.95, n - 1, loc=differences.mean(), scale=stats.sem(differences)
    )
    return {
        "n_valid_seeds": n,
        "delta_f1_mean": float(differences.mean()),
        "paired_ttest_p": float(
            stats.ttest_rel(valid[metric], valid["f1_full"]).pvalue
        ),
        "delta_f1_ci_low": float(interval[0]),
        "delta_f1_ci_high": float(interval[1]),
    }


def run(
    n_seeds: int, output_dir: Path, test_size: float = 0.30, seed_start: int = 42
) -> tuple[Path, Path]:
    """Run and checkpoint the four pre-specified configurations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "leakage_free_label_noise_raw.csv"
    summary_path = output_dir / "leakage_free_label_noise_summary.csv"
    rows = pd.read_csv(raw_path).to_dict("records") if raw_path.exists() else []
    completed = {(r["dataset"], r["generator"], int(r["seed"])) for r in rows}

    for dataset, target, generator, n_rows in CONFIGS:
        for offset in range(n_seeds):
            seed = seed_start + offset
            key = (dataset, generator, seed)
            if key in completed:
                print(
                    f"{dataset}/{generator} seed={seed} already checkpointed",
                    flush=True,
                )
                continue
            real = load_real(dataset, n=n_rows, seed=seed).reset_index(drop=True)
            train_real, test_real = split_real_for_utility(
                real, target, seed, test_size
            )
            synthetic = generate(train_real, len(real), seed, generator)
            synthetic = synthetic[
                train_real.columns.intersection(synthetic.columns)
            ].copy()

            full = utility_metrics_on_holdout(
                train_real, synthetic, test_real, target, seed
            )
            hif = audit_hif(train_real, synthetic, seed=seed)
            hif_mask = hif["row_penalties"] < 0.5
            label_mask, floor = label_model_mask(train_real, synthetic, target, seed)
            hif_utility = (
                utility_metrics_on_holdout(
                    train_real, synthetic.loc[hif_mask], test_real, target, seed
                )
                if hif_mask.sum() >= 10
                else {"f1": np.nan}
            )
            label_utility = (
                utility_metrics_on_holdout(
                    train_real, synthetic.loc[label_mask], test_real, target, seed
                )
                if label_mask.sum() >= 10
                else {"f1": np.nan}
            )
            row = {
                "dataset": dataset,
                "generator": generator,
                "target": target,
                "seed": seed,
                "n_real_total": len(real),
                "n_real_train": len(train_real),
                "n_real_test": len(test_real),
                "n_synthetic": len(synthetic),
                "test_size": test_size,
                "label_floor": floor,
                "f1_full": full["f1"],
                "f1_hif": hif_utility["f1"],
                "retention_hif_pct": float(hif_mask.mean() * 100),
                "f1_label_model": label_utility["f1"],
                "retention_label_model_pct": float(label_mask.mean() * 100),
            }
            rows.append(row)
            completed.add(key)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(
                f"{dataset}/{generator} seed={seed}: HIF {row['f1_full']:.3f}->{row['f1_hif']:.3f} "
                f"({row['retention_hif_pct']:.1f}%), label {row['f1_full']:.3f}->{row['f1_label_model']:.3f} "
                f"({row['retention_label_model_pct']:.1f}%)",
                flush=True,
            )

    raw = pd.DataFrame(rows)
    summaries = []
    for (dataset, generator), group in raw.groupby(
        ["dataset", "generator"], sort=False
    ):
        for arm, metric, retention in (
            ("HIF", "f1_hif", "retention_hif_pct"),
            ("Label model", "f1_label_model", "retention_label_model_pct"),
        ):
            summaries.append(
                {
                    "dataset": dataset,
                    "generator": generator,
                    "arm": arm,
                    "f1_full_mean": group["f1_full"].mean(),
                    "f1_filtered_mean": group[metric].mean(),
                    "retention_mean_pct": group[retention].mean(),
                    "label_floor_mean": group["label_floor"].mean(),
                    **paired_summary(group, metric),
                }
            )
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    return raw_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/leakage_free"
    )
    args = parser.parse_args()
    raw, summary = run(args.seeds, args.output_dir, args.test_size, args.seed_start)
    print(f"Raw results: {raw}\nSummary: {summary}")
