"""
Experiment: Conditional Label-Noise Baseline for Integrity Filtering (M3).

Q: Does HIF filtering add value over generic conditional label curation in the
hub-target regime where filtering helps?

Baseline: train a Random Forest on the REAL data to predict the downstream
target from the remaining features (a conditional label model), score each
synthetic row by the probability assigned to its observed label, and prune
rows below a confidence floor calibrated as the 5th percentile of the
out-of-bag true-class probability (mirroring HIF's delta_h calibration).

Run:
    python scripts/16_label_noise_baseline.py --seeds 10
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
    utility_metrics,
)

CONFIGS = [
    {
        "dataset": "census_acs",
        "generator": "ctgan",
        "target": "household_income",
        "n": 2000,
    },
    {
        "dataset": "census_acs",
        "generator": "vine",
        "target": "household_income",
        "n": 2000,
    },
    {
        "dataset": "online_purchases",
        "generator": "ctgan",
        "target": "item_total",
        "n": 664,
    },
    {
        "dataset": "online_purchases",
        "generator": "vine",
        "target": "item_total",
        "n": 664,
    },
]


def _binarize_target(u_real: pd.DataFrame, u_syn: pd.DataFrame, target: str):
    """Replicate utility_metrics' target encoding (median split / category map)."""
    if pd.api.types.is_numeric_dtype(u_real[target]) and u_real[target].nunique() > 2:
        m = u_real[target].median()
        u_real[target] = (u_real[target] > m).astype(int)
        u_syn[target] = (u_syn[target] > m).astype(int)
    elif not pd.api.types.is_numeric_dtype(u_real[target]):
        cats = u_real[target].astype("category").cat.categories.tolist()
        u_real[target] = u_real[target].map({c: i for i, c in enumerate(cats)})
        u_syn[target] = u_syn[target].map({c: i for i, c in enumerate(cats)}).fillna(-1)
    return u_real, u_syn


def label_model_filter(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    target: str,
    seed: int,
) -> dict:
    """Prune synthetic rows whose observed label is improbable under a
    real-data-trained conditional label model (RF + OOB 5th-pct floor)."""
    real_util, syn_util, feature_cols = _utility_feature_frame(real, syn, target)
    if len(feature_cols) < 1:
        return {"mask": np.ones(len(syn), dtype=bool), "floor": np.nan}
    real_util, syn_util = _binarize_target(real_util.copy(), syn_util.copy(), target)
    if real_util[target].nunique() < 2:
        return {"mask": np.ones(len(syn), dtype=bool), "floor": np.nan}

    X_real = real_util[feature_cols]
    y_real = real_util[target].astype(int)
    X_syn = syn_util[feature_cols]
    y_syn = syn_util[target].astype(int)

    clf = RandomForestClassifier(
        n_estimators=100, random_state=seed, oob_score=True, n_jobs=-1
    )
    clf.fit(X_real, y_real)

    # OOB probability of the TRUE class on real rows -> confidence floor.
    oob_proba = clf.oob_decision_function_  # shape (n_real, n_classes)
    classes = list(clf.classes_)
    true_idx = np.array([classes.index(v) for v in y_real])
    p_true_oob = oob_proba[np.arange(len(y_real)), true_idx]
    floor = max(float(np.quantile(p_true_oob, 0.05)), 0.01)

    # Probability each synthetic row's OBSERVED label is correct.
    syn_proba = clf.predict_proba(X_syn)
    obs_idx = np.clip(y_syn.values, 0, len(classes) - 1)
    p_obs = syn_proba[np.arange(len(y_syn)), obs_idx]

    mask = p_obs >= floor
    return {"mask": mask, "floor": floor}


def paired_stats(deltas: np.ndarray) -> dict:
    d = deltas[~np.isnan(deltas)]
    if len(d) < 3:
        return {"t_p": np.nan, "wilcoxon_p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    t_res = stats.ttest_1samp(d, 0.0)
    try:
        w_res = stats.wilcoxon(d)
        w_p = float(w_res.pvalue)
    except ValueError:
        w_p = np.nan
    mean = float(np.mean(d))
    half = float(stats.t.ppf(0.975, len(d) - 1) * stats.sem(d))
    return {
        "t_p": round(float(t_res.pvalue), 4),
        "wilcoxon_p": round(w_p, 4),
        "ci_lo": round(mean - half, 4),
        "ci_hi": round(mean + half, 4),
    }


def run_config(cfg: dict, n_seeds: int) -> list[dict]:
    rows: list[dict] = []
    for seed_i in range(n_seeds):
        seed = 42 + seed_i
        print(f"  Seed {seed} ({seed_i + 1}/{n_seeds})")
        real = load_real(cfg["dataset"], n=cfg["n"], seed=seed).reset_index(drop=True)
        syn = generate(real, cfg["n"], seed, cfg["generator"])
        syn = syn[real.columns.intersection(syn.columns).tolist()]

        util_full = utility_metrics(real, syn, cfg["target"], seed)

        hif = audit_hif(real, syn, seed=seed)
        hif_mask = hif["row_penalties"] < 0.5
        syn_hif = syn[hif_mask]
        util_hif = (
            utility_metrics(real, syn_hif, cfg["target"], seed)
            if len(syn_hif) > 10
            else {"f1": np.nan, "accuracy": np.nan, "trr": np.nan}
        )

        lm = label_model_filter(real, syn, cfg["target"], seed)
        syn_lm = syn[lm["mask"]]
        util_lm = (
            utility_metrics(real, syn_lm, cfg["target"], seed)
            if len(syn_lm) > 10
            else {"f1": np.nan, "accuracy": np.nan, "trr": np.nan}
        )

        rows.append(
            {
                "dataset": cfg["dataset"],
                "generator": cfg["generator"],
                "target": cfg["target"],
                "seed": seed,
                "f1_full": util_full["f1"],
                "f1_hif": util_hif["f1"],
                "retention_hif": len(syn_hif) / len(syn) * 100,
                "f1_label_model": util_lm["f1"],
                "retention_label_model": len(syn_lm) / len(syn) * 100,
                "label_floor": lm["floor"],
                "hif_violation_rate": float((~hif_mask).mean() * 100),
            }
        )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for key, grp in df.groupby(["dataset", "generator"]):
        base = {
            "dataset": key[0],
            "generator": key[1],
            "n_seeds": len(grp),
        }
        for arm, f1_col, ret_col in [
            ("HIF", "f1_hif", "retention_hif"),
            ("LabelModel", "f1_label_model", "retention_label_model"),
        ]:
            deltas = (grp[f1_col] - grp["f1_full"]).values
            st = paired_stats(deltas)
            base.update(
                {
                    f"{arm}_dF1_mean": round(float(np.nanmean(deltas)), 4),
                    f"{arm}_dF1_sd": round(float(np.nanstd(deltas, ddof=1)), 4),
                    f"{arm}_t_p": st["t_p"],
                    f"{arm}_wilcoxon_p": st["wilcoxon_p"],
                    f"{arm}_ci": f"[{st['ci_lo']}, {st['ci_hi']}]",
                    f"{arm}_retention": round(float(grp[ret_col].mean()), 1),
                }
            )
        base["full_f1"] = round(float(grp["f1_full"].mean()), 4)
        base["label_floor_mean"] = round(float(grp["label_floor"].mean()), 4)
        out_rows.append(base)
    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for cfg in CONFIGS:
        print(
            f"\n{'=' * 70}\n{cfg['dataset']} x {cfg['generator']} "
            f"(target={cfg['target']})\n{'=' * 70}"
        )
        all_rows.extend(run_config(cfg, args.seeds))

    raw = pd.DataFrame(all_rows)
    raw.to_csv(output_dir / "label_noise_baseline_raw.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(output_dir / "label_noise_baseline_summary.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
