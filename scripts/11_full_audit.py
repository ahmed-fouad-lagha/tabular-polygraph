"""
Full audit: HIF vs utility across ALL dataset × target × generator combinations.

Goal: find the combinations where low HIF predicts low downstream utility.
If no such combinations exist, the paper story is dead.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tabular_polygraph.dataset.downloader import load_cached  # noqa: E402
from tabular_polygraph.fidelity.downstream import tstr_score  # noqa: E402
from tabular_polygraph.fidelity.logical import hif_score  # noqa: E402
from tabular_polygraph.fidelity.marginal import ks_distribution_scores  # noqa: E402
from tabular_polygraph.generators import (  # noqa: E402
    CTGANGenerator,
    GaussianCopulaGenerator,
    TVAEGenerator,
)


def _sanitize_categoricals(real, syn):
    syn = syn.copy()
    for col in real.select_dtypes(include=["object", "category"]).columns:
        valid = real[col].dropna().unique()
        mask = ~syn[col].isin(valid)
        if mask.any():
            syn.loc[mask, col] = np.random.choice(valid, size=mask.sum())
    return syn


def run_generator(real, gen_type, rows, seed, epochs=50):
    np.random.seed(seed)
    if gen_type == "gaussian":
        gen = GaussianCopulaGenerator()
    elif gen_type == "ctgan":
        gen = CTGANGenerator(epochs=epochs, batch_size=min(100, len(real)))
    elif gen_type == "tvae":
        gen = TVAEGenerator(epochs=epochs)
    else:
        raise ValueError(f"Unknown: {gen_type}")
    gen.fit(real)
    syn = gen.generate(rows, seed=seed).drop(columns=["syn_id"], errors="ignore")
    syn = _sanitize_categoricals(real, syn)
    return syn


DATASET_TARGETS = {
    "supermarket_sales": {
        "regression": ["unit_price", "gross_income"],
        "classification": [
            "branch",
            "product_line",
            "customer_type",
            "payment",
            "gender",
        ],
    },
    "online_purchases": {
        "regression": ["list_price", "purchase_price"],
        "classification": ["category"],
    },
    "credit": {
        "regression": ["limit_bal", "bill_amt1", "pay_amt1"],
        "classification": ["default_payment", "education", "marriage"],
    },
    "adult": {
        "regression": ["age", "hours_per_week"],
        "classification": ["income", "workclass", "occupation"],
    },
}

RESULTS_PATH = PROJECT_ROOT / "results" / "full_audit.csv"


def main():
    generators = ["gaussian", "ctgan", "tvae"]
    seeds = [42, 43, 44]
    rows = 500
    epochs = 50  # faster for CTGAN/TVAE

    all_results = []

    for ds_id, targets in DATASET_TARGETS.items():
        real_full = load_cached(ds_id)
        if real_full is None:
            print(f"SKIP {ds_id}: not cached")
            continue

        drop_cols = [c for c in real_full.columns if real_full[c].isnull().mean() > 0.3]
        real_full = real_full.drop(columns=drop_cols, errors="ignore")

        all_targets = targets.get("regression", []) + targets.get("classification", [])

        print(f"\n{'=' * 80}")
        print(f"  {ds_id}: {len(real_full)} rows × {len(real_full.columns)} cols")
        print(f"  Targets: {all_targets}")
        print(f"{'=' * 80}")

        for gen_type in generators:
            for seed in seeds:
                real = real_full.sample(
                    min(rows, len(real_full)), random_state=seed
                ).reset_index(drop=True)

                try:
                    t0 = time.time()
                    syn = run_generator(real, gen_type, len(real), seed, epochs)
                    dt = time.time() - t0

                    # HIF
                    try:
                        hif_res = hif_score(real, syn, verbose=False, hif_epochs=5)
                        hif_val = hif_res["hif_score"]
                        viol_rate = hif_res["violation_rate"]
                    except Exception as e:
                        print(f"    HIF error: {e}")
                        hif_val = np.nan
                        viol_rate = np.nan

                    # KS
                    try:
                        ks = ks_distribution_scores(real, syn)
                        ks_val = float(np.mean(list(ks.values()))) if ks else np.nan
                    except Exception:
                        ks_val = np.nan

                    # TSTR for each target
                    for target in all_targets:
                        if target not in real.columns:
                            continue
                        task = (
                            "classification"
                            if target in targets.get("classification", [])
                            else "regression"
                        )
                        try:
                            tstr_res = tstr_score(
                                real, syn, target_col=target, task=task, seed=seed
                            )
                            tstr_val = tstr_res.get("tstr_score", np.nan)
                            trr_val = tstr_res.get("trr_score", np.nan)
                            ratio_val = tstr_res.get("ratio", np.nan)
                        except Exception as e:
                            print(f"    TSTR error {target}: {e}")
                            tstr_val = trr_val = ratio_val = np.nan

                        row = {
                            "dataset": ds_id,
                            "generator": gen_type,
                            "seed": seed,
                            "target": target,
                            "task": task,
                            "hif_score": hif_val,
                            "violation_rate": viol_rate,
                            "ks_score": ks_val,
                            "tstr_score": tstr_val,
                            "trr_score": trr_val,
                            "tstr_ratio": ratio_val,
                            "time_s": round(dt, 1),
                        }
                        all_results.append(row)

                        flag = ""
                        if not np.isnan(trr_val) and not np.isnan(ratio_val):
                            if trr_val > 0.3 and ratio_val < 0.7:
                                flag = " *** GAP ***"
                            elif trr_val < 0.1:
                                flag = " (no signal)"

                        print(
                            f"  {gen_type:8s} seed={seed} target={target:20s} "
                            f"HIF={hif_val:.3f}  KS={ks_val:.1f}  "
                            f"TSTR={tstr_val:.3f}  TRR={trr_val:.3f}  ratio={ratio_val:.3f}"
                            f"  ({dt:.0f}s){flag}"
                        )

                except Exception as e:
                    print(f"  {gen_type} seed={seed} ERROR: {e}")

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_PATH, index=False)

    # ANALYSIS
    print(f"\n\n{'=' * 80}")
    print("  ANALYSIS: Which combinations show the paper story?")
    print(f"{'=' * 80}")

    print(
        "\n--- Filtered: TRR > 0.2 (has real signal) and ratio < 0.8 (utility hurt) ---"
    )
    good = df[(df["trr_score"] > 0.2) & (df["tstr_ratio"] < 0.8)]
    if good.empty:
        print("  NONE FOUND.")
    else:
        for (ds, tgt), grp in good.groupby(["dataset", "target"]):
            for _, r in grp.iterrows():
                print(
                    f"  {ds}/{tgt}: {r['generator']} HIF={r['hif_score']:.3f} TRR={r['trr_score']:.3f} ratio={r['tstr_ratio']:.3f}"
                )

    print("\n--- Correlation: HIF vs TSTR ratio (per dataset) ---")
    for ds_id in df["dataset"].unique():
        sub = df[
            (df["dataset"] == ds_id)
            & df["tstr_ratio"].notna()
            & df["hif_score"].notna()
            & (df["trr_score"] > 0.15)
        ]
        if len(sub) > 3:
            rho = sub["hif_score"].corr(sub["tstr_ratio"])
            print(f"  {ds_id}: ρ(HIF, ratio) = {rho:.3f}  (n={len(sub)})")
        else:
            print(f"  {ds_id}: insufficient data")

    print(f"\n  Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
