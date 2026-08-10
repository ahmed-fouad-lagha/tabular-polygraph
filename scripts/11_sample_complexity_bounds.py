"""
Experiment: Sample-Complexity Bounds at HIF's Operating Configuration (Q4).

Q: "Is there a computable relationship between the sample-complexity bounds
    cited in Sec. 3 (Valiant; Pensia; Neykov) and HIF's own reliability given
    the hub-conditioned sample sizes actually used (2,000 rows, up to 10
    features)?"

Instantiation: the identity/uniformity-testing result m >= sqrt(S) / eps^2
(Paninski 2008; Valiant & Valiant 2014; Pensia et al. 2024) inverted at the
sample sizes HIF actually conditions on. For hub h and category c, the
sentinel estimates structure from the n_{h,c} real rows whose hub value is c;
over that cell's observed joint support S_{h,c} the smallest L1 deviation
detectable is eps_{h,c} = (S_{h,c} / n_{h,c}^2)^(1/4). The aggregate analogue
tests the full joint over n rows and S_joint distinct configurations.

Run:
    python scripts/11_sample_complexity_bounds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ruff: noqa: E402
from _exp_utils import load_real
from tabular_polygraph.fidelity.hif.binning import apply_binning, fit_binning
from tabular_polygraph.fidelity.hif.sentinel import LogicalSentinelEnsemble

DATASETS = [
    "census_acs",
    "adult",
    "credit",
    "online_purchases",
    "supermarket_sales",
]
N_ROWS = 2000
TOP_N_HUBS = 5
MAX_DEPTH = 12
RANDOM_STATE = 42


def detectability(support: int, n: int) -> float:
    """Inverted identity-testing bound: smallest L1 deviation detectable.

    m >= sqrt(S) / eps^2  =>  eps >= (S / n^2)^(1/4).
    """
    if n < 1:
        return np.nan
    return float((support / (n * n)) ** 0.25)


def compute_per_dataset(ds_id: str) -> dict:
    real = load_real(ds_id, n=N_ROWS)
    columns = list(real.columns)
    bin_edges = fit_binning(real[columns], columns)
    real_f = apply_binning(real[columns], columns, bin_edges)

    sentinel = LogicalSentinelEnsemble(
        top_n_hubs=TOP_N_HUBS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    sentinel.fit(real_f)
    hubs = sentinel.hubs[:TOP_N_HUBS]

    cell_rows = []
    for hub in hubs:
        for c, n_c in real_f[hub].value_counts().items():
            cell = real_f[real_f[hub] == c]
            s_c = cell.drop(columns=[hub]).drop_duplicates().shape[0]
            cell_rows.append(
                {
                    "dataset": ds_id,
                    "hub": hub,
                    "category": c,
                    "n_cell": int(n_c),
                    "s_cell": int(s_c),
                    "saturation": round(s_c / n_c, 3),
                    "eps_cell": round(detectability(s_c, n_c), 4),
                }
            )

    n_total = len(real)
    s_joint = int(real.drop_duplicates().shape[0])
    eps_joint = detectability(s_joint, n_total)

    cell_df = pd.DataFrame(cell_rows)
    return {
        "dataset": ds_id,
        "n_rows": n_total,
        "n_hubs": len(hubs),
        "hubs": hubs,
        "median_n_cell": float(cell_df["n_cell"].median()),
        "min_n_cell": float(cell_df["n_cell"].min()),
        "median_s_cell": float(cell_df["s_cell"].median()),
        "median_eps_cell": float(cell_df["eps_cell"].median()),
        "worst_eps_cell": float(cell_df["eps_cell"].max()),
        "s_joint": s_joint,
        "eps_joint": eps_joint,
    }


def main(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        res = compute_per_dataset(ds)
        print(f"  hubs: {', '.join(res['hubs'])}")
        print(
            f"  per-cell n (median {res['median_n_cell']:.0f}, min {res['min_n_cell']:.0f}) "
            f"| support median {res['median_s_cell']:.0f} "
            f"| eps_cell median {res['median_eps_cell']:.3f}, worst {res['worst_eps_cell']:.3f} "
            f"| eps_joint {res['eps_joint']:.3f}"
        )
        res["hubs"] = ", ".join(res["hubs"])
        rows.append(res)

    df = pd.DataFrame(rows).round(4)
    df.to_csv(output_dir / "sample_complexity_bounds.csv", index=False)

    print("\n" + "=" * 100)
    print("Sample-Complexity Bounds at HIF's Operating Configuration")
    print("=" * 100)
    print(
        f"{'dataset':<18} | {'n':>5} | {'hubs':<34} | {'med n_cell':>10} | "
        f"{'worst eps_cell':>14} | {'eps_joint':>9}"
    )
    for _, r in df.iterrows():
        print(
            f"{r['dataset']:<18} | {r['n_rows']:>5.0f} | {r['hubs']:<34} | "
            f"{r['median_n_cell']:>10.0f} | {r['worst_eps_cell']:>14.3f} | "
            f"{r['eps_joint']:>9.3f}"
        )
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample-complexity bounds at HIF's operating configuration (Q4)"
    )
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    main(Path(args.output_dir))
