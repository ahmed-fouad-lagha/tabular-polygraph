"""
Experiment: Real-Data Reference Floor for the Cross-Architecture Audit.

Computes the HIF auditor's own operating characteristics on genuine held-out
real data, providing the reference row for Table 1 and the empirical
false-rejection (violation) rate of the framework.

Protocol (held-out, deterministic):
    * adult / credit:          fit on 2,000 real rows (random_state=42), score
      2,000 additional held-out rows drawn from the cache remainder.
    * census_acs (2,462 rows): fit on 2,000, score the remaining 462 rows.
    * online_purchases (664) and supermarket_sales (1,000): fit on 80% of the
      real cohort (random_state=42), score the remaining 20%.

A "replayed training" row (auditor fit and scored on the same data) is also
written to expose the framework-level overfitting signature: the auditor scores
its own training data higher than fresh held-out real data.

Run:
    python scripts/14_real_data_reference.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))

from tabular_polygraph.dataset import load_cached  # noqa: E402
from tabular_polygraph.fidelity.hif.orchestrator import hif_score  # noqa: E402

DATASETS = ["adult", "credit", "census_acs", "online_purchases", "supermarket_sales"]
SEED = 42
OUTPUT = PROJECT_ROOT / "outputs" / "real_data_reference.csv"


def audit_rows(
    real: pd.DataFrame, synthetic: pd.DataFrame, split: str, n_train: int, n_test: int
) -> dict:
    cols = real.columns.intersection(synthetic.columns).tolist()
    res = hif_score(real, synthetic, columns=cols, random_state=SEED)
    return {
        "dataset": ds_id,
        "split": split,
        "n_train": n_train,
        "n_test": len(synthetic),
        "hif_score": round(float(res["hif_score"]), 4),
        "violation_rate": round(float(res["violation_rate"]), 4),
    }


rows: list[dict] = []
for ds_id in DATASETS:
    full = load_cached(ds_id).reset_index(drop=True)
    n = len(full)
    if n >= 4000:
        train = full.sample(2000, random_state=SEED)
        test = full.drop(train.index).sample(2000, random_state=SEED)
    elif n >= 2000:
        train = full.sample(2000, random_state=SEED)
        test = full.drop(train.index)
    else:
        train = full.sample(int(n * 0.8), random_state=SEED)
        test = full.drop(train.index)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    held_out = audit_rows(train, test, "held_out", len(train), len(test))
    replay = audit_rows(train, train, "replay_train", len(train), len(train))
    rows.extend([held_out, replay])

    print(
        f"{ds_id}: held-out HIF={held_out['hif_score']:.3f} "
        f"viol={held_out['violation_rate'] * 100:.1f}% (n_test={len(test)}) | "
        f"replay HIF={replay['hif_score']:.3f} "
        f"viol={replay['violation_rate'] * 100:.1f}%"
    )

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print(f"\nWrote {OUTPUT}")
