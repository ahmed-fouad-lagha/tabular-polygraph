import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

_spec = importlib.util.spec_from_file_location(
    "arithmetic_identity_verification",
    SCRIPTS_DIR / "09_arithmetic_identity_verification.py",
)
module = importlib.util.module_from_spec(_spec)
sys.modules["arithmetic_identity_verification"] = module
_spec.loader.exec_module(module)

TOLERANCE = module.TOLERANCE


@pytest.fixture
def supermarket_real():
    rng = np.random.default_rng(0)
    n = 300
    unit_price = rng.uniform(10, 100, size=n)
    quantity = rng.integers(1, 6, size=n)
    subtotal = unit_price * quantity
    tax = 0.05 * subtotal
    total = subtotal + tax
    cogs = rng.uniform(0.2, 0.8, size=n) * subtotal
    gross_income = total - cogs
    return pd.DataFrame(
        {
            "total": total,
            "unit_price": unit_price,
            "quantity": quantity,
            "cogs": cogs,
            "gross_income": gross_income,
        }
    )


@pytest.fixture
def online_real():
    rng = np.random.default_rng(1)
    n = 300
    purchase_price = rng.uniform(5, 200, size=n)
    quantity = rng.integers(1, 4, size=n)
    item_subtotal = purchase_price * quantity
    item_tax = 0.08 * item_subtotal
    item_total = item_subtotal + item_tax
    return pd.DataFrame(
        {
            "item_total": item_total,
            "item_subtotal": item_subtotal,
            "item_tax": item_tax,
            "purchase_price": purchase_price,
            "quantity": quantity,
        }
    )


def test_identities_hold_on_consistent_frames(supermarket_real, online_real):
    for ds_id, frame in [
        ("supermarket_sales", supermarket_real),
        ("online_purchases", online_real),
    ]:
        err = module.DATASET_IDENTITIES[ds_id]["error"](frame)
        assert err.shape == (len(frame),)
        assert np.isfinite(err).all()
        assert float(err.max()) < TOLERANCE
        assert float(err.min()) >= 0.0


def test_identities_detect_violations(supermarket_real, online_real):
    corrupted_super = supermarket_real.copy()
    corrupted_super["total"] = (
        corrupted_super["total"].sample(frac=1.0, random_state=0).values
    )
    corrupted_online = online_real.copy()
    corrupted_online["item_subtotal"] = corrupted_online["item_subtotal"] * 1.5

    err_super = module.DATASET_IDENTITIES["supermarket_sales"]["error"](corrupted_super)
    err_online = module.DATASET_IDENTITIES["online_purchases"]["error"](
        corrupted_online
    )
    assert (err_super > TOLERANCE).mean() > 0.9
    assert (err_online > TOLERANCE).mean() > 0.9


def test_real_data_satisfies_identities():
    for ds_id, spec in module.DATASET_IDENTITIES.items():
        try:
            real = module.load_real(ds_id, n=spec["n"])
        except ValueError as exc:  # dataset not cached
            pytest.skip(str(exc))
        err = spec["error"](real[spec["cols"]])
        assert float(err.max()) < TOLERANCE


def test_summary_aggregates_seeds():
    rng = np.random.default_rng(3)
    n = len(module.DATASET_IDENTITIES) * len(module.GENERATORS)
    detail = pd.DataFrame(
        {
            "dataset": [
                ds for ds in module.DATASET_IDENTITIES for _ in module.GENERATORS
            ],
            "generator": list(module.GENERATORS) * len(module.DATASET_IDENTITIES),
            "flag_rate": rng.uniform(0.1, 1.0, size=n),
            "base_violation_rate": rng.uniform(0.5, 1.0, size=n),
            "confirmation_rate_flagged": rng.uniform(0.9, 1.0, size=n),
            "confirmation_rate_unflagged": rng.uniform(0.5, 1.0, size=n),
            "median_severity_flagged": rng.uniform(0.1, 10.0, size=n),
            "median_severity_unflagged": rng.uniform(0.1, 10.0, size=n),
            "spearman_penalty_severity": rng.uniform(-0.1, 0.6, size=n),
        }
    )
    summary = module.summarize(detail)
    assert len(summary) == n
    assert summary["spearman_sem"].isna().all()  # single seed per cell
    assert set(summary.columns) >= {
        "flag_rate",
        "base_violation_rate",
        "confirmation_rate_flagged",
        "confirmation_rate_unflagged",
        "spearman_penalty_severity",
        "spearman_sem",
    }
