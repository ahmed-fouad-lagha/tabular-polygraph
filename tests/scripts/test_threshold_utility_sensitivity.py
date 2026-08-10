import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

_spec = importlib.util.spec_from_file_location(
    "threshold_utility_sensitivity",
    SCRIPTS_DIR / "10_threshold_utility_sensitivity.py",
)
module = importlib.util.module_from_spec(_spec)
sys.modules["threshold_utility_sensitivity"] = module
_spec.loader.exec_module(module)


def test_thresholds_define_operating_points():
    assert module.THRESHOLDS == [0.3, 0.5, 0.7]


def test_summarize_reports_one_row_per_threshold():
    rng = np.random.default_rng(7)
    n = 9  # 3 seeds per threshold
    full_f1 = rng.uniform(0.40, 0.45, size=n)
    raw = pd.DataFrame(
        {
            "seed": np.arange(n),
            "threshold": list(module.THRESHOLDS) * 3,
            "full_f1": full_f1,
            "rule_f1": full_f1,
            "filtered_f1": full_f1 + rng.uniform(0.05, 0.2, size=n),
            "retention": rng.uniform(30, 95, size=n),
        }
    )
    summary = module.summarize(raw)
    assert len(summary) == len(module.THRESHOLDS)
    assert set(summary["threshold"]) == set(module.THRESHOLDS)
    assert (summary["delta_f1"] > 0).all()
    assert (summary["p_ttest"] < 0.05).all()
    assert (summary["ci_low"] > 0).all()


def test_summarize_detects_no_effect_when_differences_are_zero():
    n = 9
    raw = pd.DataFrame(
        {
            "seed": np.arange(n),
            "threshold": list(module.THRESHOLDS) * 3,
            "full_f1": 0.5,
            "rule_f1": 0.5,
            "filtered_f1": 0.5,
            "retention": 90.0,
        }
    )
    summary = module.summarize(raw)
    assert (summary["delta_f1"].abs() < 1e-6).all()
    # zero-variance differences give an undefined t-test; never a spurious effect
    assert not (summary["p_ttest"] < 0.05).any()
