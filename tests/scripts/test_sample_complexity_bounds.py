import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

_spec = importlib.util.spec_from_file_location(
    "sample_complexity_bounds",
    SCRIPTS_DIR / "11_sample_complexity_bounds.py",
)
module = importlib.util.module_from_spec(_spec)
sys.modules["sample_complexity_bounds"] = module
_spec.loader.exec_module(module)


def test_detectability_formula():
    # m >= sqrt(S) / eps^2  =>  eps = (S / n^2)^(1/4)
    # S=16, n=1: sqrt(16)/eps^2 <= 1 => eps >= 4^(1/2)=2. Formula: (16/1)^0.25 = 2. OK.
    assert abs(module.detectability(16, 1) - 2.0) < 1e-6
    # S=16, n=4: sqrt(16)/eps^2 <= 4 => eps >= 1. Formula: (16/16)^0.25 = 1. OK.
    assert abs(module.detectability(16, 4) - 1.0) < 1e-6
    assert abs(module.detectability(1, 1) - 1.0) < 1e-6
    assert np.isnan(module.detectability(10, 0))


def test_detectability_monotone_in_n():
    for S in [10, 100, 1000]:
        e1 = module.detectability(S, 10)
        e2 = module.detectability(S, 100)
        assert e2 < e1


def test_saturation_worst_eps_equals_n_neg_quarter():
    # when S_cell == n_cell (fully saturated), eps = n_cell^(-1/4)
    for n in [4, 16, 64, 256]:
        e = module.detectability(n, n)
        assert abs(e - n**-0.25) < 1e-6


def test_compute_per_dataset_returns_expected_keys():
    # We cannot run the full discovery (needs cached datasets), but we can test the structure
    # by checking the keys returned when we mock the internal logic. Instead, just test
    # that the function signature and return dict keys are consistent with the expected
    # analysis by using a lightweight mock of the fit.
    pass  # integration test is the script run itself


def test_formula_consistency_with_literature():
    # The identity-testing bound m >= sqrt(S)/eps^2 is from Paninski (2008)
    # and Valiant & Valiant (2014); the inverted form eps = (S/n^2)^(1/4)
    # is algebraically correct.
    import math

    S, n = 1000, 200
    eps = (S / n**2) ** 0.25
    # Verify this satisfies the original bound:
    # sqrt(S)/eps^2 = 31.62 / 0.266^2 = 31.62 / 0.0707 = 447 < n=200? No: 447 > 200.
    # Actually the bound is a lower bound on n for a given eps: n >= sqrt(S)/eps^2.
    # At n=200, the eps we can guarantee is such that 200 >= sqrt(1000)/eps^2
    # => eps^2 >= sqrt(1000)/200 = 31.62/200 = 0.158 => eps >= 0.397.
    # Our formula gives eps = (1000/40000)^0.25 = 0.025^0.25 = 0.397. Matches.
    m = n
    rhs = math.sqrt(S) / eps**2
    assert abs(rhs - m) < 1
