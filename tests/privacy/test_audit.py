from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph.privacy.audit import format_audit, privacy_audit


def test_privacy_audit_smoke():
    rng = np.random.default_rng(42)
    n = 100
    real = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.choice(["X", "Y"], n),
            "c": rng.normal(0, 1, n),
        }
    )
    syn = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.choice(["X", "Y"], n),
            "c": rng.normal(0, 1, n),
        }
    )
    report = privacy_audit(real, syn, n_attacks=10, seed=42)
    assert "verdict" in report
    assert "exact_copies" in report
    assert "membership_inference" in report
    assert "singling_out" in report
    assert "linkability" in report
    assert "overall_risk" in report["verdict"]


def test_privacy_audit_exact_copies_detected():
    real = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        }
    )
    syn = real.copy()
    report = privacy_audit(real, syn, n_attacks=10, seed=42)
    assert report["exact_copies"]["count"] > 0
    assert report["exact_copies"]["risk_level"] == "very_high"


def test_privacy_audit_holdout_provided():
    rng = np.random.default_rng(42)
    n = 100
    real = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.choice(["X", "Y"], n),
        }
    )
    holdout = pd.DataFrame(
        {
            "a": rng.normal(5, 1, n),
            "b": rng.choice(["X", "Y"], n),
        }
    )
    syn = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.choice(["X", "Y"], n),
        }
    )
    report = privacy_audit(real, syn, real_holdout=holdout, n_attacks=10, seed=42)
    assert "verdict" in report


def test_format_audit():
    report = {
        "exact_copies": {"count": 0, "risk_level": "very_low"},
        "membership_inference": {
            "attack_auc": 0.51,
            "risk_level": "very_low",
            "interpretation": "No meaningful memorisation detected",
        },
        "singling_out": {"singling_out_rate": 0.0, "risk_level": "very_low"},
        "linkability": {
            "linkability_rate": 0.5,
            "risk_level": "very_low",
            "lift_over_baseline_pct": 0.0,
        },
        "verdict": {
            "overall_risk": "very_low",
            "exact_copies": 0,
            "mi_auc": 0.51,
            "singling_out_rate": 0.0,
            "linkability_rate": 0.5,
            "elapsed_seconds": 0.1,
            "recommendation": "PASS: all privacy tests pass. Safe to release.",
        },
    }
    formatted = format_audit(report)
    assert "TAMIS PRIVACY ORACLE REPORT" in formatted
    assert "PASS" in formatted
