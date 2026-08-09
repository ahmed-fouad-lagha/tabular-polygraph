import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

_spec = importlib.util.spec_from_file_location(
    "heldout_matched_threshold", SCRIPTS_DIR / "05_heldout_matched_threshold.py"
)
matched_threshold = importlib.util.module_from_spec(_spec)
sys.modules["heldout_matched_threshold"] = matched_threshold
_spec.loader.exec_module(matched_threshold)

compute_matched_f1 = matched_threshold.compute_matched_f1


def _reference_matched_f1(scores, labels, level):
    n_flag = max(1, int(len(scores) * level))
    order = np.argsort(scores)[::-1]
    preds = np.zeros(len(scores), dtype=int)
    preds[order[:n_flag]] = 1
    return float(f1_score(labels, preds, zero_division=0.0))


def test_matched_f1_perfect_ranking():
    rng = np.random.default_rng(0)
    n = 1000
    labels = np.zeros(n, dtype=bool)
    corrupt = rng.choice(n, size=int(n * 0.4), replace=False)
    labels[corrupt] = True
    scores = np.full(n, 0.1)
    scores[corrupt] = 0.9
    assert compute_matched_f1(scores, labels, 0.4) == pytest.approx(1.0)


def test_matched_f1_matches_reference():
    rng = np.random.default_rng(42)
    for _ in range(5):
        n = rng.integers(100, 2000)
        labels = rng.random(n) < 0.4
        scores = rng.random(n)
        for level in (0.1, 0.3, 0.4, 0.5, 0.7, 1.0):
            assert compute_matched_f1(scores, labels, level) == pytest.approx(
                _reference_matched_f1(scores, labels, level)
            )


def test_matched_f1_marks_corrupt_rows():
    rng = np.random.default_rng(1)
    n = 500
    level = 0.4
    labels = np.zeros(n, dtype=bool)
    corrupt = rng.choice(n, size=int(n * level), replace=False)
    labels[corrupt] = True
    scores = rng.random(n)
    f1 = compute_matched_f1(scores, labels, level)
    assert 0.0 <= f1 <= 1.0
    assert f1 == pytest.approx(_reference_matched_f1(scores, labels, level))


def test_matched_f1_zero_level_flags_single_row():
    n = 100
    labels = np.zeros(n, dtype=bool)
    labels[:50] = True
    scores = np.arange(n, dtype=float)
    n_flag = max(1, int(n * 0))
    assert n_flag == 1
    assert compute_matched_f1(scores, labels, 0.0) == pytest.approx(
        _reference_matched_f1(scores, labels, 0.0)
    )


def test_matched_f1_full_level_all_corrupt():
    n = 200
    labels = np.ones(n, dtype=bool)
    scores = np.arange(n, dtype=float)
    assert compute_matched_f1(scores, labels, 1.0) == pytest.approx(1.0)


def test_matched_f1_no_true_positives():
    n = 200
    labels = np.zeros(n, dtype=bool)
    scores = np.arange(n, dtype=float)
    assert compute_matched_f1(scores, labels, 0.4) == pytest.approx(0.0)


def test_matched_f1_flags_more_than_default_threshold():
    rng = np.random.default_rng(7)
    n = 1000
    labels = np.zeros(n, dtype=bool)
    corrupt = rng.choice(n, size=int(n * 0.4), replace=False)
    labels[corrupt] = True
    scores = rng.uniform(0.5, 0.6, size=n)
    scores[corrupt] = rng.uniform(0.9, 1.0, size=len(corrupt))
    default_f1 = float(f1_score(labels, (scores > 0.5).astype(int), zero_division=0.0))
    assert default_f1 < 1.0
    assert compute_matched_f1(scores, labels, 0.4) == pytest.approx(1.0)
    assert compute_matched_f1(scores, labels, 0.4) > default_f1
