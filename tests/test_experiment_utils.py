"""Regression tests for leakage-free experiment helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from _exp_utils import split_real_for_utility, utility_metrics_on_holdout  # noqa: E402


def _load_leakage_free_runner():
    script = SCRIPTS_DIR / "19_leakage_free_utility.py"
    spec = importlib.util.spec_from_file_location("leakage_free_utility", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_utility_metrics_on_holdout_handles_numeric_target() -> None:
    train = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0] * 4,
            "segment": ["a", "a", "b", "b"] * 4,
            "target": [0.0, 1.0, 2.0, 3.0] * 4,
        }
    )
    test = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0],
            "segment": ["a", "a", "b", "unseen"],
            "target": [0.0, 1.0, 2.0, 3.0],
        }
    )

    result = utility_metrics_on_holdout(train, train.copy(), test, "target")

    assert 0.0 <= result["f1"] <= 1.0
    assert 0.0 <= result["accuracy"] <= 1.0


def test_split_real_for_utility_is_disjoint_and_stratifies_discrete_targets() -> None:
    real = pd.DataFrame(
        {"row_id": range(20), "feature": range(20), "target": [0, 1] * 10}
    )

    train, test = split_real_for_utility(real, "target", seed=42, test_size=0.30)

    assert len(train) == 14
    assert len(test) == 6
    assert set(train["row_id"]).isdisjoint(set(test["row_id"]))
    assert test["target"].value_counts().to_dict() == {0: 3, 1: 3}


def test_utility_metrics_on_holdout_handles_categorical_target() -> None:
    train = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0] * 4,
            "target": ["low", "low", "high", "high"] * 4,
        }
    )
    test = pd.DataFrame(
        {"feature": [0.0, 1.0, 2.0, 3.0], "target": ["low", "low", "high", "high"]}
    )

    result = utility_metrics_on_holdout(train, train.copy(), test, "target")

    assert 0.0 <= result["f1"] <= 1.0
    assert 0.0 <= result["accuracy"] <= 1.0


def test_utility_metrics_on_holdout_rejects_missing_target() -> None:
    frame = pd.DataFrame({"feature": range(12), "target": [0, 1] * 6})

    result = utility_metrics_on_holdout(
        frame, frame.drop(columns="target"), frame, "target"
    )

    assert np.isnan(result["f1"])
    assert np.isnan(result["accuracy"])


def test_leakage_free_runner_never_passes_test_rows_to_fitters(tmp_path) -> None:
    runner = _load_leakage_free_runner()
    real = pd.DataFrame(
        {
            "row_id": range(20),
            "feature": range(20),
            "target": [0, 1] * 10,
        }
    )
    calls: dict[str, object] = {}

    def fake_load_real(dataset_id: str, n: int, seed: int) -> pd.DataFrame:
        calls["load_count"] = int(calls.get("load_count", 0)) + 1
        assert (dataset_id, n, seed) == ("toy", 20, 42)
        return real.copy()

    def fake_generate(train: pd.DataFrame, n: int, seed: int, generator: str):
        calls["generator_rows"] = set(train["row_id"])
        assert len(train) == 14
        assert n == 20
        assert seed == 42
        assert generator == "gaussian_copula"
        return train.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    def fake_audit(train: pd.DataFrame, synthetic: pd.DataFrame, seed: int) -> dict:
        calls["auditor_rows"] = set(train["row_id"])
        assert set(train["row_id"]) == calls["generator_rows"]
        return {
            "row_penalties": np.zeros(len(synthetic)),
            "hif_score": 1.0,
            "violation_rate": 0.0,
        }

    def fake_utility(
        train: pd.DataFrame,
        synthetic: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        seed: int,
    ) -> dict:
        calls["utility_train_rows"] = set(train["row_id"])
        calls["utility_test_rows"] = set(test["row_id"])
        assert set(train["row_id"]).isdisjoint(set(test["row_id"]))
        assert len(test) == 6
        assert target == "target"
        assert seed == 42
        return {"f1": 0.5, "accuracy": 0.5}

    runner.load_real = fake_load_real
    runner.generate = fake_generate
    runner.audit_hif = fake_audit
    runner.utility_metrics_on_holdout = fake_utility
    raw_path, summary_path = runner.run_experiment(
        dataset_id="toy",
        target="target",
        generator_type="gaussian_copula",
        n_rows=20,
        n_seeds=1,
        test_size=0.30,
        output_dir=tmp_path,
    )

    assert calls["generator_rows"] == calls["auditor_rows"]
    assert calls["generator_rows"] == calls["utility_train_rows"]
    assert raw_path.exists()
    assert summary_path.exists()

    runner.run_experiment(
        dataset_id="toy",
        target="target",
        generator_type="gaussian_copula",
        n_rows=20,
        n_seeds=1,
        test_size=0.30,
        output_dir=tmp_path,
    )
    assert calls["load_count"] == 1
