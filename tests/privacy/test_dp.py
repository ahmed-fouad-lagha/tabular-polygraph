from __future__ import annotations

import numpy as np
import pytest

from tabular_polygraph.privacy.dp import (
    PrivacyBudget,
    gaussian_mechanism,
    laplace_mechanism,
    privatise_histogram,
)


class TestPrivacyBudget:
    def test_valid_budget(self):
        b = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert b.epsilon == 1.0
        assert b.delta == 1e-5
        assert b.remaining_epsilon == 1.0

    def test_epsilon_must_be_positive(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            PrivacyBudget(epsilon=0)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            PrivacyBudget(epsilon=-1)

    def test_delta_must_be_non_negative(self):
        PrivacyBudget(epsilon=1.0, delta=0)
        PrivacyBudget(epsilon=1.0, delta=1e-5)
        with pytest.raises(ValueError, match="delta must be non-negative"):
            PrivacyBudget(epsilon=1.0, delta=-0.01)

    def test_consume_tracks_epsilon(self):
        b = PrivacyBudget(epsilon=1.0)
        b.consume(eps=0.3, label="step1")
        assert b._used_epsilon == 0.3
        assert b.remaining_epsilon == 0.7
        b.consume(eps=0.7, label="step2")
        assert b.remaining_epsilon == 0.0

    def test_consume_exhaustion_raises(self):
        b = PrivacyBudget(epsilon=0.5)
        b.consume(eps=0.5)
        with pytest.raises(RuntimeError, match="Privacy budget exhausted"):
            b.consume(eps=0.1)

    def test_consume_logs(self):
        b = PrivacyBudget(epsilon=2.0)
        b.consume(eps=0.5, delta=1e-5, label="test_op")
        log = b.log
        assert len(log) == 1
        assert log[0]["label"] == "test_op"
        assert log[0]["epsilon"] == 0.5
        assert log[0]["delta"] == 1e-5

    def test_log_is_copy(self):
        b = PrivacyBudget(epsilon=1.0)
        b.consume(0.5)
        log = b.log
        log.append({"label": "x"})
        assert len(b.log) == 1

    def test_repr(self):
        b = PrivacyBudget(epsilon=1.0)
        assert "ε=1.0" in repr(b)


class TestLaplaceMechanism:
    def test_noisy_output(self):
        result = laplace_mechanism(value=42.0, sensitivity=1.0, epsilon=1.0)
        assert isinstance(result, float)
        assert result != 42.0

    def test_reproducible_seed(self):
        a = laplace_mechanism(42.0, 1.0, 1.0, seed=99)
        b = laplace_mechanism(42.0, 1.0, 1.0, seed=99)
        assert a == b

    def test_different_seeds_differ(self):
        a = laplace_mechanism(42.0, 1.0, 1.0, seed=1)
        b = laplace_mechanism(42.0, 1.0, 1.0, seed=2)
        assert a != b

    def test_epsilon_must_be_positive(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            laplace_mechanism(42.0, 1.0, epsilon=0)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            laplace_mechanism(42.0, 1.0, epsilon=-1)

    def test_consumes_budget(self):
        budget = PrivacyBudget(epsilon=1.0)
        laplace_mechanism(42.0, 1.0, 0.3, budget=budget, label="lap")
        assert budget.remaining_epsilon == 0.7
        assert budget.log[0]["label"] == "lap"

    def test_array_input(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = laplace_mechanism(arr, sensitivity=1.0, epsilon=1.0, seed=42)
        assert result.shape == (3,)
        assert not np.allclose(result, arr)


class TestGaussianMechanism:
    def test_noisy_output(self):
        result = gaussian_mechanism(value=42.0, sensitivity=1.0, epsilon=1.0)
        assert isinstance(result, float)
        assert result != 42.0

    def test_reproducible_seed(self):
        a = gaussian_mechanism(42.0, 1.0, 1.0, seed=99)
        b = gaussian_mechanism(42.0, 1.0, 1.0, seed=99)
        assert a == b

    def test_epsilon_must_be_positive(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            gaussian_mechanism(42.0, 1.0, epsilon=0)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            gaussian_mechanism(42.0, 1.0, epsilon=-1)

    def test_delta_must_be_positive(self):
        with pytest.raises(ValueError, match="delta must be positive"):
            gaussian_mechanism(42.0, 1.0, epsilon=1.0, delta=0)
        with pytest.raises(ValueError, match="delta must be positive"):
            gaussian_mechanism(42.0, 1.0, epsilon=1.0, delta=-0.01)

    def test_consumes_budget(self):
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        gaussian_mechanism(42.0, 1.0, 0.3, budget=budget, label="gauss")
        assert budget.remaining_epsilon == 0.7

    def test_array_input(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = gaussian_mechanism(arr, sensitivity=1.0, epsilon=1.0, seed=42)
        assert result.shape == (3,)


class TestPrivatiseHistogram:
    def test_noisy_counts(self):
        counts = {"a": 10, "b": 20, "c": 30}
        result = privatise_histogram(counts, epsilon=1.0, seed=42)
        assert set(result.keys()) == {"a", "b", "c"}
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6

    def test_reproducible_seed(self):
        counts = {"a": 10, "b": 20}
        a = privatise_histogram(counts, epsilon=1.0, seed=7)
        b = privatise_histogram(counts, epsilon=1.0, seed=7)
        for k in counts:
            assert a[k] == b[k]

    def test_all_zero_keys(self):
        counts = {"a": 0, "b": 0}
        result = privatise_histogram(counts, epsilon=1.0, seed=42)
        assert list(result.keys()) == ["a", "b"]

    def test_consumes_budget(self):
        budget = PrivacyBudget(epsilon=1.0)
        privatise_histogram({"x": 5}, epsilon=0.5, budget=budget)
        assert budget._used_epsilon == 0.5

    def test_negative_noise_clipped(self):
        counts = {"x": 1}
        result = privatise_histogram(counts, epsilon=1000.0, seed=0)
        for v in result.values():
            assert v >= 0
