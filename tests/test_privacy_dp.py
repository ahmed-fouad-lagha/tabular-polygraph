import pandas as pd
import pytest

from tabular_polygraph.generators import GaussianCopulaGenerator


def _privacy_seed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25 + (i % 45) for i in range(400)],
            "income": [30000 + i * 200 for i in range(400)],
            "state": (["CA", "TX", "NY", "FL"] * 100),
            "education": (["HS", "College", "Graduate", "College"] * 100),
        }
    )


def test_no_exact_copies():
    from tabular_polygraph.privacy.audit import privacy_audit

    real = _privacy_seed()
    gen = GaussianCopulaGenerator()
    gen.fit(real)
    syn = gen.generate(300, seed=4)

    report = privacy_audit(real, syn, n_attacks=40, seed=1)
    assert report["exact_copies"]["count"] == 0


def test_membership_inference_auc_in_range():
    from tabular_polygraph.privacy.disclosure import membership_inference_risk

    real = _privacy_seed()
    gen = GaussianCopulaGenerator()
    gen.fit(real)
    syn = gen.generate(300, seed=5)

    result = membership_inference_risk(
        real_train=real.iloc[:300],
        real_holdout=real.iloc[300:],
        synthetic=syn,
        n_sample=60,
        seed=2,
    )
    assert "attack_auc" in result
    assert 0.0 <= result["attack_auc"] <= 1.0


def test_audit_verdict_keys():
    from tabular_polygraph.privacy import privacy_audit

    real = _privacy_seed()
    gen = GaussianCopulaGenerator()
    gen.fit(real)
    syn = gen.generate(250, seed=6)

    report = privacy_audit(real, syn, n_attacks=40, seed=3)
    v = report["verdict"]
    for key in [
        "overall_risk",
        "exact_copies",
        "mi_auc",
        "singling_out_rate",
        "linkability_rate",
        "recommendation",
    ]:
        assert key in v


def test_format_audit_returns_string():
    from tabular_polygraph.privacy import format_audit, privacy_audit

    real = _privacy_seed()
    gen = GaussianCopulaGenerator()
    gen.fit(real)
    syn = gen.generate(220, seed=7)

    report = privacy_audit(real, syn, n_attacks=30, seed=4)
    text = format_audit(report)
    assert "TAMIS PRIVACY ORACLE" in text


def test_budget_exhaustion_raises():
    from tabular_polygraph.privacy.dp import PrivacyBudget, laplace_mechanism

    budget = PrivacyBudget(epsilon=0.5)
    with pytest.raises(RuntimeError, match="exhausted"):
        laplace_mechanism(1.0, sensitivity=1.0, epsilon=1.0, budget=budget)


def test_privatise_histogram():
    from tabular_polygraph.privacy.dp import privatise_histogram

    counts = {"A": 100, "B": 50, "C": 25}
    result = privatise_histogram(counts, epsilon=2.0, seed=42)
    assert set(result.keys()) == set(counts.keys())
    assert abs(sum(result.values()) - 1.0) < 0.01


def test_budget_log_tracks_labels():
    from tabular_polygraph.privacy.dp import PrivacyBudget, laplace_mechanism

    budget = PrivacyBudget(epsilon=5.0)
    laplace_mechanism(1.0, 1.0, 1.0, budget=budget, label="mean_query")
    laplace_mechanism(1.0, 1.0, 1.0, budget=budget, label="std_query")
    labels = [entry["label"] for entry in budget.log]
    assert "mean_query" in labels
    assert "std_query" in labels
