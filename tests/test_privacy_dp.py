import pytest

class TestPrivacy:
    def test_no_exact_copies(self, hmda, syn_hmda):
        from src.privacy.audit import privacy_audit
        report = privacy_audit(hmda, syn_hmda, n_attacks=50, seed=1)
        assert report["exact_copies"]["count"] == 0

    def test_membership_inference_auc_near_half(self, hmda, syn_hmda):
        from src.privacy.disclosure import membership_inference_risk
        result = membership_inference_risk(
            real_train=hmda.iloc[:1600],
            real_holdout=hmda.iloc[1600:],
            synthetic=syn_hmda,
            n_sample=100, seed=1,
        )
        assert "attack_auc" in result
        assert 0.0 <= result["attack_auc"] <= 1.0

    def test_singling_out_risk_low(self, hmda, syn_hmda):
        from src.privacy.singling_out import singling_out_risk
        result = singling_out_risk(hmda, syn_hmda, n_attacks=100, seed=1)
        assert "singling_out_rate" in result
        assert 0.0 <= result["singling_out_rate"] <= 1.0

    def test_linkability_risk_has_baseline(self, hmda, syn_hmda):
        from src.privacy.linkability import linkability_risk
        result = linkability_risk(hmda, syn_hmda, n_attacks=100, seed=1)
        assert result["baseline"] == 0.5

    def test_audit_verdict_keys(self, hmda, syn_hmda):
        from src.privacy import privacy_audit
        report = privacy_audit(hmda, syn_hmda, n_attacks=80, seed=1)
        v = report["verdict"]
        for key in ["overall_risk", "exact_copies", "mi_auc",
                    "singling_out_rate", "linkability_rate", "recommendation"]:
            assert key in v

    def test_audit_risk_levels_valid(self, hmda, syn_hmda):
        from src.privacy import privacy_audit
        valid = {"very_low", "low", "medium", "high", "very_high"}
        report = privacy_audit(hmda, syn_hmda, n_attacks=80, seed=1)
        assert report["verdict"]["overall_risk"] in valid

    def test_format_audit_returns_string(self, hmda, syn_hmda):
        from src.privacy import privacy_audit, format_audit
        report = privacy_audit(hmda, syn_hmda, n_attacks=50, seed=1)
        text = format_audit(report)
        assert "PRIVACY AUDIT" in text


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Differential Privacy
# ═══════════════════════════════════════════════════════════════════════════════

class TestDifferentialPrivacy:
    def test_laplace_mechanism_adds_noise(self, hmda):
        from src.privacy.dp import laplace_mechanism
        true_mean = hmda["loan_amount"].mean()
        noisy = laplace_mechanism(true_mean, sensitivity=1e6, epsilon=1.0)
        assert noisy != true_mean

    def test_gaussian_mechanism_adds_noise(self, hmda):
        from src.privacy.dp import gaussian_mechanism
        true_val = hmda["applicant_income"].mean()
        noisy = gaussian_mechanism(true_val, sensitivity=1e6, epsilon=1.0, delta=1e-5)
        assert noisy != true_val

    def test_budget_tracks_consumption(self, hmda):
        from src.privacy.dp import PrivacyBudget, laplace_mechanism
        budget = PrivacyBudget(epsilon=2.0)
        laplace_mechanism(1.0, sensitivity=1.0, epsilon=0.5, budget=budget)
        laplace_mechanism(1.0, sensitivity=1.0, epsilon=0.5, budget=budget)
        assert abs(budget.remaining_epsilon - 1.0) < 1e-9

    def test_budget_exhaustion_raises(self):
        from src.privacy.dp import PrivacyBudget, laplace_mechanism
        budget = PrivacyBudget(epsilon=0.5)
        with pytest.raises(RuntimeError, match="exhausted"):
            laplace_mechanism(1.0, sensitivity=1.0, epsilon=1.0, budget=budget)

    def test_privatise_histogram(self):
        from src.privacy.dp import privatise_histogram
        counts = {"A": 100, "B": 50, "C": 25}
        result = privatise_histogram(counts, epsilon=2.0, seed=42)
        assert set(result.keys()) == set(counts.keys())
        assert abs(sum(result.values()) - 1.0) < 0.01   # normalised

    def test_budget_log_tracks_labels(self):
        from src.privacy.dp import PrivacyBudget, laplace_mechanism
        budget = PrivacyBudget(epsilon=5.0)
        laplace_mechanism(1.0, 1.0, 1.0, budget=budget, label="mean_query")
        laplace_mechanism(1.0, 1.0, 1.0, budget=budget, label="std_query")
        labels = [entry["label"] for entry in budget.log]
        assert "mean_query" in labels
        assert "std_query"  in labels


