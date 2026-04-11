import numpy as np
import pandas as pd


class TestMarginalFidelity:
    def test_scores_in_range(self, hmda, syn_hmda):
        from src.fidelity.marginal import moment_matching_scores

        scores = moment_matching_scores(hmda, syn_hmda.drop(columns=["syn_id"]))
        for col, score in scores.items():
            assert 0 <= score <= 100, f"{col} score {score} out of range"

    def test_high_fidelity_on_large_sample(self, hmda, syn_hmda):
        from src.fidelity.marginal import (
            mean_moment_matching_score,
            moment_matching_scores,
        )

        scores = moment_matching_scores(hmda, syn_hmda.drop(columns=["syn_id"]))
        assert mean_moment_matching_score(scores) >= 80.0

    def test_identical_data_scores_100(self, hmda):
        from src.fidelity.marginal import moment_matching_scores

        scores = moment_matching_scores(hmda, hmda)
        for score in scores.values():
            assert score >= 99.0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Fidelity — Logical
# ═══════════════════════════════════════════════════════════════════════════════


class TestLogicalFidelity:
    def test_neuro_lcv_penalizes_unseen_categories(self):
        import torch
        from src.fidelity.logical import lcv_score

        np.random.seed(0)
        torch.manual_seed(0)

        real = pd.DataFrame(
            {
                "state": ["CA", "CA", "NY", "NY", "TX", "TX"] * 20,
                "county": ["001", "003", "005", "007", "009", "011"] * 20,
                "class": ["A", "B", "A", "B", "A", "B"] * 20,
            }
        )

        clean = real.sample(frac=1.0, random_state=0).reset_index(drop=True)
        bad = clean.copy()
        bad.loc[:19, "state"] = "__ILLOGICAL__"
        bad.loc[:19, "county"] = "__ILLOGICAL__"

        clean_result = lcv_score(
            real, clean, columns=["state", "county", "class"], epochs=6, verbose=False
        )
        bad_result = lcv_score(
            real, bad, columns=["state", "county", "class"], epochs=6, verbose=False
        )

        assert clean_result["lcv_score"] > bad_result["lcv_score"]
        assert clean_result["mean_penalty"] < bad_result["mean_penalty"]

    def test_neuro_lcv_canonicalizes_code_columns(self):
        import torch
        from src.fidelity.logical import lcv_score

        np.random.seed(1)
        torch.manual_seed(1)

        real = pd.DataFrame(
            {
                "state_fips": [
                    "06",
                    "08",
                    "06",
                    "04",
                    "06",
                    "08",
                    "12",
                    "06",
                    "08",
                    "04",
                ]
                * 40,
                "county": [
                    "037",
                    "109",
                    "083",
                    "019",
                    "001",
                    "097",
                    "071",
                    "073",
                    "005",
                    "111",
                ]
                * 40,
            }
        )

        clean = pd.DataFrame(
            {
                "state_fips": [6, 8, 6, 4, 6, 8, 12, 6, 8, 4] * 40,
                "county": [37, 109, 83, 19, 1, 97, 71, 73, 5, 111] * 40,
            }
        )

        bad = clean.astype({"state_fips": "object", "county": "object"}).copy()
        bad.loc[:199, "state_fips"] = "__ILLOGICAL__"
        bad.loc[:199, "county"] = "__ILLOGICAL__"

        clean_result = lcv_score(
            real, clean, columns=["state_fips", "county"], epochs=6, verbose=False
        )
        bad_result = lcv_score(
            real, bad, columns=["state_fips", "county"], epochs=6, verbose=False
        )

        assert clean_result["lcv_score"] > bad_result["lcv_score"]
        assert clean_result["mean_penalty"] < bad_result["mean_penalty"]
        assert clean_result["violation_rate"] <= bad_result["violation_rate"]

    def test_rule_violation_score_penalizes_corruption(self):
        from src.fidelity.logical import rule_violation_score

        real = pd.DataFrame(
            {
                "state": ["CA", "CA", "TX", "TX", "NY", "NY"] * 40,
                "county": ["001", "001", "005", "005", "003", "003"] * 40,
                "segment": ["urban", "urban", "rural", "rural", "urban", "urban"] * 40,
            }
        )

        clean = real.sample(frac=1.0, random_state=10).reset_index(drop=True)
        bad = clean.copy()
        bad.loc[:79, "county"] = "999"

        clean_rules = rule_violation_score(
            real, clean, columns=["state", "county", "segment"]
        )
        bad_rules = rule_violation_score(
            real, bad, columns=["state", "county", "segment"]
        )

        assert clean_rules["num_rules_mined"] > 0
        assert clean_rules["rule_violation_rate"] < bad_rules["rule_violation_rate"]
        assert clean_rules["num_rule_violations"] < bad_rules["num_rule_violations"]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Fidelity — Joint
# ═══════════════════════════════════════════════════════════════════════════════


class TestJointFidelity:
    def test_correlation_score_range(self, hmda, syn_hmda):
        from src.fidelity.joint import correlation_distance_score

        score = correlation_distance_score(hmda, syn_hmda.drop(columns=["syn_id"]))
        assert 0 <= score <= 100

    def test_identical_data_perfect_joint(self, hmda):
        from src.fidelity.joint import correlation_distance_score

        score = correlation_distance_score(hmda, hmda)
        assert score >= 99.0

    def test_pairwise_report_returns_dict(self, hmda, syn_hmda):
        from src.fidelity.joint import pairwise_correlation_report

        result = pairwise_correlation_report(hmda, syn_hmda.drop(columns=["syn_id"]))
        assert isinstance(result, dict)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Fidelity — Temporal
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemporalFidelity:
    def test_stationarity_agreement(self, fred_macro, syn_macro):
        from src.fidelity.temporal.stationarity import stationarity_score

        result = stationarity_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        rate = result["_summary"]["agreement_rate"]
        assert 0 <= rate <= 100

    def test_cointegration_agreement(self, fred_macro, syn_macro):
        from src.fidelity.temporal.cointegration import cointegration_score

        result = cointegration_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        assert 0 <= result["_summary"]["agreement_rate"] <= 100

    def test_breaks_score(self, fred_macro, syn_macro):
        from src.fidelity.temporal.breaks import breaks_score

        result = breaks_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        assert 0 <= result["_summary"]["break_match_rate"] <= 100

    def test_causality_score(self, fred_macro, syn_macro):
        from src.fidelity.causality import causality_score

        result = causality_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        assert 0 <= result["_summary"]["agreement_rate"] <= 100


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Fidelity — Stylized Facts & Downstream
# ═══════════════════════════════════════════════════════════════════════════════


class TestStyleAndDownstream:
    def test_stylized_facts_score(self, fred_macro, syn_macro):
        from src.fidelity.stylized_facts import stylized_facts_score

        result = stylized_facts_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        assert 0 <= result["_summary"]["mean_score"] <= 100

    def test_tstr_classification(self, credit_risk):
        from src.generators import GaussianCopulaGenerator
        from src.fidelity.downstream import tstr_score

        gen = GaussianCopulaGenerator()
        gen.fit(credit_risk)
        syn = gen.generate(300, seed=1).drop(columns=["syn_id"])
        result = tstr_score(
            credit_risk, syn, target_col="default_12m", task="classification"
        )
        assert "tstr_score" in result
        assert "trr_score" in result
        assert "ratio" in result

    def test_tstr_regression(self, hmda):
        from src.generators import GaussianCopulaGenerator
        from src.fidelity.downstream import tstr_score

        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        syn = gen.generate(300, seed=1).drop(columns=["syn_id"])
        result = tstr_score(hmda, syn, target_col="loan_amount", task="regression")
        assert result["metric"] == "r2"
        assert "ratio" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Fidelity Report
# ═══════════════════════════════════════════════════════════════════════════════


class TestFidelityReport:
    def test_cross_sectional_report_keys(self, hmda, syn_hmda):
        from src.fidelity import fidelity_report

        report = fidelity_report(hmda, syn_hmda.drop(columns=["syn_id"]))
        for key in [
            "moment_matching",
            "distribution_fit",
            "joint",
            "stylized_facts",
            "privacy_basic",
            "summary",
        ]:
            assert key in report

    def test_summary_scores_in_range(self, hmda, syn_hmda):
        from src.fidelity import fidelity_report

        report = fidelity_report(hmda, syn_hmda.drop(columns=["syn_id"]))
        s = report["summary"]
        assert 0 <= s["overall_fidelity"] <= 100
        assert 0 <= s["moment_matching_score"] <= 100
        assert 0 <= s["ks_score"] <= 100
        assert 0 <= s["joint_score"] <= 100
        assert s["exact_copies"] == 0

    def test_temporal_section_for_time_series(self, fred_macro, syn_macro):
        from src.fidelity import fidelity_report

        report = fidelity_report(
            fred_macro, syn_macro.drop(columns=["syn_id"]), dataset_type="time_series"
        )
        assert "temporal" in report
        assert "stationarity" in report["temporal"]
        assert "cointegration" in report["temporal"]
        assert "breaks" in report["temporal"]
        assert "causality" in report["temporal"]

    def test_downstream_section_with_target(self, hmda, syn_hmda):
        from src.fidelity import fidelity_report

        report = fidelity_report(
            hmda,
            syn_hmda.drop(columns=["syn_id"]),
            target_col="loan_amount",
            include_downstream=True,
        )
        assert "downstream" in report
        assert "tstr_score" in report["downstream"]

    def test_no_temporal_for_cross_sectional(self, hmda, syn_hmda):
        from src.fidelity import fidelity_report

        report = fidelity_report(
            hmda, syn_hmda.drop(columns=["syn_id"]), dataset_type="cross_sectional"
        )
        assert "temporal" not in report

    def test_format_report_returns_string(self, hmda, syn_hmda):
        from src.fidelity import fidelity_report, format_report

        report = fidelity_report(hmda, syn_hmda.drop(columns=["syn_id"]))
        text = format_report(report)
        assert isinstance(text, str)
        assert "FIDELITY" in text
