import numpy as np
import pandas as pd

from tabular_polygraph.generators import GaussianCopulaGenerator


class TestMarginalFidelity:
    def test_scores_in_range(self, census_acs):
        from tabular_polygraph.fidelity.marginal import moment_matching_scores

        gen = GaussianCopulaGenerator()
        gen.fit(census_acs)
        syn = gen.generate(300, seed=42).drop(columns=["syn_id"])

        scores = moment_matching_scores(census_acs, syn)
        for _, score in scores.items():
            assert 0 <= score <= 100

    def test_identical_data_scores_100(self, census_acs):
        from tabular_polygraph.fidelity.marginal import moment_matching_scores

        scores = moment_matching_scores(census_acs, census_acs)
        for score in scores.values():
            assert score >= 99.0


class TestLogicalFidelity:
    def test_hif_handles_single_category_feature(self):
        from tabular_polygraph.fidelity.logical import hif_score

        real = pd.DataFrame({"cat": ["A"] * 20})
        syn = pd.DataFrame({"cat": ["A"] * 20})

        result = hif_score(real, syn, verbose=False)

        assert result["hif_score"] == 1.0
        assert result["violation_rate"] == 0.0
        assert result["mean_penalty"] == 0.0

    def test_hif_small_dataset_train_with_verbose(self):
        from tabular_polygraph.fidelity.logical import hif_score

        real = pd.DataFrame(
            {
                "a": ["x", "y"] * 20,
                "b": ["m", "n"] * 20,
            }
        )
        syn = real.copy()

        result = hif_score(real, syn, verbose=True, random_state=42)
        assert 0.0 <= result["hif_score"] <= 1.0

    def test_lse_oracle_trains_and_audits(self):
        from tabular_polygraph.fidelity.logical import LogicalSentinelEnsemble

        # Increase data diversity for better Sentinel training
        real = pd.DataFrame(
            {
                "a": ["X", "Y", "Z", "W"] * 50,
                "b": ["1", "2", "3", "4"] * 50,
                "c": ["M", "N", "O", "P"] * 50,
            }
        )
        syn = real.copy()
        syn.loc[0, "b"] = "99"  # Violation

        oracle = LogicalSentinelEnsemble()
        oracle.fit(real)
        score, penalties, meta = oracle.audit(syn)

        assert len(penalties) == len(syn)
        assert penalties[0] > 0.0
        assert penalties[1] == 0.0

    def test_nic_scorer_manifold_continuity(self):
        from tabular_polygraph.fidelity.logical import NeighborInvariantContinuity

        # Increase feature space for PCA(n_components=32)
        data_dict = {
            f"cat_{i}": [chr(65 + (j % 4)) for j in range(100)] for i in range(40)
        }
        real_cat = pd.DataFrame(data_dict)
        real_num = pd.DataFrame({"val": np.linspace(0, 10, 100)})

        syn_cat = real_cat.iloc[:2].copy()
        syn_num = pd.DataFrame({"val": [5.0, 50.0]})  # 50.0 is an outlier

        scorer = NeighborInvariantContinuity()
        scorer.fit(real_cat, real_num)
        score, penalties = scorer.score(syn_cat, syn_num)

        assert penalties[0] < 0.5
        assert penalties[1] > 0.5

    def test_rule_violation_score_penalizes_corruption(self):
        from tabular_polygraph.fidelity.logical import rule_violation_score

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
        assert clean_rules["total_rule_hits"] < bad_rules["total_rule_hits"]
        assert (
            clean_rules["num_rows_with_violations"]
            < bad_rules["num_rows_with_violations"]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Fidelity — Joint
# ═══════════════════════════════════════════════════════════════════════════════


class TestJointFidelity:
    def test_correlation_score_range(self, census_acs):
        from tabular_polygraph.fidelity.joint import correlation_distance_score

        gen = GaussianCopulaGenerator()
        gen.fit(census_acs)
        syn = gen.generate(300, seed=7).drop(columns=["syn_id"])

        score = correlation_distance_score(census_acs, syn)
        assert 0 <= score <= 100

    def test_pairwise_report_returns_dict(self, census_acs):
        from tabular_polygraph.fidelity.joint import pairwise_correlation_report

        gen = GaussianCopulaGenerator()
        gen.fit(census_acs)
        syn = gen.generate(200, seed=8).drop(columns=["syn_id"])

        result = pairwise_correlation_report(census_acs, syn)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestDownstreamFidelity:
    def test_tstr_scales_each_training_split_independently(self, monkeypatch):
        from tabular_polygraph.fidelity import downstream

        real = pd.DataFrame(
            {
                "feature": np.arange(100, dtype=float),
                "target": (np.arange(100) > 49).astype(int),
            }
        )
        synthetic = real.copy()
        synthetic["feature"] = synthetic["feature"] + 1000.0

        captured_means: list[np.ndarray] = []

        def fake_logreg(X_train, y_train, X_test):
            captured_means.append(X_train.mean(axis=0))
            return np.zeros(len(X_test), dtype=float)

        monkeypatch.setattr(downstream, "_simple_logreg", fake_logreg)

        result = downstream.tstr_score(real, synthetic, target_col="target")

        assert result["task"] == "classification"
        assert len(captured_means) == 2
        for mean_vector in captured_means:
            assert np.allclose(mean_vector, 0.0, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Fidelity — Temporal
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemporalFidelity:
    def test_stationarity_agreement(self, fred_macro, syn_macro):
        from tabular_polygraph.fidelity.temporal.stationarity import stationarity_score

        result = stationarity_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        rate = result["_summary"]["agreement_rate"]
        assert 0 <= rate <= 100

    def test_cointegration_agreement(self, fred_macro, syn_macro):
        from tabular_polygraph.fidelity.temporal.cointegration import (
            cointegration_score,
        )

        result = cointegration_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        assert 0 <= result["_summary"]["agreement_rate"] <= 100

    def test_breaks_score(self, fred_macro, syn_macro):
        from tabular_polygraph.fidelity.temporal.breaks import breaks_score

        result = breaks_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        assert 0 <= result["_summary"]["break_match_rate"] <= 100

    def test_causality_score(self, fred_macro, syn_macro):
        from tabular_polygraph.fidelity.causality import causality_score

        result = causality_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        assert 0 <= result["_summary"]["agreement_rate"] <= 100


class TestFidelityReport:
    def test_cross_sectional_report_keys(self, census_acs):
        from tabular_polygraph.fidelity import fidelity_report

        gen = GaussianCopulaGenerator()
        gen.fit(census_acs)
        syn = gen.generate(300, seed=21).drop(columns=["syn_id"])

        report = fidelity_report(census_acs, syn)
        for key in [
            "moment_matching",
            "distribution_fit",
            "joint",
            "stylized_facts",
            "privacy_basic",
            "summary",
        ]:
            assert key in report

    def test_temporal_section_for_time_series(self, fred_macro, syn_macro):
        from tabular_polygraph.fidelity import fidelity_report

        report = fidelity_report(
            fred_macro, syn_macro.drop(columns=["syn_id"]), dataset_type="time_series"
        )
        assert "temporal" in report
        assert "stationarity" in report["temporal"]
        assert "cointegration" in report["temporal"]
        assert "breaks" in report["temporal"]
        assert "causality" in report["temporal"]

    def test_format_report_returns_string(self, census_acs):
        from tabular_polygraph.fidelity import fidelity_report, format_report

        gen = GaussianCopulaGenerator()
        gen.fit(census_acs)
        syn = gen.generate(150, seed=13).drop(columns=["syn_id"])

        report = fidelity_report(census_acs, syn)
        text = format_report(report)
        assert isinstance(text, str)
        assert "FIDELITY" in text


# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
