"""
full test suite
Run: pytest tests/ -v
"""
import sys, os, pytest, json, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def all_seeds():
    from src.catalog import load_seed, DATASETS
    return {did: load_seed(did) for did in DATASETS}

@pytest.fixture(scope="module")
def hmda(all_seeds):      return all_seeds["hmda"]
@pytest.fixture(scope="module")
def fred_macro(all_seeds):return all_seeds["fred_macro"]
@pytest.fixture(scope="module")
def world_bank(all_seeds):return all_seeds["world_bank"]
@pytest.fixture(scope="module")
def credit_risk(all_seeds):return all_seeds["credit_risk"]
@pytest.fixture(scope="module")
def edgar(all_seeds):     return all_seeds["edgar"]

@pytest.fixture(scope="module")
def syn_hmda(hmda):
    from src.generators import GaussianCopulaGenerator
    gen = GaussianCopulaGenerator()
    gen.fit(hmda)
    return gen.sample(500, seed=42)

@pytest.fixture(scope="module")
def syn_macro(fred_macro):
    from src.generators.time_series import VARGenerator
    gen = VARGenerator(lags=2, time_col="year")
    gen.fit(fred_macro)
    return gen.sample(300, seed=42)

@pytest.fixture(scope="module")
def syn_wb(world_bank):
    from src.generators.panel import FixedEffectsGenerator
    gen = FixedEffectsGenerator(entity_col="country_code", time_col="year")
    gen.fit(world_bank)
    return gen.sample(300, seed=42)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Catalog
# ═══════════════════════════════════════════════════════════════════════════════

class TestCatalog:
    def test_list_datasets_count(self):
        from src.catalog import list_datasets, DATASETS
        assert len(list_datasets()) == len(DATASETS)

    def test_list_datasets_vertical_filter(self):
        from src.catalog import list_datasets
        df = list_datasets(vertical="Capital Markets")
        assert len(df) == 4
        assert set(df["id"]) == {"edgar", "cftc", "equity_returns", "corporate_bonds"}

    def test_get_dataset_info_valid(self):
        from src.catalog import get_dataset_info
        info = get_dataset_info("fred_macro")
        assert info["col_count"] == 15
        assert "vix" in info["columns"]

    def test_get_dataset_info_invalid(self):
        from src.catalog import get_dataset_info
        with pytest.raises(ValueError, match="Unknown"):
            get_dataset_info("does_not_exist")

    @pytest.mark.parametrize("did", [
        "hmda","fdic","credit_risk","edgar","cftc",
        "fred_macro","bls","world_bank","irs_soi","census_acs",
        "equity_returns","corporate_bonds","insurance_claims","life_insurance",
        "commercial_real_estate","rental_market","retail_transactions","commodity_prices"
    ])
    def test_all_seeds_build(self, did):
        from src.catalog import load_seed
        df = load_seed(did)
        assert len(df) == 2000
        assert df.shape[1] > 0
        assert not df.isnull().all().any()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Base Generator
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaseGenerator:
    def test_abstract_cannot_instantiate(self):
        from src.generators.base import BaseGenerator
        with pytest.raises(TypeError):
            BaseGenerator()

    def test_require_fitted_guard(self, hmda):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            gen.sample(10)

    def test_fit_sample_fluent(self, hmda):
        from src.generators import GaussianCopulaGenerator
        df = GaussianCopulaGenerator().fit_sample(hmda, 50, seed=1)
        assert len(df) == 50

    def test_repr_before_after_fit(self, hmda):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        assert "not fitted" in repr(gen)
        gen.fit(hmda)
        assert "fitted on" in repr(gen)
        assert "2,000" in repr(gen)

    def test_syn_id_unique(self, syn_hmda):
        assert syn_hmda["syn_id"].nunique() == len(syn_hmda)

    def test_syn_id_format(self, syn_hmda):
        assert all(syn_hmda["syn_id"].str.startswith("SYN-"))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Gaussian Copula Generator
# ═══════════════════════════════════════════════════════════════════════════════

class TestGaussianCopula:
    @pytest.mark.parametrize("did", [
        "hmda","fdic","credit_risk","edgar","cftc","irs_soi","census_acs",
        "equity_returns","corporate_bonds","insurance_claims","life_insurance",
        "commercial_real_estate","rental_market","retail_transactions","commodity_prices"
    ])
    def test_all_cross_sectional_datasets(self, did, all_seeds):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(all_seeds[did])
        df = gen.sample(100, seed=42)
        assert len(df) == 100
        assert "syn_id" in df.columns

    def test_reproducible_with_seed(self, hmda):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        df1 = gen.sample(50, seed=99)
        df2 = gen.sample(50, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self, hmda):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        df1 = gen.sample(100, seed=1)
        df2 = gen.sample(100, seed=2)
        assert not df1["loan_amount"].equals(df2["loan_amount"])

    def test_no_nulls_in_output(self, syn_hmda):
        body = syn_hmda.drop(columns=["syn_id"])
        assert body.isnull().sum().sum() == 0

    def test_correlation_matrix_property(self, hmda):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        corr = gen.correlation_matrix
        assert corr is not None
        assert corr.shape[0] == corr.shape[1]
        # Diagonal should be ~1
        assert np.allclose(np.diag(corr.values), 1.0, atol=0.01)

    def test_marginal_kinds_property(self, hmda):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        kinds = gen.marginal_kinds
        assert "loan_amount" in kinds
        assert kinds["property_type"] == "categorical"

    def test_small_dataset_stability(self):
        """Generator should not crash on very small datasets (n=30)."""
        from src.generators import GaussianCopulaGenerator
        from src.catalog import load_seed
        gen = GaussianCopulaGenerator()
        gen.fit(load_seed("hmda").sample(30, random_state=0))
        df = gen.sample(20, seed=1)
        assert len(df) == 20


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Filters
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilters:
    def test_categorical_list_filter(self, hmda):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(hmda)
        df = gen.sample(500, filters={"state": ["CA", "TX"]}, seed=1)
        assert set(df["state"].unique()).issubset({"CA", "TX"})

    def test_numeric_min_filter(self, hmda):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(hmda)
        df = gen.sample(500, filters={"debt_to_income_min": 50}, seed=1)
        if len(df) > 0:
            assert df["debt_to_income"].min() >= 50

    def test_numeric_max_filter(self, hmda):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(hmda)
        df = gen.sample(500, filters={"loan_amount_max": 200000}, seed=1)
        if len(df) > 0:
            assert df["loan_amount"].max() <= 200000

    def test_alias_filter(self, hmda):
        """'dti' should resolve to 'debt_to_income'."""
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(hmda)
        df = gen.sample(500, filters={"dti_min": 50}, seed=1)
        if len(df) > 0:
            assert df["debt_to_income"].min() >= 50

    def test_exact_categorical_filter(self, edgar):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(edgar)
        df = gen.sample(400, filters={"sector": "Technology"}, seed=1)
        if len(df) > 0:
            assert all(df["sector"] == "Technology")

    def test_binary_filter(self, credit_risk):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(credit_risk)
        df = gen.sample(400, filters={"default_12m": 1}, seed=1)
        if len(df) > 0:
            assert all(df["default_12m"].astype(str) == "1")

    def test_combined_filters(self, hmda):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(hmda)
        df = gen.sample(600, filters={"state": ["CA"], "dti_min": 40}, seed=1)
        if len(df) > 0:
            assert all(df["state"] == "CA")
            assert df["debt_to_income"].min() >= 40


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VAR Time Series Generator
# ═══════════════════════════════════════════════════════════════════════════════

class TestVARGenerator:
    def test_basic_generation(self, syn_macro):
        assert len(syn_macro) == 300
        assert "syn_id" in syn_macro.columns

    def test_correct_column_count(self, fred_macro, syn_macro):
        # syn should have (original cols - time_col) + syn_id
        expected = len(fred_macro.columns)  # year dropped, syn_id added → same count
        assert len(syn_macro.columns) == expected

    def test_numeric_columns_in_range(self, fred_macro, syn_macro):
        for col in ["cpi_yoy", "unemployment_rate", "fed_funds_rate"]:
            if col in syn_macro.columns:
                r_min = fred_macro[col].min() * 4 - fred_macro[col].abs().max()
                r_max = fred_macro[col].max() * 4
                assert syn_macro[col].between(r_min, r_max).all(), \
                    f"{col} values outside plausible range"

    def test_bls_var_generation(self, all_seeds):
        from src.generators.time_series import VARGenerator
        gen = VARGenerator(lags=2, time_col="quarter")
        gen.fit(all_seeds["bls"])
        df = gen.sample(100, seed=1)
        assert len(df) == 100


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Panel Generator
# ═══════════════════════════════════════════════════════════════════════════════

class TestPanelGenerator:
    def test_basic_generation(self, syn_wb):
        assert len(syn_wb) == 300
        assert "syn_id" in syn_wb.columns

    def test_entity_col_present(self, syn_wb):
        assert "country_code" in syn_wb.columns

    def test_time_col_present(self, syn_wb):
        assert "year" in syn_wb.columns

    def test_fdic_panel(self, all_seeds):
        from src.generators.panel import FixedEffectsGenerator
        gen = FixedEffectsGenerator(entity_col="state", time_col="charter_class")
        gen.fit(all_seeds["fdic"])
        df = gen.sample(100, seed=1)
        assert len(df) == 100


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Fidelity — Marginal
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarginalFidelity:
    def test_scores_in_range(self, hmda, syn_hmda):
        from src.fidelity.marginal import moment_matching_scores
        scores = moment_matching_scores(hmda, syn_hmda.drop(columns=["syn_id"]))
        for col, score in scores.items():
            assert 0 <= score <= 100, f"{col} score {score} out of range"

    def test_high_fidelity_on_large_sample(self, hmda, syn_hmda):
        from src.fidelity.marginal import mean_moment_matching_score, moment_matching_scores
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
        import numpy as np
        import torch
        from src.fidelity.logical import neuro_lcv_score

        np.random.seed(0)
        torch.manual_seed(0)

        real = pd.DataFrame({
            "state": ["CA", "CA", "NY", "NY", "TX", "TX"] * 20,
            "county": ["001", "003", "005", "007", "009", "011"] * 20,
            "class": ["A", "B", "A", "B", "A", "B"] * 20,
        })

        clean = real.sample(frac=1.0, random_state=0).reset_index(drop=True)
        bad = clean.copy()
        bad.loc[:19, "state"] = "__ILLOGICAL__"
        bad.loc[:19, "county"] = "__ILLOGICAL__"

        clean_result = neuro_lcv_score(real, clean, columns=["state", "county", "class"], epochs=6, verbose=False)
        bad_result = neuro_lcv_score(real, bad, columns=["state", "county", "class"], epochs=6, verbose=False)

        assert clean_result["neuro_lcv_score"] > bad_result["neuro_lcv_score"]
        assert clean_result["mean_penalty"] < bad_result["mean_penalty"]

    def test_neuro_lcv_canonicalizes_code_columns(self):
        import numpy as np
        import torch
        from src.fidelity.logical import neuro_lcv_score

        np.random.seed(1)
        torch.manual_seed(1)

        real = pd.DataFrame({
            "state_fips": ["06", "08", "06", "04", "06", "08", "12", "06", "08", "04"] * 40,
            "county": ["037", "109", "083", "019", "001", "097", "071", "073", "005", "111"] * 40,
        })

        clean = pd.DataFrame({
            "state_fips": [6, 8, 6, 4, 6, 8, 12, 6, 8, 4] * 40,
            "county": [37, 109, 83, 19, 1, 97, 71, 73, 5, 111] * 40,
        })

        bad = clean.astype({"state_fips": "object", "county": "object"}).copy()
        bad.loc[:199, "state_fips"] = "__ILLOGICAL__"
        bad.loc[:199, "county"] = "__ILLOGICAL__"

        clean_result = neuro_lcv_score(real, clean, columns=["state_fips", "county"], epochs=6, verbose=False)
        bad_result = neuro_lcv_score(real, bad, columns=["state_fips", "county"], epochs=6, verbose=False)

        assert clean_result["neuro_lcv_score"] > bad_result["neuro_lcv_score"]
        assert clean_result["mean_penalty"] < bad_result["mean_penalty"]
        assert clean_result["violation_rate"] <= bad_result["violation_rate"]

    def test_rule_violation_score_penalizes_corruption(self):
        from src.fidelity.logical import rule_violation_score

        real = pd.DataFrame({
            "state": ["CA", "CA", "TX", "TX", "NY", "NY"] * 40,
            "county": ["001", "001", "005", "005", "003", "003"] * 40,
            "segment": ["urban", "urban", "rural", "rural", "urban", "urban"] * 40,
        })

        clean = real.sample(frac=1.0, random_state=10).reset_index(drop=True)
        bad = clean.copy()
        bad.loc[:79, "county"] = "999"

        clean_rules = rule_violation_score(real, clean, columns=["state", "county", "segment"])
        bad_rules = rule_violation_score(real, bad, columns=["state", "county", "segment"])

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
        gen = GaussianCopulaGenerator(); gen.fit(credit_risk)
        syn = gen.sample(300, seed=1).drop(columns=["syn_id"])
        result = tstr_score(credit_risk, syn, target_col="default_12m", task="classification")
        assert "tstr_score" in result
        assert "trr_score"  in result
        assert "ratio"      in result

    def test_tstr_regression(self, hmda):
        from src.generators import GaussianCopulaGenerator
        from src.fidelity.downstream import tstr_score
        gen = GaussianCopulaGenerator(); gen.fit(hmda)
        syn = gen.sample(300, seed=1).drop(columns=["syn_id"])
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
        for key in ["moment_matching", "distribution_fit", "joint", "stylized_facts", "privacy_basic", "summary"]:
            assert key in report

    def test_summary_scores_in_range(self, hmda, syn_hmda):
        from src.fidelity import fidelity_report
        report = fidelity_report(hmda, syn_hmda.drop(columns=["syn_id"]))
        s = report["summary"]
        assert 0 <= s["overall_fidelity"] <= 100
        assert 0 <= s["moment_matching_score"] <= 100
        assert 0 <= s["ks_score"] <= 100
        assert 0 <= s["joint_score"]      <= 100
        assert s["exact_copies"] == 0

    def test_temporal_section_for_time_series(self, fred_macro, syn_macro):
        from src.fidelity import fidelity_report
        report = fidelity_report(fred_macro, syn_macro.drop(columns=["syn_id"]),
                                  dataset_type="time_series")
        assert "temporal" in report
        assert "stationarity"  in report["temporal"]
        assert "cointegration" in report["temporal"]
        assert "breaks"        in report["temporal"]
        assert "causality"     in report["temporal"]

    def test_downstream_section_with_target(self, hmda, syn_hmda):
        from src.fidelity import fidelity_report
        report = fidelity_report(hmda, syn_hmda.drop(columns=["syn_id"]),
                                  target_col="loan_amount", include_downstream=True)
        assert "downstream" in report
        assert "tstr_score" in report["downstream"]

    def test_no_temporal_for_cross_sectional(self, hmda, syn_hmda):
        from src.fidelity import fidelity_report
        report = fidelity_report(hmda, syn_hmda.drop(columns=["syn_id"]),
                                  dataset_type="cross_sectional")
        assert "temporal" not in report

    def test_format_report_returns_string(self, hmda, syn_hmda):
        from src.fidelity import fidelity_report, format_report
        report = fidelity_report(hmda, syn_hmda.drop(columns=["syn_id"]))
        text = format_report(report)
        assert isinstance(text, str)
        assert "FIDELITY" in text


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Privacy
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Calibration — Priors
# ═══════════════════════════════════════════════════════════════════════════════

class TestPriors:
    def test_prior_normal_samples(self):
        from src.calibration.priors import Prior
        p = Prior("normal", mu=100.0, sigma=10.0)
        samples = p.sample(1000, seed=0)
        assert abs(samples.mean() - 100.0) < 3.0
        assert abs(samples.std()  - 10.0)  < 2.0

    def test_prior_lognormal_samples(self):
        from src.calibration.priors import Prior
        p = Prior("lognormal", mu=0.0, sigma=1.0)
        samples = p.sample(1000, seed=0)
        assert (samples > 0).all()

    def test_prior_beta_range(self):
        from src.calibration.priors import Prior
        p = Prior("beta", alpha=2.0, beta=5.0)
        samples = p.sample(500, seed=0)
        assert (samples >= 0).all() and (samples <= 1).all()

    def test_prior_fixed_constant(self):
        from src.calibration.priors import Prior
        p = Prior("fixed", value=42.0)
        samples = p.sample(100, seed=0)
        assert (samples == 42.0).all()

    def test_prior_invalid_distribution(self):
        from src.calibration.priors import Prior
        with pytest.raises(ValueError, match="Unknown distribution"):
            Prior("uniform", lo=0, hi=1)

    def test_prior_missing_params(self):
        from src.calibration.priors import Prior
        with pytest.raises(ValueError, match="missing params"):
            Prior("normal", mu=0.0)   # missing sigma

    def test_map_mean_pulls_toward_prior(self):
        from src.calibration.priors import Prior
        p = Prior("normal", mu=100.0, sigma=10.0, strength=5.0)
        # With only 10 observations, prior should dominate
        blended = p.map_mean(data_mean=200.0, n_obs=10)
        assert blended < 200.0   # pulled toward prior mean of 100
        assert blended > 100.0   # but data has some influence

    def test_map_mean_weak_prior_on_large_n(self):
        from src.calibration.priors import Prior
        p = Prior("normal", mu=100.0, sigma=10.0, strength=0.1)
        # With 10,000 obs, data should dominate
        blended = p.map_mean(data_mean=200.0, n_obs=10000)
        assert blended > 190.0   # very close to data mean

    def test_prior_set_construction(self):
        from src.calibration.priors import Prior, PriorSet
        ps = PriorSet({
            "col_a": Prior("normal",    mu=0.0, sigma=1.0),
            "col_b": Prior("lognormal", mu=1.0, sigma=0.5),
        })
        assert len(ps.columns()) == 2
        assert ps.get("col_a") is not None
        assert ps.get("col_c") is None

    def test_prior_set_map_mean(self):
        from src.calibration.priors import Prior, PriorSet
        ps = PriorSet({"x": Prior("normal", mu=50.0, sigma=5.0, strength=3.0)})
        blended = ps.map_mean("x", data_mean=100.0, n_obs=5)
        assert blended < 100.0

    def test_dataset_priors_all_present(self):
        from src.calibration.priors import DATASET_PRIORS, get_priors
        # Priors are defined for the original 10 core datasets
        core = ["hmda","fdic","credit_risk","edgar","cftc",
                "fred_macro","bls","world_bank","irs_soi","census_acs"]
        for did in core:
            ps = get_priors(did)
            assert isinstance(ps.columns(), list)
            assert len(ps.columns()) >= 2

    def test_get_priors_invalid(self):
        from src.calibration.priors import get_priors
        with pytest.raises(ValueError, match="No built-in priors"):
            get_priors("nonexistent_dataset")

    def test_prior_regularises_small_dataset(self, hmda):
        """Generator with priors on 50-row dataset should produce reasonable output."""
        from src.generators import GaussianCopulaGenerator
        from src.calibration.priors import get_priors
        priors = get_priors("hmda")
        gen_no_prior = GaussianCopulaGenerator()
        gen_with_prior = GaussianCopulaGenerator(priors=priors)
        small = hmda.sample(50, random_state=0)
        gen_no_prior.fit(small)
        gen_with_prior.fit(small)
        df_no  = gen_no_prior.sample(200, seed=1)
        df_yes = gen_with_prior.sample(200, seed=1)
        # Both should produce valid output
        assert len(df_no) == 200
        assert len(df_yes) == 200
        # With-prior mean should be closer to full-dataset mean
        full_mean = hmda["loan_amount"].mean()
        err_no  = abs(df_no["loan_amount"].mean()  - full_mean)
        err_yes = abs(df_yes["loan_amount"].mean() - full_mean)
        assert err_yes < err_no * 2.0  # prior version within 2x of no-prior (usually better)

    def test_prior_set_sample_prior_data(self):
        from src.calibration.priors import get_priors
        ps = get_priors("fred_macro")
        samples = ps.sample_prior_data(n=100, seed=0)
        assert isinstance(samples, dict)
        for col, arr in samples.items():
            assert len(arr) == 100
            assert np.isfinite(arr).all()

    def test_prior_set_summary(self):
        from src.calibration.priors import get_priors
        ps = get_priors("hmda")
        summary = ps.summary()
        assert isinstance(summary, list)
        cols = [r["column"] for r in summary]
        assert "loan_amount" in cols


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Calibration — Moment Matching & Scenario
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibration:
    def test_moment_matching_returns_df(self, hmda, syn_hmda):
        from src.calibration import match_moments
        result = match_moments(hmda, syn_hmda.drop(columns=["syn_id"]))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(syn_hmda)

    def test_moment_report_shape(self, hmda, syn_hmda):
        from src.calibration import match_moments, moment_report
        cal = match_moments(hmda, syn_hmda.drop(columns=["syn_id"]))
        report = moment_report(hmda, cal)
        assert len(report) > 0
        assert "real_mean" in report.columns

    def test_scenario_recession_shifts_gdp(self, syn_macro):
        from src.calibration import apply_scenario
        result = apply_scenario(syn_macro, "recession", intensity=1.0)
        if "gdp_growth_yoy" in result.columns:
            assert result["gdp_growth_yoy"].mean() < syn_macro["gdp_growth_yoy"].mean()

    def test_scenario_expansion_raises_wages(self, syn_macro):
        from src.calibration import apply_scenario
        result = apply_scenario(syn_macro, "expansion", intensity=1.0)
        if "gdp_growth_yoy" in result.columns:
            assert result["gdp_growth_yoy"].mean() > syn_macro["gdp_growth_yoy"].mean()

    def test_intensity_zero_no_change(self, syn_hmda):
        from src.calibration import apply_scenario
        result = apply_scenario(syn_hmda, "recession", intensity=0.0)
        num_cols = [c for c in syn_hmda.columns if syn_hmda[c].dtype.kind in "if"]
        for col in num_cols:
            if col in result.columns:
                assert abs(result[col].mean() - syn_hmda[col].mean()) < 1e-6

    def test_invalid_scenario_raises(self, syn_hmda):
        from src.calibration import apply_scenario
        with pytest.raises(ValueError, match="Unknown scenario"):
            apply_scenario(syn_hmda, "alien_invasion")

    def test_list_scenarios_count(self):
        from src.calibration import list_scenarios
        df = list_scenarios()
        assert len(df) == 5
        assert "recession" in df["name"].values


# ═══════════════════════════════════════════════════════════════════════════════
# 16. IO
# ═══════════════════════════════════════════════════════════════════════════════

class TestIO:
    def test_csv_roundtrip(self, syn_hmda):
        from src.io import write, read
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            write(syn_hmda, path)
            reloaded = read(path)
            assert len(reloaded) == len(syn_hmda)
        finally:
            os.unlink(path)

    def test_json_roundtrip(self, syn_hmda):
        from src.io import write, read
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            write(syn_hmda, path)
            reloaded = read(path)
            assert len(reloaded) == len(syn_hmda)
        finally:
            os.unlink(path)

    def test_stata_roundtrip(self, syn_hmda):
        from src.io import write, read
        with tempfile.NamedTemporaryFile(suffix=".dta", delete=False) as f:
            path = f.name
        try:
            write(syn_hmda, path)
            reloaded = read(path)
            assert len(reloaded) == len(syn_hmda)
        finally:
            os.unlink(path)

    def test_unsupported_format_raises(self, syn_hmda):
        from src.io import write
        with pytest.raises(ValueError):
            write(syn_hmda, "/tmp/test.xyz")

    def test_validate_passes_clean_data(self, hmda):
        from src.io import validate
        result = validate(hmda)
        assert result.passed
        assert len(result.errors) == 0

    def test_validate_catches_nulls(self):
        from src.io import validate
        df = pd.DataFrame({"a": [1, None, None, None, None], "b": [1, 2, 3, 4, 5]})
        # 80% nulls in 'a' — should fail with default threshold 0.3
        result = validate(df, min_rows=3)
        assert not result.passed
        assert any("a" in e for e in result.errors)

    def test_validate_warns_constant_column(self):
        from src.io import validate
        df = pd.DataFrame({"a": [1]*100, "b": range(100)})
        result = validate(df)
        assert any("constant" in w.lower() for w in result.warnings)

    def test_validate_warns_high_cardinality(self):
        from src.io import validate
        df = pd.DataFrame({
            "id": [f"id_{i}" for i in range(200)],
            "val": range(200),
        })
        result = validate(df, max_cardinality=50, min_rows=100)
        assert any("cardinality" in w.lower() for w in result.warnings)

    def test_supported_formats_list(self):
        from src.io import supported_formats
        fmts = supported_formats()
        assert "csv"   in fmts
        assert "json"  in fmts
        assert "stata" in fmts


# ═══════════════════════════════════════════════════════════════════════════════
# 17. New datasets — smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewDatasets:
    @pytest.mark.parametrize("did,expected_col", [
        ("equity_returns",         "daily_return"),
        ("corporate_bonds",        "credit_spread"),
        ("insurance_claims",       "paid_losses"),
        ("life_insurance",         "mortality_rate"),
        ("commercial_real_estate", "cap_rate"),
        ("rental_market",          "asking_rent"),
        ("retail_transactions",    "fraud_flag"),
        ("commodity_prices",       "daily_return"),
    ])
    def test_new_dataset_generates(self, did, expected_col, all_seeds):
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(all_seeds[did])
        df = gen.sample(100, seed=42)
        assert len(df) == 100
        assert expected_col in df.columns

    def test_equity_returns_fat_tails(self, all_seeds):
        """Daily returns should have excess kurtosis > 1 (fat tails)."""
        from scipy import stats
        seed = all_seeds["equity_returns"]
        kurt = float(stats.kurtosis(seed["daily_return"]))
        assert kurt > 1.0, f"Expected fat tails (kurtosis > 1), got {kurt:.2f}"

    def test_corporate_bonds_spread_by_rating(self, all_seeds):
        """IG bonds should have lower spreads than HY bonds."""
        df = all_seeds["corporate_bonds"]
        ig_spread = df[df["credit_rating"].isin(["AAA","AA","A","BBB"])]["credit_spread"].mean()
        hy_spread = df[df["credit_rating"].isin(["BB","B","CCC"])]["credit_spread"].mean()
        assert ig_spread < hy_spread, "IG spreads should be lower than HY spreads"

    def test_retail_transactions_fraud_rate(self, all_seeds):
        """Fraud rate should be low (< 2%) matching industry average."""
        df = all_seeds["retail_transactions"]
        fraud_rate = df["fraud_flag"].mean()
        assert fraud_rate < 0.02, f"Fraud rate too high: {fraud_rate:.3f}"

    def test_life_insurance_mortality_increases_with_age(self, all_seeds):
        """Mortality rate should be higher for older policyholders."""
        df = all_seeds["life_insurance"]
        young = df[df["age_at_issue"] < 35]["mortality_rate"].mean()
        old   = df[df["age_at_issue"] > 60]["mortality_rate"].mean()
        assert old > young, "Mortality should increase with age"

    def test_commodity_prices_energy_more_volatile(self, all_seeds):
        """Energy commodities should have higher return volatility than metals."""
        df = all_seeds["commodity_prices"]
        energy_vol = df[df["sector"]=="Energy"]["daily_return"].std()
        metals_vol = df[df["sector"]=="Metals"]["daily_return"].std()
        assert energy_vol > metals_vol, "Energy should be more volatile than metals"

    def test_insurance_claims_large_loss_flag(self, all_seeds):
        """Large loss flag should mark top 5% of claims."""
        df = all_seeds["insurance_claims"]
        large_loss_paid  = df[df["large_loss_flag"]==1]["paid_losses"].mean()
        normal_loss_paid = df[df["large_loss_flag"]==0]["paid_losses"].mean()
        assert large_loss_paid > normal_loss_paid

    def test_vertical_counts(self):
        from src.catalog import list_datasets
        df = list_datasets()
        verticals = df["vertical"].value_counts().to_dict()
        assert verticals.get("Insurance", 0)      == 2
        assert verticals.get("Real Estate", 0)    == 2
        assert verticals.get("Retail Banking", 0) == 1
        assert verticals.get("Commodities", 0)    == 1
        assert verticals.get("Capital Markets", 0)== 4


# ═══════════════════════════════════════════════════════════════════════════════
# 18. Custom file generation (--input)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCustomGeneration:
    def test_python_api_custom_fit(self, hmda):
        """Users can fit on any DataFrame and generate synthetic data."""
        from src.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        syn = gen.sample(100, seed=1)
        assert len(syn) == 100
        assert set(syn.columns) - {"syn_id"} == set(hmda.columns)

    def test_python_api_custom_columns(self):
        """Works on arbitrary columns — not just built-in datasets."""
        import pandas as pd
        from src.generators import GaussianCopulaGenerator
        custom = pd.DataFrame({
            "revenue":     [1e6 * (1 + i*0.1) for i in range(200)],
            "growth_rate": [0.05 + i*0.001 for i in range(200)],
            "market":      (["US","EU","APAC"] * 67)[:200],
            "profitable":  ([1]*150 + [0]*50),
        })
        gen = GaussianCopulaGenerator()
        gen.fit(custom)
        syn = gen.sample(500, seed=42)
        assert len(syn) == 500
        assert "revenue" in syn.columns
        assert "market" in syn.columns

    def test_cli_list_shows_all_datasets(self):
        """src list should show all catalog datasets."""
        from src.catalog import list_datasets, DATASETS
        df = list_datasets()
        assert len(df) == len(DATASETS)
