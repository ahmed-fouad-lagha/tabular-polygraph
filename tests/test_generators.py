import pytest
import pandas as pd
import numpy as np


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
    @pytest.mark.parametrize(
        "did",
        [
            "hmda",
            "fdic",
            "credit_risk",
            "edgar",
            "cftc",
            "irs_soi",
            "census_acs",
            "equity_returns",
            "corporate_bonds",
            "insurance_claims",
            "life_insurance",
            "commercial_real_estate",
            "rental_market",
            "retail_transactions",
            "commodity_prices",
        ],
    )
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
        from src.catalog import load_dataset

        gen = GaussianCopulaGenerator()
        gen.fit(load_dataset("hmda").sample(30, random_state=0))
        df = gen.sample(20, seed=1)
        assert len(df) == 20


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Filters
# ═══════════════════════════════════════════════════════════════════════════════


class TestFilters:
    def test_categorical_list_filter(self, hmda):
        from src.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        df = gen.sample(500, filters={"state": ["CA", "TX"]}, seed=1)
        assert set(df["state"].unique()).issubset({"CA", "TX"})

    def test_numeric_min_filter(self, hmda):
        from src.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        df = gen.sample(500, filters={"debt_to_income_min": 50}, seed=1)
        if len(df) > 0:
            assert df["debt_to_income"].min() >= 50

    def test_numeric_max_filter(self, hmda):
        from src.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        df = gen.sample(500, filters={"loan_amount_max": 200000}, seed=1)
        if len(df) > 0:
            assert df["loan_amount"].max() <= 200000

    def test_alias_filter(self, hmda):
        """'dti' should resolve to 'debt_to_income'."""
        from src.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        df = gen.sample(500, filters={"dti_min": 50}, seed=1)
        if len(df) > 0:
            assert df["debt_to_income"].min() >= 50

    def test_exact_categorical_filter(self, edgar):
        from src.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(edgar)
        df = gen.sample(400, filters={"sector": "Technology"}, seed=1)
        if len(df) > 0:
            assert all(df["sector"] == "Technology")

    def test_binary_filter(self, credit_risk):
        from src.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(credit_risk)
        df = gen.sample(400, filters={"default_12m": 1}, seed=1)
        if len(df) > 0:
            assert all(df["default_12m"].astype(str) == "1")

    def test_combined_filters(self, hmda):
        from src.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        df = gen.sample(600, filters={"state": ["CA"], "dti_min": 40}, seed=1)
        if len(df) > 0:
            assert all(df["state"] == "CA")
            assert df["debt_to_income"].min() >= 40


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
                assert syn_macro[col].between(r_min, r_max).all(), (
                    f"{col} values outside plausible range"
                )

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
