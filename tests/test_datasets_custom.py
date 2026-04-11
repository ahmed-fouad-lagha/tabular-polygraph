import pytest


class TestNewDatasets:
    @pytest.mark.parametrize(
        "did,expected_col",
        [
            ("equity_returns", "daily_return"),
            ("corporate_bonds", "credit_spread"),
            ("insurance_claims", "paid_losses"),
            ("life_insurance", "mortality_rate"),
            ("commercial_real_estate", "cap_rate"),
            ("rental_market", "asking_rent"),
            ("retail_transactions", "fraud_flag"),
            ("commodity_prices", "daily_return"),
        ],
    )
    def test_new_dataset_generates(self, did, expected_col, all_seeds):
        from src.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(all_seeds[did])
        df = gen.generate(100, seed=42)
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
        ig_spread = df[df["credit_rating"].isin(["AAA", "AA", "A", "BBB"])][
            "credit_spread"
        ].mean()
        hy_spread = df[df["credit_rating"].isin(["BB", "B", "CCC"])][
            "credit_spread"
        ].mean()
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
        old = df[df["age_at_issue"] > 60]["mortality_rate"].mean()
        assert old > young, "Mortality should increase with age"

    def test_commodity_prices_energy_more_volatile(self, all_seeds):
        """Energy commodities should have higher return volatility than metals."""
        df = all_seeds["commodity_prices"]
        energy_vol = df[df["sector"] == "Energy"]["daily_return"].std()
        metals_vol = df[df["sector"] == "Metals"]["daily_return"].std()
        assert energy_vol > metals_vol, "Energy should be more volatile than metals"

    def test_insurance_claims_large_loss_flag(self, all_seeds):
        """Large loss flag should mark top 5% of claims."""
        df = all_seeds["insurance_claims"]
        large_loss_paid = df[df["large_loss_flag"] == 1]["paid_losses"].mean()
        normal_loss_paid = df[df["large_loss_flag"] == 0]["paid_losses"].mean()
        assert large_loss_paid > normal_loss_paid

    def test_vertical_counts(self):
        from src.catalog import list_datasets

        df = list_datasets()
        verticals = df["vertical"].value_counts().to_dict()
        assert verticals.get("Insurance", 0) == 2
        assert verticals.get("Real Estate", 0) == 2
        assert verticals.get("Retail Banking", 0) == 1
        assert verticals.get("Commodities", 0) == 1
        assert verticals.get("Capital Markets", 0) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# 18. Custom file generation (--input)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCustomGeneration:
    def test_python_api_custom_fit(self, hmda):
        """Users can fit on any DataFrame and generate synthetic data."""
        from src.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        syn = gen.generate(100, seed=1)
        assert len(syn) == 100
        assert set(syn.columns) - {"syn_id"} == set(hmda.columns)

    def test_python_api_custom_columns(self):
        """Works on arbitrary columns — not just built-in datasets."""
        import pandas as pd
        from src.generators import GaussianCopulaGenerator

        custom = pd.DataFrame(
            {
                "revenue": [1e6 * (1 + i * 0.1) for i in range(200)],
                "growth_rate": [0.05 + i * 0.001 for i in range(200)],
                "market": (["US", "EU", "APAC"] * 67)[:200],
                "profitable": ([1] * 150 + [0] * 50),
            }
        )
        gen = GaussianCopulaGenerator()
        gen.fit(custom)
        syn = gen.generate(500, seed=42)
        assert len(syn) == 500
        assert "revenue" in syn.columns
        assert "market" in syn.columns

    def test_cli_list_shows_all_datasets(self):
        """src list should show all catalog datasets."""
        from src.catalog import list_datasets, DATASETS

        df = list_datasets()
        assert len(df) == len(DATASETS)
