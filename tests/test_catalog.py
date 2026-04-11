import pytest


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

    @pytest.mark.parametrize(
        "did",
        [
            "hmda",
            "fdic",
            "credit_risk",
            "edgar",
            "cftc",
            "fred_macro",
            "bls",
            "world_bank",
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
    def test_all_seeds_build(self, did):
        from src.catalog import load_dataset

        df = load_dataset(did)
        assert len(df) == 2000
        assert df.shape[1] > 0
        assert not df.isnull().all().any()
