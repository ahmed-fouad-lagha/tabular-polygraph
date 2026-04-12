import pytest


class TestCatalog:
    def test_list_datasets_count(self):
        from src.catalog import list_datasets, DATASETS

        assert len(list_datasets()) == len(DATASETS)

    def test_list_datasets_vertical_filter(self):
        from src.catalog import list_datasets

        df = list_datasets(vertical="Macro & Central Bank")
        assert len(df) == 3
        assert set(df["id"]) == {"fred_macro", "bls", "world_bank"}

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
            "fred_macro",
            "bls",
            "world_bank",
            "census_acs",
        ],
    )
    def test_all_seeds_build(self, did, all_seeds):
        df = all_seeds[did]

        assert len(df) == 2000
        assert df.shape[1] > 0
        # Some datasets have missing values in some columns (but not all columns)
        assert df.shape[0] > 0  # Check that data was loaded
