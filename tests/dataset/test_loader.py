from tabular_polygraph.dataset.downloader import cache_path, is_cached
from tabular_polygraph.dataset.loader import load_cached
from tabular_polygraph.dataset.registry import DATASETS


class TestCacheLogic:
    def test_cache_path_extension(self):
        path = cache_path("test_ds")
        assert path.suffix == ".parquet"
        assert "test_ds" in path.name


class TestSchemaIntegrity:
    def test_cached_data_columns(self):
        for dataset_id in DATASETS:
            if is_cached(dataset_id):
                df = load_cached(dataset_id)
                expected = set(DATASETS[dataset_id].get("columns", []))
                drop_cols = set(DATASETS[dataset_id].get("drop_cols", []))
                actual = set(df.columns)
                missing = (expected - drop_cols) - actual
                assert not missing, f"Dataset '{dataset_id}' missing columns: {missing}"
