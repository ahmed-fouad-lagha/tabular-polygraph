from tabular_polygraph.dataset.downloader import (
    cache_path,
    is_cached,
)
from tabular_polygraph.dataset.loader import load_cached
from tabular_polygraph.dataset.registry import DATASETS


class TestDatasetRegistry:
    def test_registry_completeness(self):
        """Ensure every registered dataset has mandatory metadata."""
        for _did, info in DATASETS.items():
            assert "name" in info
            assert "source" in info
            assert "url" in info
            assert "columns" in info


class TestCacheLogic:
    def test_cache_path_extension(self):
        """Verify we are using parquet for research-grade storage."""
        path = cache_path("test_ds")
        assert path.suffix == ".parquet"
        assert "test_ds" in path.name


class TestSchemaIntegrity:
    def test_cached_data_columns(self):
        """Validate that all cached datasets have the expected core columns."""
        for dataset_id in DATASETS:
            if is_cached(dataset_id):
                df = load_cached(dataset_id)
                expected = set(DATASETS[dataset_id].get("columns", []))
                drop_cols = set(DATASETS[dataset_id].get("drop_cols", []))
                actual = set(df.columns)
                missing = (expected - drop_cols) - actual
                assert not missing, f"Dataset '{dataset_id}' missing columns: {missing}"


class TestCustomGeneration:
    def test_python_api_custom_fit(self):
        """Users can fit on any DataFrame and generate synthetic data."""
        import pandas as pd

        from tabular_polygraph.generators import GaussianCopulaGenerator

        custom = pd.DataFrame(
            {
                "loan_amount": [100000 + i * 1000 for i in range(100)],
                "interest_rate": [3.5 + i * 0.01 for i in range(100)],
                "credit_score": [650 + (i % 50) for i in range(100)],
                "loan_type": (["mortgage", "auto", "personal"] * 34)[:100],
            }
        )

        gen = GaussianCopulaGenerator()
        gen.fit(custom)
        syn = gen.generate(100, seed=1)
        assert len(syn) == 100
        assert set(syn.columns) - {"syn_id"} == set(custom.columns)

    def test_python_api_custom_columns(self):
        """Works on arbitrary columns -- not just built-in datasets."""
        import pandas as pd

        from tabular_polygraph.generators import GaussianCopulaGenerator

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
        """The list command should show all datasets."""
        from tabular_polygraph.dataset import list_datasets

        df = list_datasets()
        assert len(df) == len(DATASETS)
