import pytest

from tabular_polygraph.dataset.downloader import (
    DOWNLOADERS,
    cache_path,
    is_cached,
    load_cached,
)


class TestCatalogRegistry:
    def test_registry_completeness(self):
        """Ensure every registered downloader has mandatory metadata."""
        for _did, info in DOWNLOADERS.items():
            assert "name" in info
            assert "source" in info
            assert "method" in info
            # url is optional but good to have
            # assert "url" in info


class TestCacheLogic:
    def test_cache_path_extension(self):
        """Verify we are using parquet for research-grade storage."""
        path = cache_path("test_ds")
        assert path.suffix == ".parquet"
        assert "test_ds" in path.name


class TestDataHardening:
    @pytest.mark.skipif(not is_cached("bls"), reason="BLS not cached")
    def test_bls_quarter_integrity(self):
        """Check for 'Annual' (A) quarter corruption."""
        df = load_cached("bls")
        if df is not None:
            quarters = df["quarter"].unique()
            assert all(q in [1, 2, 3, 4] for q in quarters)


class TestSchemaIntegrity:
    def test_cached_data_columns(self):
        """
        Validate that all cached datasets have the expected core columns.
        This provides the guardrail previously in _validate_df.
        """
        from tabular_polygraph.dataset.loader import DATASETS

        for did in DATASETS:
            if is_cached(did):
                df = load_cached(did)
                expected = set(DATASETS[did].get("columns", []))
                actual = set(df.columns)
                missing = expected - actual
                assert not missing, f"Dataset '{did}' missing columns: {missing}"
