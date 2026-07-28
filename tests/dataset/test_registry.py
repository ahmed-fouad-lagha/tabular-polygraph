from tabular_polygraph.dataset.registry import DATASETS


class TestDatasetRegistry:
    def test_registry_completeness(self):
        for _did, info in DATASETS.items():
            assert "name" in info
            assert "source" in info
            assert "url" in info
            assert "columns" in info
