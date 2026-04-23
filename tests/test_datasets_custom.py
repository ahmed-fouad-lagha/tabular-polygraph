# ═══════════════════════════════════════════════════════════════════════════════
# 18. Custom file generation (--input)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCustomGeneration:
    def test_python_api_custom_fit(self):
        """Users can fit on any DataFrame and generate synthetic data."""
        import pandas as pd
        from tabular_polygraph.generators import GaussianCopulaGenerator

        # Create a simple custom dataset for generation tests
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
        """Works on arbitrary columns — not just built-in datasets."""
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
        """src list should show all catalog datasets."""
        from tabular_polygraph.catalog import list_datasets, DATASETS

        df = list_datasets()
        assert len(df) == len(DATASETS)
