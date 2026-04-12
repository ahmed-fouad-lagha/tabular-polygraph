"""Dataset metadata and cached loading for downloadable datasets."""

from __future__ import annotations
import pandas as pd


DATASETS: dict[str, dict] = {
    "fred_macro": {
        "name": "FRED Macroeconomic Indicators",
        "vertical": "Macro & Central Bank",
        "source": "Federal Reserve FRED 2000-2023",
        "description": "Monthly panel: GDP, CPI, unemployment, fed funds rate, yield curve, M2, VIX.",
        "columns": [
            "year",
            "gdp_growth_yoy",
            "cpi_yoy",
            "core_cpi_yoy",
            "unemployment_rate",
            "fed_funds_rate",
            "t10y_rate",
            "t2y_rate",
            "yield_curve_spread",
            "m2_growth",
            "housing_starts",
            "industrial_production",
            "consumer_sentiment",
            "oil_price_yoy",
            "vix",
        ],
        "col_count": 15,
        "tags": ["CSV", "Parquet", "JSON"],
        "use_cases": [
            "Macro regime models",
            "Rate forecasting",
            "Recession probability",
        ],
    },
    "bls": {
        "name": "BLS Employment & Wages",
        "vertical": "Macro & Central Bank",
        "source": "Bureau of Labor Statistics QCEW 2022",
        "description": "Quarterly employment and wage data by NAICS industry, ownership and state.",
        "columns": [
            "naics_sector",
            "ownership",
            "state",
            "avg_weekly_wage",
            "total_employment",
            "yoy_employment_change",
            "yoy_wage_change",
            "establishments",
            "quarter",
        ],
        "col_count": 9,
        "tags": ["CSV", "Parquet"],
        "use_cases": [
            "Labour market models",
            "Wage inflation forecasting",
            "Regional economic analysis",
        ],
    },
    "world_bank": {
        "name": "World Bank Development Indicators",
        "vertical": "Macro & Central Bank",
        "source": "World Bank WDI 2022",
        "description": "Cross-country annual panel: GDP per capita, inflation, current account, FDI, debt-to-GDP.",
        "columns": [
            "country_code",
            "income_group",
            "region",
            "year",
            "gdp_per_capita",
            "gdp_growth",
            "inflation",
            "current_account_pct_gdp",
            "fdi_pct_gdp",
            "govt_debt_pct_gdp",
            "population_growth",
            "gini",
        ],
        "col_count": 12,
        "tags": ["CSV", "Parquet", "JSON"],
        "use_cases": [
            "Sovereign risk models",
            "EM macro forecasting",
            "ESG country scoring",
        ],
    },
    "census_acs": {
        "name": "Census ACS Income & Housing",
        "vertical": "Tax & Income",
        "source": "US Census ACS 5-Year 2022",
        "description": "Household income, housing costs, poverty status and demographics by PUMA geography.",
        "columns": [
            "puma",
            "state",
            "household_income",
            "housing_cost",
            "cost_burden_pct",
            "poverty_status",
            "employment_status",
            "household_size",
            "tenure",
            "age_group",
            "education",
        ],
        "col_count": 11,
        "tags": ["GDPR safe", "CSV", "Parquet"],
        "use_cases": [
            "Affordability models",
            "Poverty prediction",
            "Housing demand forecasting",
        ],
    },
}


def list_datasets(vertical: str | None = None) -> pd.DataFrame:
    """Return a DataFrame summarising all available dataset profiles."""
    rows = []
    for key, m in DATASETS.items():
        if vertical and m["vertical"].lower() != vertical.lower():
            continue
        rows.append(
            {
                "id": key,
                "name": m["name"],
                "vertical": m["vertical"],
                "columns": m["col_count"],
                "source": m["source"],
            }
        )
    return pd.DataFrame(rows)


def get_dataset_info(dataset_id: str) -> dict:
    """Return full metadata for a single dataset profile."""
    if dataset_id not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_id}'. Available: {', '.join(sorted(DATASETS))}"
        )
    return DATASETS[dataset_id]


def load_dataset(dataset_id: str, n: int = 2000) -> pd.DataFrame:
    """
    Load real data for a dataset via API downloader.

    All datasets are real, downloadable public data. To use, first download:
        from src.catalog.downloader import download
        download("fred_macro")   # requires FRED_API_KEY env var
        download("bls")
        download("world_bank")
        download("census_acs")

    Args:
        dataset_id: One of ["fred_macro", "bls", "world_bank", "census_acs"]
        n: Max number of records to return (default 2000)

    Returns:
        Sampled DataFrame of real data from cache or download
    """
    if dataset_id not in DATASETS:
        available = ", ".join(sorted(DATASETS.keys()))
        raise ValueError(
            f"Unknown dataset '{dataset_id}'.\n"
            f"Available real datasets: {available}\n"
            f"Download real data: from src.catalog.downloader import download"
        )

    # Always load from cache if available
    from .downloader import load_cached  # type: ignore

    cached = load_cached(dataset_id)
    if cached is not None and len(cached) >= 100:
        return cached.sample(min(n, len(cached)), random_state=42).reset_index(
            drop=True
        )
    else:
        raise ValueError(
            f"Dataset '{dataset_id}' not cached. Download real data first: python main.py download {dataset_id}"
        )
