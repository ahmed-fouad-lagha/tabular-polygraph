"""Dataset metadata and cached loading for downloadable datasets."""

from __future__ import annotations

import pandas as pd

DATASETS: dict[str, dict] = {
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
    "adult": {
        "name": "Adult Census Income",
        "vertical": "Tax & Income",
        "source": "UCI Machine Learning Repository",
        "description": "Demographic data from the 1994 US Census database. Standard benchmark for synthetic data.",
        "columns": [
            "age",
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
            "income",
        ],
        "col_count": 13,
        "tags": ["CSV", "Classification", "Benchmark"],
        "use_cases": [
            "Income prediction",
            "Fairness auditing",
            "Synthetic data benchmarking",
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
        from tabular_polygraph.dataset.downloader import download
        download("bls")
        download("census_acs")
        download("adult")

    Args:
        dataset_id: One of ["bls", "census_acs", "adult"]
        n: Max number of records to return (default 2000)

    Returns:
        Sampled DataFrame of real data from cache or download
    """
    if dataset_id not in DATASETS:
        available = ", ".join(sorted(DATASETS.keys()))
        raise ValueError(
            f"Unknown dataset '{dataset_id}'.\n"
            f"Available real datasets: {available}\n"
            f"Download real data: from tabular_polygraph.dataset.downloader import download"
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
            f"Dataset '{dataset_id}' not cached. Download real data first: tabular-polygraph download {dataset_id}"
        )
