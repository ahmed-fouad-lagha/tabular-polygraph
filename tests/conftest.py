import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _fallback_seed(did: str, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    if did == "fred_macro":
        years = np.arange(1980, 1980 + n)
        return pd.DataFrame(
            {
                "year": years,
                "gdp_growth_yoy": rng.normal(2.0, 1.0, n),
                "cpi_yoy": rng.normal(2.5, 0.8, n),
                "unemployment_rate": rng.normal(5.5, 1.2, n).clip(2.0, 15.0),
                "fed_funds_rate": rng.normal(3.0, 1.0, n).clip(0.0, 10.0),
                "vix": rng.normal(20.0, 6.0, n).clip(8.0, 80.0),
                "industrial_production": rng.normal(100.0, 15.0, n),
                "retail_sales": rng.normal(500.0, 50.0, n),
                "housing_starts": rng.normal(1400.0, 200.0, n),
                "consumer_confidence": rng.normal(95.0, 12.0, n),
                "yield_10y": rng.normal(3.0, 0.8, n),
                "yield_2y": rng.normal(2.5, 0.7, n),
                "money_supply_m2": rng.normal(10000.0, 1200.0, n),
                "payroll_growth": rng.normal(1.5, 0.7, n),
                "dollar_index": rng.normal(100.0, 8.0, n),
            }
        )

    if did == "bls":
        quarter = np.tile(np.array(["Q1", "Q2", "Q3", "Q4"]), int(np.ceil(n / 4)))[:n]
        return pd.DataFrame(
            {
                "quarter": quarter,
                "cpi_yoy": rng.normal(2.5, 0.7, n),
                "unemployment_rate": rng.normal(5.0, 1.0, n).clip(2.0, 15.0),
                "labor_force_participation": rng.normal(63.0, 1.0, n),
                "avg_hourly_earnings_yoy": rng.normal(3.0, 0.8, n),
            }
        )

    if did == "world_bank":
        countries = np.array(["US", "DE", "FR", "JP", "IN", "BR", "ZA", "MX"])
        return pd.DataFrame(
            {
                "country_code": rng.choice(countries, n),
                "year": rng.integers(1995, 2023, n),
                "gdp_per_capita": rng.normal(15000.0, 8000.0, n).clip(500.0, None),
                "inflation": rng.normal(3.0, 2.0, n).clip(-2.0, 30.0),
                "unemployment": rng.normal(7.0, 3.0, n).clip(1.0, 35.0),
            }
        )

    if did == "census_acs":
        return pd.DataFrame(
            {
                "age": rng.integers(18, 90, n),
                "income": rng.normal(60000.0, 25000.0, n).clip(1000.0, None),
                "education": rng.choice(["HS", "College", "Graduate"], n),
                "employment_status": rng.choice(
                    ["employed", "unemployed"], n, p=[0.92, 0.08]
                ),
            }
        )

    raise ValueError(f"Unsupported fallback dataset: {did}")


@pytest.fixture(scope="module")
def all_seeds():
    from tabular_polygraph.catalog import load_dataset, DATASETS

    result = {}
    for did in DATASETS:
        try:
            result[did] = load_dataset(did)
        except ValueError:
            result[did] = _fallback_seed(did)
    return result


@pytest.fixture(scope="module")
def fred_macro(all_seeds):
    return all_seeds.get("fred_macro")


@pytest.fixture(scope="module")
def world_bank(all_seeds):
    return all_seeds.get("world_bank")


@pytest.fixture(scope="module")
def census_acs(all_seeds):
    return all_seeds.get("census_acs")


@pytest.fixture(scope="module")
def syn_macro(fred_macro):
    from tabular_polygraph.generators.time_series import VARGenerator

    gen = VARGenerator(lags=2, time_col="year")
    gen.fit(fred_macro)
    return gen.generate(300, seed=42)


@pytest.fixture(scope="module")
def syn_wb(world_bank):
    from tabular_polygraph.generators.panel import FixedEffectsGenerator

    gen = FixedEffectsGenerator(entity_col="country_code", time_col="year")
    gen.fit(world_bank)
    return gen.generate(300, seed=42)
