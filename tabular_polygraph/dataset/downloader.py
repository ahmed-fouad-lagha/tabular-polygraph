"""
Download data from public sources and cache it locally.

Usage
-----
    # Download one dataset
    from tabular_polygraph.dataset.downloader import download, status
    download("fred_macro")

    # Download all
    download("all")

    # Check what's cached
    status()

CLI
---
    tabular-polygraph download fred_macro
    tabular-polygraph download all
    tabular-polygraph download status

Sources
-------
All sources are public, and require no authentication except FRED
(which needs a free API key from fred.stlouisfed.org/docs/api/api_key.html).
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


def _cache_dir() -> Path:
    """Get the cache directory and ensure it exists."""
    base = Path(
        os.environ.get(
            "TABULAR_POLYGRAPH_CACHE", Path.home() / ".tabular_polygraph" / "cache"
        )
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path(dataset_id: str) -> Path:
    return _cache_dir() / f"{dataset_id}.parquet"


def is_cached(dataset_id: str) -> bool:
    return cache_path(dataset_id).exists()


DOWNLOADERS: dict[str, dict] = {
    "fred_macro": {
        "name": "FRED Macroeconomic Indicators",
        "source": "Federal Reserve FRED",
        "url": "https://api.stlouisfed.org/fred/series/observations",
        "method": "fred_api",
        "size_hint": "~1 MB — fast",
        "requires": "FRED_API_KEY environment variable",
        "indicators": {
            "GDP": "gdp_growth_yoy",
            "CPIAUCSL": "cpi_yoy",
            "CPILFESL": "core_cpi_yoy",
            "UNRATE": "unemployment_rate",
            "FEDFUNDS": "fed_funds_rate",
            "GS10": "t10y_rate",
            "GS2": "t2y_rate",
            "M2SL": "m2_growth",
            "HOUST": "housing_starts",
            "INDPRO": "industrial_production",
            "UMCSENT": "consumer_sentiment",
            "VIXCLS": "vix",
        },
    },
    "bls": {
        "name": "BLS Employment & Wages",
        "source": "Bureau of Labor Statistics QCEW",
        "url": "https://www.bls.gov/cew/",
        "method": "bls_api",
        "size_hint": "~200 MB",
        "indicators": {
            "industry_code": "naics_sector",
            "own_code": "ownership",
            "area_fips": "state",
            "avg_wkly_wage": "avg_weekly_wage",
            "month3_emplvl": "total_employment",
            "oty_month3_emplvl_pct_chg": "yoy_employment_change",
            "oty_avg_wkly_wage_pct_chg": "yoy_wage_change",
            "qtrly_estabs": "establishments",
            "qtr": "quarter",
            "year": "year",
        },
    },
    "world_bank": {
        "name": "World Bank Development Indicators",
        "source": "World Bank WDI API",
        "url": "https://api.worldbank.org/v2/country/all/indicator",
        "method": "worldbank_api",
        "size_hint": "~5 MB — fast",
        "indicators": {
            "NY.GDP.MKTP.KD.ZG": "gdp_growth",
            "NY.GDP.PCAP.KD": "gdp_per_capita",
            "FP.CPI.TOTL.ZG": "inflation",
            "BN.CAB.XOKA.GD.ZS": "current_account_pct_gdp",
            "BX.KLT.DINV.WD.GD.ZS": "fdi_pct_gdp",
            "GC.DOD.TOTL.GD.ZS": "govt_debt_pct_gdp",
            "SP.POP.GROW": "population_growth",
            "SI.POV.GINI": "gini",
        },
    },
    "census_acs": {
        "name": "Census ACS Income & Housing",
        "source": "US Census ACS 5-Year API",
        "url": "https://api.census.gov/data/2022/acs/acs5",
        "method": "census_api",
        "size_hint": "~10 MB",
        "indicators": {
            "B19013_001E": "household_income",
            "B25105_001E": "housing_cost",
            "B25071_001E": "cost_burden_pct",
            "B17001_002E": "poverty_count",
            "B17001_001E": "poverty_total",
            "B23025_005E": "unemployed_count",
            "B23025_002E": "labor_force_total",
            "B25010_001E": "household_size",
            "B25003_002E": "owner_occupied_count",
            "B25003_001E": "tenure_total",
            "B01002_001E": "age_group",
            "B15003_022E": "bachelors_count",
            "B15003_001E": "education_total",
        },
    },
    "adult": {
        "name": "Adult Census Income",
        "source": "UCI Machine Learning Repository",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "method": "direct_csv",
        "size_hint": "~4 MB — fast",
        "indicators": {
            "age": "age",
            "workclass": "workclass",
            "education": "education",
            "marital_status": "marital_status",
            "occupation": "occupation",
            "relationship": "relationship",
            "race": "race",
            "sex": "sex",
            "capital_gain": "capital_gain",
            "capital_loss": "capital_loss",
            "hours_per_week": "hours_per_week",
            "native_country": "native_country",
            "income": "income",
        },
    },
}


def _download_world_bank(dataset_id: str, n_sample: int = 10000) -> pd.DataFrame:
    """Download World Bank WDI via API. No key needed."""

    indicators = DOWNLOADERS["world_bank"]["indicators"]
    base_url = DOWNLOADERS["world_bank"]["url"]
    frames = {}

    for code, col_name in indicators.items():
        url = f"{base_url}/{code}?format=json&per_page=20000&mrv=25&date=2000:2023"
        print(f"    Fetching {col_name} ({code})...", end=" ", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            records = data[1] if len(data) > 1 else []
            rows = []
            for rec in records:
                if rec.get("value") is not None:
                    rows.append(
                        {
                            "country_code": rec.get("countryiso3code"),
                            "year": int(rec["date"]),
                            col_name: float(rec["value"]),
                        }
                    )
            frames[col_name] = pd.DataFrame(rows)
            print(f"{len(rows)} rows")
        except Exception as e:
            print(f"FAILED ({e})")
            continue
        time.sleep(0.3)  # be polite to the API

    if not frames:
        raise RuntimeError("No World Bank data downloaded")

    # Merge all indicators on country_code + year
    df = None
    for _col_name, frame in frames.items():
        if df is None:
            df = frame
        else:
            # Ensure we don't multiply rows during indicator merge
            df = df.merge(frame, on=["country_code", "year"], how="outer")

    if df is None:
        raise RuntimeError("No World Bank data merged")

    # Add metadata columns
    meta_url = "https://api.worldbank.org/v2/country?format=json&per_page=300"
    try:
        with urllib.request.urlopen(meta_url, timeout=30) as r:
            meta_data = json.loads(r.read())
        meta_rows = []
        for c in meta_data[1]:
            meta_rows.append(
                {
                    "country_code": c.get("id"),
                    "income_group": c.get("incomeLevel", {}).get("value", "Unknown"),
                    "region": c.get("region", {}).get("value", "Unknown"),
                }
            )
        meta_df = pd.DataFrame(meta_rows).drop_duplicates(subset=["country_code"])
        df = df.merge(meta_df, on="country_code", how="left")
    except Exception:
        df["income_group"] = "Unknown"
        df["region"] = "Unknown"

    df = df.dropna(subset=["gdp_growth"]) if "gdp_growth" in df.columns else df.dropna()
    df = df.reset_index(drop=True)
    return df.sample(min(n_sample, len(df)), random_state=42)


def _download_fred(dataset_id: str, n_sample: int = 10000) -> pd.DataFrame:
    """Download FRED series via API. Requires FRED_API_KEY env var."""

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY environment variable not set.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then: export FRED_API_KEY=your_key_here"
        )

    indicators = DOWNLOADERS["fred_macro"]["indicators"]
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    frames = {}

    for series_id, col_name in indicators.items():
        params = urllib.parse.urlencode(
            {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": "2000-01-01",
                "frequency": "m",
            }
        )
        url = f"{base_url}?{params}"
        print(f"    Fetching {col_name} ({series_id})...", end=" ", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            rows = [
                {"date": obs["date"], col_name: float(obs["value"])}
                for obs in data["observations"]
                if obs["value"] != "."
            ]
            frames[col_name] = pd.DataFrame(rows)
            print(f"{len(rows)} observations")
        except Exception as e:
            print(f"FAILED ({e})")
        time.sleep(0.2)

    if not frames:
        raise RuntimeError("No FRED data downloaded")

    df = None
    for _col_name, frame in frames.items():
        if df is None:
            df = frame
        else:
            df = df.merge(frame, on="date", how="outer")
    if df is None:
        raise RuntimeError("No FRED data merged")

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    # Target specific indicator for dropna to avoid fragility
    primary_col = (
        "gdp_growth_yoy"
        if "gdp_growth_yoy" in df.columns
        else list(indicators.values())[0]
    )
    df = df.dropna(subset=[primary_col])

    # Add VIX if available (it's in a different series)
    if "vix" not in df.columns:
        df["vix"] = np.nan

    return df.sample(min(n_sample, len(df)), random_state=42)


def _download_census(dataset_id: str, n_sample: int = 10000) -> pd.DataFrame:
    """Download Census ACS via API. Pulls PUMA-level demographic profiles."""

    var_map = DOWNLOADERS["census_acs"]["indicators"]
    variables = ",".join(var_map.keys())

    states = [f"{i:02d}" for i in range(1, 57)]
    all_rows = []

    for state in states:
        params = urllib.parse.urlencode(
            {
                "get": variables,
                "for": "public use microdata area:*",
                "in": f"state:{state}",
            }
        )
        url = f"https://api.census.gov/data/2022/acs/acs5?{params}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            headers = data[0]
            for row in data[1:]:
                all_rows.append(dict(zip(headers, row, strict=False)))
            if int(state) % 10 == 0:
                print(f"    Progress: {state}/56 states...", flush=True)
        except Exception:
            continue
        time.sleep(0.05)

    if not all_rows:
        raise RuntimeError("No Census data downloaded")

    df = pd.DataFrame(all_rows)
    # Rename variables
    df = df.rename(columns=var_map)

    # Convert to numeric
    for col in df.columns:
        if col not in ["state", "public use microdata area"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature Engineering (Ratios)
    # We remove .fillna() to ensure missing data is handled explicitly by dropna()
    df["poverty_status"] = df["poverty_count"] / df["poverty_total"]
    df["employment_status"] = 1 - (df["unemployed_count"] / df["labor_force_total"])
    df["tenure"] = df["owner_occupied_count"] / df["tenure_total"]
    df["education"] = df["bachelors_count"] / df["education_total"]
    df["puma"] = df["public use microdata area"]

    cols_to_keep = [
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
    ]
    df = df[cols_to_keep].dropna()

    return df.sample(min(n_sample, len(df)), random_state=42)


def _download_bls(dataset_id: str, n_sample: int = 10000) -> pd.DataFrame:
    """Download BLS QCEW state-level employment and wages."""

    # We fetch Quarterly data for a high-resolution time series.
    # To keep the download fast, we focus on the US National level and Top 5 states.
    years = list(range(2018, 2024))
    quarters = [1, 2, 3, 4]

    # Area Codes: US National (US000) + Top 10 states (CA, TX, NY, FL, IL, PA, OH, GA, NC, WA)
    area_codes = [
        "US000",
        "06000",
        "48000",
        "36000",
        "12000",
        "17000",
        "42000",
        "39000",
        "13000",
        "37000",
        "53000",
    ]

    base_url = "https://data.bls.gov/cew/data/api"
    frames = []

    for year in years:
        print(f"    Fetching QCEW quarterly data for {year}...", flush=True)
        for qtr in quarters:
            for area_code in area_codes:
                url = f"{base_url}/{year}/{qtr}/area/{area_code}.csv"
                try:
                    with urllib.request.urlopen(url, timeout=5) as r:
                        df_q = pd.read_csv(
                            io.BytesIO(r.read()),
                            dtype={"area_fips": str, "qtr": str, "industry_code": str},
                        )
                        # Filter to Private Sector (ownership=5) and NAICS Sectors (2-digit or Total)
                        df_q = df_q[
                            (df_q["own_code"] == 5)
                            & (df_q["industry_code"].str.len() <= 3)
                        ]
                        if not df_q.empty:
                            frames.append(df_q)
                except Exception:
                    continue
                time.sleep(0.005)

    if not frames:
        raise RuntimeError("BLS download produced 0 usable rows after cleaning")

    raw = pd.concat(frames, ignore_index=True)

    required = DOWNLOADERS["bls"]["indicators"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"BLS payload missing expected columns: {missing}")

    df = raw[list(required)].rename(columns=required)

    # Convert numeric columns
    for col in [
        "avg_weekly_wage",
        "total_employment",
        "yoy_employment_change",
        "yoy_wage_change",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Final cleaning
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce")
    df = df.dropna(
        subset=[
            "naics_sector",
            "ownership",
            "state",
            "avg_weekly_wage",
            "total_employment",
            "year",
            "quarter",
        ]
    )

    return df.sample(min(n_sample, len(df)), random_state=42)


def _download_adult(dataset_id: str, n_sample: int = 50000) -> pd.DataFrame:
    """Download Adult dataset from UCI repository."""
    url = DOWNLOADERS["adult"]["url"]
    cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    print("    Fetching Adult dataset from UCI...")
    df = pd.read_csv(url, names=cols, sep=r",\s*", engine="python", na_values="?")
    df = df.dropna()

    # Drop fnlwgt and education-num as they are redundant or administrative
    df = df.drop(columns=["fnlwgt", "education-num"])

    # Clean up column names to match indicators
    df.columns = [c.replace("-", "_") for c in df.columns]

    return df.sample(min(n_sample, len(df)), random_state=42)


# Mapping of registry methods to internal downloader functions
METHOD_MAP = {
    "worldbank_api": _download_world_bank,
    "fred_api": _download_fred,
    "bls_api": _download_bls,
    "census_api": _download_census,
    "direct_csv": _download_adult,
}


def download(
    dataset_id: str,
    force: bool = False,
    n_sample: int = 50_000,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Download dataset(s) and cache them locally.

    Parameters
    ----------
    dataset_id : dataset ID, or "all" to download everything
    force      : re-download even if cached
    n_sample   : max rows to cache (default 50,000)

    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame]
        A single DataFrame if a specific ID was requested,
        or a dictionary of {id: DataFrame} if 'all' was requested.

    Example
    -------
        from tabular_polygraph.dataset.downloader import download
        df = download("world_bank")
        all_data = download("all")  # yields a dictionary
    """
    if dataset_id == "all":
        results = {}
        for did in DOWNLOADERS:
            try:
                results[did] = download(did, force=force, n_sample=n_sample)
            except Exception as e:
                print(f"  ✗ {did}: {e}")

        failed = [did for did in DOWNLOADERS if did not in results]
        if failed:
            print(f"\n  {len(failed)} dataset(s) failed: {', '.join(failed)}")

        return results

    if dataset_id not in DOWNLOADERS:
        available = ", ".join(DOWNLOADERS)
        raise ValueError(
            f"No downloader for '{dataset_id}'. Available: {available}\n"
            f"For other datasets, pass your own CSV to gen.fit(your_df)."
        )

    cached = cache_path(dataset_id)
    if not force:
        df_cached = load_cached(dataset_id)
        if df_cached is not None:
            print(f"  ✓ {dataset_id} — loaded from cache ({cached})")
            return df_cached

    info = DOWNLOADERS[dataset_id]
    print(f"\n  Downloading {info['name']}...")
    print(f"  Source: {info['source']}")
    print(f"  Size:   {info['size_hint']}")

    method = info["method"]
    if method in METHOD_MAP:
        df = METHOD_MAP[method](dataset_id, n_sample)
    else:
        raise NotImplementedError(
            f"Downloader for '{dataset_id}' (method: {method}) requires manual download.\n"
            f"Please download the data from {info['url']} and save it as a CSV, then load it with gen.fit(your_csv)."
        )

    # 3. Cache it
    df.to_parquet(cached, index=False)
    print(f"  ✓ Cached {len(df):,} rows → {cached}")
    return df


def load_cached(dataset_id: str) -> pd.DataFrame | None:
    """
    Load cached real data if available, else return None.
    """
    p = cache_path(dataset_id)
    if p.exists():
        return pd.read_parquet(p)
    return None


def status() -> pd.DataFrame:
    """Show which datasets have been downloaded and cached."""
    rows = []
    for did, _info in DOWNLOADERS.items():
        p = cache_path(did)
        if p.exists():
            df = pd.read_parquet(p)
            rows.append(
                {
                    "dataset": did,
                    "status": "✓ cached",
                    "rows": f"{len(df):,}",
                    "size": f"{p.stat().st_size // 1024:,} KB",
                    "path": str(p),
                }
            )
        else:
            rows.append(
                {
                    "dataset": did,
                    "status": "○ not downloaded",
                    "rows": "—",
                    "size": "—",
                    "path": "—",
                }
            )
    return pd.DataFrame(rows)
