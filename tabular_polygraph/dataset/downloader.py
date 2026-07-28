"""
Download data from public sources and cache it locally.

Usage
-----
    # Download one dataset
    from tabular_polygraph.dataset.downloader import download, status
    download("adult")

    # Download all
    download("all")

    # Check what's cached
    status()

CLI
---
    tabular-polygraph download adult
    tabular-polygraph download all
    tabular-polygraph download status
"""

from __future__ import annotations

__all__ = ["download", "status", "cache_path", "is_cached", "load_cached"]

import os
from pathlib import Path

import pandas as pd

from .registry import DATASETS


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


def _download_census(n_sample: int = 10000) -> pd.DataFrame:
    """Download Census ACS via API. Pulls PUMA-level demographic profiles."""
    import json
    import time
    import urllib.parse
    import urllib.request

    import numpy as np

    var_map = DATASETS["census_acs"]["indicators"]
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
                all_rows.append(dict(zip(headers, row, strict=True)))
            if int(state) % 10 == 0:
                print(f"    Progress: {state}/56 states...", flush=True)
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Census ACS download failed for state {state}: {exc}",
                UserWarning,
                stacklevel=2,
            )
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

    # Replace Inf from division by zero with NaN so dropna() catches them
    df = df.replace([np.inf, -np.inf], np.nan)

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


def _download_adult(n_sample: int = 50000) -> pd.DataFrame:
    """Download Adult dataset from UCI repository."""
    url = DATASETS["adult"]["url"]
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


def _download_credit(n_sample: int = 50000) -> pd.DataFrame:
    """Download Credit Card Default dataset from UCI repository."""
    url = DATASETS["credit"]["url"]
    print("    Fetching Credit Card Default dataset from UCI...")
    df = pd.read_excel(url, header=1)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    rename_map = DATASETS["credit"]["indicators"]
    df = df.rename(columns=rename_map)
    df.columns = [c.strip().lower() for c in df.columns]

    # Cast categorical columns to string so HIF can distinguish them from continuous
    cat_cols = [
        "sex",
        "education",
        "marriage",
        "pay_0",
        "pay_2",
        "pay_3",
        "pay_4",
        "pay_5",
        "pay_6",
        "default_payment",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df.sample(min(n_sample, len(df)), random_state=42)


def _download_supermarket_sales(n_sample: int = 50000) -> pd.DataFrame:
    """Download Supermarket Sales dataset from Plotly datasets."""
    url = DATASETS["supermarket_sales"]["url"]
    print("    Fetching Supermarket Sales dataset from Plotly...")

    df = pd.read_csv(url)

    # Rename columns using indicators
    rename_map = DATASETS["supermarket_sales"]["indicators"]
    df = df.rename(columns=rename_map)

    # Cast categorical columns to string
    cat_cols = [
        "invoice_id",
        "branch",
        "city",
        "customer_type",
        "gender",
        "product_line",
        "payment",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # customer_rating is numeric (1.0-10.0), not categorical
    if "customer_rating" in df.columns:
        df["customer_rating"] = pd.to_numeric(df["customer_rating"], errors="coerce")

    # Drop date/time and constant/useless columns
    df = df.drop(
        columns=["date", "time", "gross_margin_pct", "invoice_id"], errors="ignore"
    )

    return df.sample(min(n_sample, len(df)), random_state=42)


def _download_online_purchases(n_sample: int = 50000) -> pd.DataFrame:
    """Download Online Purchases dataset from GitHub."""
    url = DATASETS["online_purchases"]["url"]
    print("    Fetching Online Purchases dataset from GitHub...")

    df = pd.read_csv(url)

    # Rename columns using indicators
    rename_map = DATASETS["online_purchases"]["indicators"]
    df = df.rename(columns=rename_map)

    # Clean price columns (remove $ sign if present)
    for col in [
        "list_price",
        "purchase_price",
        "item_subtotal",
        "item_tax",
        "item_total",
    ]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cast category to string
    if "category" in df.columns:
        df["category"] = df["category"].fillna("Unknown").astype(str)

    # Drop date columns for now
    df = df.drop(columns=["order_date", "shipment_date"], errors="ignore")

    return df.sample(min(n_sample, len(df)), random_state=42)


_DOWNLOADERS = {
    "census_acs": _download_census,
    "adult": _download_adult,
    "credit": _download_credit,
    "supermarket_sales": _download_supermarket_sales,
    "online_purchases": _download_online_purchases,
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
        df = download("adult")
        all_data = download("all")  # yields a dictionary
    """
    if dataset_id == "all":
        results = {}
        for ds_id in DATASETS:
            try:
                results[ds_id] = download(ds_id, force=force, n_sample=n_sample)
            except Exception as e:
                print(f"  ✗ {ds_id}: {e}")

        failed = [ds_id for ds_id in DATASETS if ds_id not in results]
        if failed:
            print(f"\n  {len(failed)} dataset(s) failed: {', '.join(failed)}")

        return results

    if dataset_id not in DATASETS:
        available = ", ".join(DATASETS)
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

    info = DATASETS[dataset_id]
    print(f"\n  Downloading {info['name']}...")
    print(f"  Source: {info['source']}")
    print(f"  Size:   {info['size_hint']}")

    df = _DOWNLOADERS[dataset_id](n_sample)

    df.to_parquet(cached, index=False)
    print(f"  ✓ Cached {len(df):,} rows → {cached}")
    return df


def load_cached(dataset_id: str) -> pd.DataFrame | None:
    """Load cached real data if available, else return None."""
    p = cache_path(dataset_id)
    if p.exists():
        df = pd.read_parquet(p)
        drop_cols = DATASETS.get(dataset_id, {}).get("drop_cols", [])
        if drop_cols:
            existing = [c for c in drop_cols if c in df.columns]
            if existing:
                df = df.drop(columns=existing)
        return df
    return None


def status() -> pd.DataFrame:
    """Show which datasets have been downloaded and cached."""
    rows = []
    for dataset_id in DATASETS:
        p = cache_path(dataset_id)
        if p.exists():
            rows.append(
                {
                    "dataset": dataset_id,
                    "status": "cached",
                    "size": f"{p.stat().st_size // 1024:,} KB",
                    "path": str(p),
                }
            )
        else:
            rows.append(
                {
                    "dataset": dataset_id,
                    "status": "not downloaded",
                    "size": "—",
                    "path": "—",
                }
            )
    return pd.DataFrame(rows)
