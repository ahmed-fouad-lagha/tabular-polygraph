"""Dataset metadata and cached loading for downloadable datasets."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from .registry import DATASETS


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
                "columns": len(m["columns"]),
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
    return copy.deepcopy(DATASETS[dataset_id])


def _snapshot_path(dataset_id: str) -> Path | None:
    """Path to the repository-bundled dataset snapshot, if present.

    The ``data/cache/*.parquet`` files are exact copies of the downloads that
    produced every number in the manuscript. Bundling them means a fresh clone
    reproduces the paper's results offline and independently of any upstream
    source (e.g. the Census API) changing over time.
    """
    root = Path(__file__).resolve().parents[2]
    p = root / "data" / "cache" / f"{dataset_id}.parquet"
    return p if p.exists() else None


def load_cached(dataset_id: str) -> pd.DataFrame | None:
    """Load cached real data if available, else return None.

    The repository snapshot (``data/cache/``) is authoritative when present;
    otherwise the local cache (``~/.tabular_polygraph/cache``) is used.

    Columns listed in ``drop_cols`` are silently removed so downstream
    consumers never see administrative / identifier columns.
    """
    from .downloader import cache_path

    p = _snapshot_path(dataset_id) or cache_path(dataset_id)
    if p.exists():
        df = pd.read_parquet(p)
        drop_cols = DATASETS.get(dataset_id, {}).get("drop_cols", [])
        if drop_cols:
            existing = [c for c in drop_cols if c in df.columns]
            if existing:
                df = df.drop(columns=existing)
        return df
    return None


def load_dataset(dataset_id: str, n: int = 2000) -> pd.DataFrame:
    """
    Load real data for a dataset via API downloader.

    To use, first download:
        from tabular_polygraph.dataset.downloader import download
        download("census_acs")
        download("adult")
        download("credit")
        download("supermarket_sales")
        download("online_purchases")

    Args:
        dataset_id: One of ["census_acs", "adult", "credit", "supermarket_sales", "online_purchases"]
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

    cached = load_cached(dataset_id)
    if cached is not None:
        if len(cached) >= 100:
            return cached.sample(min(n, len(cached)), random_state=42).reset_index(
                drop=True
            )
        raise ValueError(
            f"Dataset '{dataset_id}' has only {len(cached)} rows (need ≥100). "
            f"Cached data is too small for meaningful analysis."
        )
    raise ValueError(
        f"Dataset '{dataset_id}' not cached. Download real data first: tabular-polygraph download {dataset_id}"
    )
