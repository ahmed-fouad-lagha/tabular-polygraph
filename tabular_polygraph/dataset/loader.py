"""Dataset metadata and cached loading for downloadable datasets."""

from __future__ import annotations

import copy

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
