"""Download command — fetch real-world datasets."""

from __future__ import annotations

import sys

from .utils import C, _c, dim, err, header, info, ok, warn


def cmd_download(args):
    from tabular_polygraph.dataset.downloader import (
        download,
        is_cached,
        status,
    )
    from tabular_polygraph.dataset.registry import DATASETS

    if args.dataset == "status":
        header("Download status")
        df = status()
        for _, row in df.iterrows():
            icon = _c("✓", C.GREEN) if "cached" in row["status"] else _c("○", C.GRAY)
            print(f"    {icon} {row['dataset']:<28} {row['size']}")
        print()
        dim("  Run: tabular-polygraph download <id>   to download real data")
        dim("  Run: tabular-polygraph download all    to download everything")
        print()
        return

    dataset_id = args.dataset

    if dataset_id != "all" and dataset_id not in DATASETS:
        err(f"No downloader for '{dataset_id}'.")
        info(f"Available: {', '.join(DATASETS)}")
        info("For other datasets, use gen.fit(your_csv) to bring your own data.")
        sys.exit(1)

    header(f"Downloading: {dataset_id}", "from public sources")

    if dataset_id != "all" and is_cached(dataset_id) and not args.force:
        warn("Already cached. Use --force to re-download.")
        return

    try:
        result = download(dataset_id, force=args.force, n_sample=args.sample)
        if isinstance(result, dict):
            print()
            ok(f"Downloaded {len(result)} datasets")
        else:
            print()
            ok(f"{len(result):,} rows cached")
            info(f"Generator will now use real data for '{dataset_id}'")
    except ValueError as e:
        err(str(e))
        sys.exit(1)
    except Exception as e:
        err(f"Download failed: {e}")
        sys.exit(1)
    print()


__all__ = [
    "cmd_download",
]
