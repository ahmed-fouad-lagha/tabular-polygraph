"""List and info commands — dataset discovery."""

from __future__ import annotations

import sys

from .utils import _c, C, dim, err, header


def cmd_list(args):
    from tabular_polygraph.dataset import list_datasets

    df = list_datasets(vertical=args.vertical if hasattr(args, "vertical") else None)
    header(
        "Available datasets",
        f"{len(df)} datasets · {df['vertical'].nunique()} verticals",
    )

    for vertical in df["vertical"].unique():
        sub = df[df["vertical"] == vertical]
        print(f"\n  {_c(vertical.upper(), C.CYAN)}")
        for _, row in sub.iterrows():
            print(f"    {_c(row['id'], C.BOLD):<28} {row['name']:<40}")

    print()
    dim("  tabular-polygraph info <id>    full metadata + columns")
    dim("  tabular-polygraph generate <id>    generate synthetic data")
    print()


def cmd_info(args):
    from tabular_polygraph.dataset import get_dataset_info

    try:
        meta = get_dataset_info(args.dataset)
    except ValueError as e:
        from .utils import err as _err
        _err(str(e))
        sys.exit(1)

    header(f"Dataset: {args.dataset}", meta["name"])

    pairs = [
        ("Vertical", meta["vertical"]),
        ("Source", meta["source"]),
        ("Columns", str(len(meta["columns"]))),
        ("Tags", ", ".join(meta["tags"])),
    ]
    for label, value in pairs:
        print(f"    {_c(label + ':', C.GRAY):<22}{value}")

    print(f"\n    {_c('Columns:', C.GRAY)}")
    cols = meta["columns"]
    for i in range(0, len(cols), 4):
        print("    " + "  ".join(_c(c, C.CYAN) for c in cols[i : i + 4]))

    print(f"\n    {_c('Use cases:', C.GRAY)}")
    for uc in meta["use_cases"]:
        print(f"    · {uc}")

    print(f"\n    {_c('Description:', C.GRAY)}")
    print(f"    {meta['description']}")
    print()


__all__ = [
    "cmd_list",
    "cmd_info",
]
