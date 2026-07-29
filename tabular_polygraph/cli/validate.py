"""Validate command — checks a real data file for issues."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tabular_polygraph.io.formats import _check_safe_path

from .utils import C, _c, _json_clean, err, header, info, ok, section


def cmd_validate(args):
    from tabular_polygraph.io import read, validate

    header(f"Validating: {args.file}")

    if not Path(args.file).exists():
        err(f"File not found: {args.file}")
        sys.exit(1)

    info(f"Loading: {args.file}")
    df = read(args.file)
    info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()

    result = validate(
        df,
        null_threshold=args.null_threshold,
        duplicate_threshold=args.dup_threshold,
        max_cardinality=args.max_cardinality,
        min_rows=args.min_rows,
    )

    if args.json or args.output:
        report = {
            "file": args.file,
            "passed": result.passed,
            "errors": result.errors,
            "warnings": result.warnings,
            "stats": result.stats,
        }
        if args.json:
            print(json.dumps(_json_clean(report), indent=2))
        if args.output:
            _check_safe_path(Path(args.output))
            Path(args.output).write_text(json.dumps(report, indent=2, default=str))
            ok(f"Report saved → {args.output}")
        return

    if result.passed:
        ok("Validation PASSED")
    else:
        err("Validation FAILED")

    if result.errors:
        section(f"Errors ({len(result.errors)})")
        for e in result.errors:
            print(f"    {_c('✗', C.RED)} {e}")

    if result.warnings:
        section(f"Warnings ({len(result.warnings)})")
        for w in result.warnings:
            print(f"    {_c('!', C.YELLOW)} {w}")

    section("Column summary")
    col_stats = result.stats.get("columns", {})
    for col, cs in list(col_stats.items())[:20]:
        null_pct = f"{cs['null_frac'] * 100:.1f}%"
        print(
            f"    {col:<28}  dtype={cs['dtype']:<12}  nulls={null_pct:<8}  unique={cs['n_unique']}"
        )

    print()
    print(
        f"    Duplicate rows: {result.stats.get('duplicate_rows', 0)}  ({result.stats.get('duplicate_frac', 0) * 100:.1f}%)"
    )
    print()


__all__ = [
    "cmd_validate",
]
