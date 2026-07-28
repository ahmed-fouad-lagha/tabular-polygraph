"""Argument parser and main entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from tabular_polygraph.cli.download import cmd_download
from tabular_polygraph.cli.evaluate import cmd_evaluate
from tabular_polygraph.cli.generate import cmd_generate
from tabular_polygraph.cli.listinfo import cmd_info, cmd_list
from tabular_polygraph.cli.validate import cmd_validate

from .helpers import _positive_int
from .utils import dim


def main():
    parser = argparse.ArgumentParser(prog="tabular-polygraph", add_help=False)
    parser.add_argument("--help", "-h", action="help")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # list
    p = sub.add_parser("list", help="List available datasets.")
    p.add_argument(
        "--vertical",
        type=str,
        default=None,
        metavar="V",
        help="Filter by vertical e.g. 'Capital Markets'",
    )
    p.set_defaults(func=cmd_list)

    # info
    p = sub.add_parser("info", help="Show full dataset metadata.")
    p.add_argument("dataset")
    p.set_defaults(func=cmd_info)

    # generate
    p = sub.add_parser("generate", help="Generate synthetic data.")
    p.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Built-in dataset ID (e.g. adult). Omit if using --input.",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to your own CSV. Fit the generator on it instead of a built-in dataset.",
    )
    p.add_argument("--rows", type=_positive_int, default=1000, metavar="N")
    p.add_argument(
        "--output",
        type=str,
        default="output.csv",
        metavar="FILE",
        help="Output path. Extension determines format: .csv .json .dta .xlsx",
    )
    p.add_argument(
        "--generator",
        type=str,
        default="auto",
        choices=["auto", "copula", "ctgan", "tvae", "vine"],
        metavar="TYPE",
        help="auto | copula | ctgan | tvae | vine",
    )
    p.add_argument(
        "--fit-rows",
        type=int,
        default=None,
        metavar="N",
        help="Rows to use for fitting (default: all cached rows)",
    )
    p.add_argument(
        "--filter",
        type=str,
        action="append",
        metavar="EXPR",
        help="e.g. --filter state:CA,TX  --filter dti_min:45",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Override default training epochs for deep generators",
    )
    p.add_argument(
        "-t",
        "--target",
        type=str,
        default=None,
        metavar="COL",
        help="Target column for downstream TSTR ML evaluation",
    )
    p.add_argument("--seed", type=int, default=42, metavar="INT")
    p.add_argument(
        "--hif-epochs",
        type=int,
        default=10,
        metavar="N",
        help="Training epochs for the logical integrity oracle (default: 10)",
    )
    p.add_argument(
        "--hif-hubs",
        type=int,
        default=5,
        help="Number of high-dependency hubs to audit",
    )
    p.add_argument(
        "--hif-depth", type=int, default=12, help="Max depth for sentinel forests"
    )
    p.add_argument(
        "--drop-cols",
        type=str,
        default=None,
        metavar="COLS",
        help="Comma-separated columns to drop before fit/eval, e.g. tract_id,customer_id",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable detailed debug logs"
    )
    p.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress outputs"
    )
    p.set_defaults(func=cmd_generate)

    # evaluate
    p = sub.add_parser("evaluate", help="Full fidelity report.")
    p.add_argument("real")
    p.add_argument("synthetic")
    p.add_argument(
        "--type",
        type=str,
        default="cross_sectional",
        metavar="TYPE",
        help="cross_sectional",
    )
    p.add_argument(
        "--hif-epochs", type=int, default=10, help="Training epochs for neural auditor"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for deterministic evaluation (default: 42)",
    )
    p.add_argument(
        "--target",
        type=str,
        default=None,
        metavar="COL",
        help="Target column for TSTR downstream evaluation",
    )
    p.add_argument("--json", action="store_true", help="Also print JSON output")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Save JSON report to file",
    )
    p.add_argument(
        "--hif-hubs",
        type=int,
        default=5,
        help="Number of high-dependency hubs to audit",
    )
    p.add_argument(
        "--hif-depth", type=int, default=12, help="Max depth for sentinel forests"
    )
    p.add_argument(
        "--drop-cols",
        type=str,
        default=None,
        metavar="COLS",
        help="Comma-separated columns to drop from both real and synthetic before scoring",
    )
    p.add_argument(
        "--rule-min-confidence",
        type=float,
        default=0.95,
        metavar="F",
        help="Minimum confidence for mined logical rules (default: 0.95)",
    )
    p.add_argument(
        "--rule-min-support",
        type=float,
        default=0.005,
        metavar="F",
        help="Minimum support for mined logical rules (default: 0.005)",
    )
    p.add_argument(
        "--rule-max-rules",
        type=int,
        default=25,
        metavar="N",
        help="Maximum number of mined logical rules to keep (default: 25)",
    )
    p.add_argument(
        "--rule-min-lift",
        type=float,
        default=1.0,
        metavar="F",
        help="Minimum lift for mined logical rules (default: 1.0)",
    )
    p.add_argument(
        "--rule-max-antecedents",
        type=int,
        default=2,
        metavar="N",
        help="Maximum antecedent size for mined logical rules (default: 2)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable detailed debug logs"
    )
    p.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress outputs"
    )
    p.set_defaults(func=cmd_evaluate)

    # validate
    p = sub.add_parser("validate", help="Validate a real data file before fitting.")
    p.add_argument("file")
    p.add_argument("--null-threshold", type=float, default=0.3, metavar="F")
    p.add_argument("--dup-threshold", type=float, default=0.05, metavar="F")
    p.add_argument("--max-cardinality", type=int, default=500, metavar="N")
    p.add_argument("--min-rows", type=int, default=50, metavar="N")
    p.set_defaults(func=cmd_validate)

    # download
    p = sub.add_parser("download", help="Download real bulk data from public sources.")
    p.add_argument("dataset", help="Dataset ID, 'all', or 'status'")
    p.add_argument("--force", action="store_true", help="Re-download even if cached")
    p.add_argument(
        "--sample",
        type=int,
        default=50000,
        metavar="N",
        help="Max rows to cache (default: 50000)",
    )
    p.set_defaults(func=cmd_download)

    args = parser.parse_args()

    if getattr(args, "verbose", False):
        logging.basicConfig(
            level=logging.WARNING, format="  [%(levelname)s] %(message)s"
        )
        logging.getLogger("tabular_polygraph").setLevel(logging.DEBUG)
    elif getattr(args, "quiet", False):
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.WARNING)

    if not args.command:
        parser.print_help()
        print()
        dim("  Examples:")
        dim("    tabular-polygraph list --vertical 'Real Estate'")
        dim("    tabular-polygraph generate adult --rows 500 --output syn.csv")
        dim("    tabular-polygraph evaluate real.csv synthetic.csv")
        dim("    tabular-polygraph validate my_data.csv")
        print()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
