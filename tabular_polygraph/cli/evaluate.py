"""Evaluate command — fidelity report between two CSV files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tabular_polygraph._utils import set_seed

from .utils import _json_clean, err, header, info, ok


def cmd_evaluate(args):
    from tabular_polygraph.fidelity import fidelity_report, format_report

    from .helpers import (
        _apply_eval_drop_cols,
        _load_eval_frames,
        _rule_params_from_args,
    )

    set_seed(args.seed)
    header("Fidelity evaluation", f"{args.real}  vs  {args.synthetic}")

    try:
        real, syn = _load_eval_frames(args.real, args.synthetic)
    except FileNotFoundError as e:
        err(f"File not found: {e}")
        sys.exit(1)

    real, syn = _apply_eval_drop_cols(real, syn, getattr(args, "drop_cols", None))

    info(f"Rows — real: {len(real):,}  synthetic: {len(syn):,}")

    dataset_type = getattr(args, "type", "cross_sectional") or "cross_sectional"
    target_col = getattr(args, "target", None)

    try:
        rule_params = _rule_params_from_args(args)
    except ValueError as e:
        err(str(e))
        sys.exit(1)

    info("Running Hybrid Integrity Framework: The Tabular Polygraph...")
    report = fidelity_report(
        real,
        syn,
        dataset_type=dataset_type,
        target_col=target_col,
        include_downstream=bool(target_col),
        hif_epochs=getattr(args, "hif_epochs", 10),
        hif_hubs=getattr(args, "hif_hubs", 5),
        hif_depth=getattr(args, "hif_depth", 12),
        random_state=args.seed,
        verbose=getattr(args, "verbose", False),
        include_privacy=getattr(args, "privacy", False),
        **rule_params,
    )

    print(format_report(report))

    if getattr(args, "json", False):
        print(json.dumps(_json_clean(report), indent=2))

    if getattr(args, "output", None):
        Path(args.output).write_text(json.dumps(report, indent=2, default=str))
        ok(f"Report saved → {args.output}")


__all__ = [
    "cmd_evaluate",
]
