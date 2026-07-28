"""Shared CLI helpers — parsing, generator instantiation, rule params."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tabular_polygraph.generators import BaseGenerator


def _parse_filters(filter_args):
    from .utils import warn

    filters = {}
    if not filter_args:
        return filters
    for f in filter_args:
        if ":" not in f:
            warn(f"Skipping malformed filter '{f}' — use key:value")
            continue
        key, val = f.split(":", 1)
        key = key.strip().lower().replace("-", "_")
        if key.endswith("_min") or key.endswith("_max"):
            try:
                filters[key] = float(val)
            except ValueError:
                warn(f"Filter '{key}' expects a number, got '{val}'")
        else:
            filters[key] = val.split(",") if "," in val else val
    return filters


def _parse_drop_cols(drop_cols_arg: str | None) -> list[str]:
    if not drop_cols_arg:
        return []
    cols = [c.strip() for c in drop_cols_arg.split(",") if c.strip()]
    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _resolve_generator_type(dataset_id: str, generator_type: str) -> str:
    if generator_type != "auto":
        return generator_type
    return "copula"


def _drop_existing_columns(df, drop_cols: list[str] | None):
    if not drop_cols:
        return df
    drop_present = [c for c in drop_cols if c in df.columns]
    if not drop_present:
        return df
    return df.drop(columns=drop_present)


def _positive_int(val: str) -> int:
    ival = int(val)
    if ival <= 0:
        raise argparse.ArgumentTypeError(f"Must be a positive integer > 0, got {val}")
    return ival


def _create_generator_instance(generator_type: str, **kwargs) -> BaseGenerator:
    if generator_type == "ctgan":
        from tabular_polygraph.generators import CTGANGenerator
        return CTGANGenerator(**kwargs)
    elif generator_type == "tvae":
        from tabular_polygraph.generators import TVAEGenerator
        return TVAEGenerator(**kwargs)
    elif generator_type == "vine":
        from tabular_polygraph.generators import VineCopulaGenerator
        return VineCopulaGenerator(**kwargs)
    else:
        from tabular_polygraph.generators import GaussianCopulaGenerator
        return GaussianCopulaGenerator(**kwargs)


def _load_generator(
    dataset_id,
    generator_type="auto",
    drop_cols: list[str] | None = None,
    fit_rows: int | None = None,
    **kwargs,
):
    from tabular_polygraph.dataset import load_dataset
    from tabular_polygraph.dataset.loader import load_cached
    from tabular_polygraph.utils import DEFAULT_DROP_LIST
    from .utils import info

    generator_type = _resolve_generator_type(dataset_id, generator_type)
    fit_n = fit_rows
    if fit_n is not None and fit_n < 1:
        raise ValueError("--fit-rows must be >= 1")

    if fit_n is None:
        seed_df = load_cached(dataset_id)
        if seed_df is None:
            raise ValueError(
                f"Dataset '{dataset_id}' not found in cache.\n"
                f"Run: tabular-polygraph download {dataset_id}"
            )
        if len(seed_df) < 10:
            raise ValueError(
                f"Dataset '{dataset_id}' is too small to fit a generator ({len(seed_df)} rows).\n"
                "Please download a larger sample using: tabular-polygraph download --force --sample 1000"
            )
        seed_df = seed_df.reset_index(drop=True)
    else:
        seed_df = load_dataset(dataset_id, n=fit_n)

    all_drop = list(set((drop_cols or []) + list(DEFAULT_DROP_LIST)))
    seed_df = _drop_existing_columns(seed_df, all_drop)

    if any(c for c in all_drop if c not in DEFAULT_DROP_LIST):
        info(f"Dropping columns before fit/eval: {all_drop}")

    gen = _create_generator_instance(generator_type, **kwargs)
    gen.fit(seed_df)
    return gen, seed_df, generator_type


def _load_eval_frames(real_path: str, syn_path: str):
    from tabular_polygraph.io import read
    from .utils import info

    for p in [Path(real_path), Path(syn_path)]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    info(f"Loading real:      {real_path}")
    real = read(real_path)
    info(f"Loading synthetic: {syn_path}")
    syn = read(syn_path)
    return real, syn


def _apply_eval_drop_cols(real, syn, drop_cols_arg: str | None):
    from tabular_polygraph.utils import DEFAULT_DROP_LIST
    from .utils import info

    drop_cols = _parse_drop_cols(drop_cols_arg) or []
    drop_cols = list(set(drop_cols + list(DEFAULT_DROP_LIST)))

    real_drop = [c for c in drop_cols if c in real.columns]
    syn_drop = [c for c in drop_cols if c in syn.columns]
    if real_drop:
        real = real.drop(columns=real_drop)
    if syn_drop:
        syn = syn.drop(columns=syn_drop)
    info(f"Dropped columns before evaluation: {drop_cols}")
    return real, syn


def _rule_params_from_args(args) -> dict:
    params = {
        "rule_min_confidence": float(getattr(args, "rule_min_confidence", 0.95)),
        "rule_min_support": float(getattr(args, "rule_min_support", 0.005)),
        "rule_max_rules": int(getattr(args, "rule_max_rules", 25)),
        "rule_min_lift": float(getattr(args, "rule_min_lift", 1.0)),
        "rule_max_antecedents": int(getattr(args, "rule_max_antecedents", 2)),
    }

    if not (0.0 <= params["rule_min_confidence"] <= 1.0):
        raise ValueError("--rule-min-confidence must be between 0 and 1")
    if not (0.0 <= params["rule_min_support"] <= 1.0):
        raise ValueError("--rule-min-support must be between 0 and 1")
    if params["rule_max_rules"] < 1:
        raise ValueError("--rule-max-rules must be >= 1")
    if params["rule_min_lift"] < 0.0:
        raise ValueError("--rule-min-lift must be >= 0")
    if params["rule_max_antecedents"] < 1:
        raise ValueError("--rule-max-antecedents must be >= 1")

    return params


__all__ = [
    "_parse_filters",
    "_parse_drop_cols",
    "_resolve_generator_type",
    "_drop_existing_columns",
    "_positive_int",
    "_create_generator_instance",
    "_load_generator",
    "_load_eval_frames",
    "_apply_eval_drop_cols",
    "_rule_params_from_args",
]
