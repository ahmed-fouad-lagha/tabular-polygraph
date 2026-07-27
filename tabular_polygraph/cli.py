"""
Commands:
    list                        List datasets (with vertical filter)
    info <dataset>              Full dataset metadata
    generate <dataset>          Generate synthetic data
    evaluate <real> <syn>       Full fidelity report
    validate <file>             Validate a real data file
    download                    Fetch real-world datasets
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tabular_polygraph.utils import set_seed

if TYPE_CHECKING:
    from tabular_polygraph.generators import BaseGenerator


class C:
    """Helper class for terminal colours."""

    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def _c(t, c):
    return f"{c}{t}{C.RESET}"


def ok(m):
    print(_c("  ✓ ", C.GREEN) + m)


def info(m):
    print(_c("  → ", C.CYAN) + m)


def warn(m):
    print(_c("  ! ", C.YELLOW) + m)


def err(m):
    print(_c("  ✗ ", C.RED) + m, file=sys.stderr)


def dim(m):
    print(_c(m, C.GRAY))


def header(title, sub=""):
    print()
    print(_c("  " + title, C.BOLD))
    if sub:
        print(_c("  " + sub, C.GRAY))
    print(_c("  " + "─" * max(len(title), len(sub)), C.GRAY))


def section(title):
    print()
    print(_c("  ┌─ " + title, C.CYAN))


def bar(score, width=22):
    score = float(score)
    if not np.isfinite(score):
        score = 0.0
    score = max(0.0, min(100.0, score))
    filled = int(score / 100 * width)
    if score >= 90:
        col = C.GREEN
    elif score >= 75:
        col = C.YELLOW
    else:
        col = C.RED
    return _c("█" * filled, col) + _c("░" * (width - filled), C.GRAY)


def risk_colour(level):
    return {
        "very_low": _c(level, C.GREEN),
        "low": _c(level, C.GREEN),
        "medium": _c(level, C.YELLOW),
        "high": _c(level, C.RED),
        "very_high": _c(level, C.RED),
    }.get(level, level)


# Helpers


def _parse_filters(filter_args):
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
    # Preserve order while removing duplicates.
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


def _json_clean(obj):
    import numpy as _np

    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _create_generator_instance(generator_type: str, **kwargs) -> BaseGenerator:
    """Instantiate a generator of the given type with kwargs."""
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
    """Load and fit a generator for the given dataset."""
    from tabular_polygraph.dataset import load_dataset
    from tabular_polygraph.dataset.downloader import load_cached
    from tabular_polygraph.utils import DEFAULT_DROP_LIST

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

    # Merge user drop_cols with DEFAULT_DROP_LIST
    all_drop = list(set((drop_cols or []) + list(DEFAULT_DROP_LIST)))
    seed_df = _drop_existing_columns(seed_df, all_drop)

    # Log what we dropped if it's more than just the defaults
    if any(c for c in all_drop if c not in DEFAULT_DROP_LIST):
        info(f"Dropping columns before fit/eval: {all_drop}")
    elif all_drop:
        # Just generic info that we've cleaned the dataset
        pass

    gen = _create_generator_instance(generator_type, **kwargs)
    gen.fit(seed_df)
    return gen, seed_df, generator_type


def _load_eval_frames(real_path: str, syn_path: str):
    for p in [Path(real_path), Path(syn_path)]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    from tabular_polygraph.io import read

    info(f"Loading real:      {real_path}")
    real = read(real_path)
    info(f"Loading synthetic: {syn_path}")
    syn = read(syn_path)
    return real, syn


def _apply_eval_drop_cols(real, syn, drop_cols_arg: str | None):
    from tabular_polygraph.utils import DEFAULT_DROP_LIST

    drop_cols = _parse_drop_cols(drop_cols_arg) or []
    # Automatically add default blocklist
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


def _prepare_generate_request(args):
    input_file = getattr(args, "input", None)
    dataset_id = getattr(args, "dataset", None)

    if not input_file and not dataset_id:
        err("Provide a dataset ID (e.g. tabular-polygraph generate bls)")
        err("or your own file (e.g. tabular-polygraph generate --input data.csv)")
        sys.exit(1)

    if input_file:
        header(f"Generating from: {input_file}", f"rows={args.rows:,}")
    else:
        header(
            f"Generating: {dataset_id}",
            f"generator={args.generator}  rows={args.rows:,}",
        )

    filters = _parse_filters(getattr(args, "filter", None))
    drop_cols = _parse_drop_cols(getattr(args, "drop_cols", None))
    if filters:
        info(f"Filters: {filters}")
    if drop_cols:
        info(f"Dropping columns before fit/eval: {drop_cols}")

    return input_file, dataset_id, filters, drop_cols


def _fit_custom_input_generator(
    input_file: str,
    drop_cols: list[str],
    generator_type: str = "copula",
    epochs: int | None = None,
    verbose: bool = False,
):
    if not Path(input_file).exists():
        raise FileNotFoundError(input_file)

    from tabular_polygraph.io import read
    from tabular_polygraph.io import validate as validate_df
    from tabular_polygraph.utils import DEFAULT_DROP_LIST

    seed_df = read(input_file)
    all_drop = list(set((drop_cols or []) + list(DEFAULT_DROP_LIST)))
    seed_df = _drop_existing_columns(seed_df, all_drop)
    result = validate_df(seed_df, min_rows=10)
    if not result.passed:
        raise ValueError("\n".join(result.errors))
    if result.warnings:
        for w_msg in result.warnings:
            warn(w_msg)

    gen_kwargs: dict[str, Any] = {"verbose": verbose}
    if epochs is not None:
        gen_kwargs["epochs"] = epochs

    gen = _create_generator_instance(generator_type, **gen_kwargs)
    gen.fit(seed_df)
    info(
        f"Loaded {len(seed_df):,} rows × {len(seed_df.columns)} columns from {input_file}"
    )
    return gen, seed_df, generator_type


def _fit_generate_generator(input_file, dataset_id, args, drop_cols):
    t0 = time.time()
    print()
    info("Fitting generator...")
    try:
        if input_file:
            gen, seed_df, gen_type = _fit_custom_input_generator(
                input_file,
                drop_cols,
                generator_type=getattr(args, "generator", "copula"),
                epochs=getattr(args, "epochs", None),
                verbose=getattr(args, "verbose", False),
            )
        else:
            gen_kwargs: dict[str, Any] = {"verbose": getattr(args, "verbose", False)}
            if getattr(args, "epochs", None) is not None:
                gen_kwargs["epochs"] = args.epochs
            gen, seed_df, gen_type = _load_generator(
                dataset_id,
                args.generator,
                drop_cols=drop_cols,
                fit_rows=getattr(args, "fit_rows", None),
                **gen_kwargs,
            )
    except FileNotFoundError as e:
        err(f"File not found: {e}")
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        err(str(e))
        sys.exit(1)

    ok(f"{gen}  [{gen_type}]  ({time.time() - t0:.1f}s)")
    return gen, seed_df, gen_type


def _compute_generate_report(
    seed_df,
    syn,
    gen_type,
    seed=42,
    hif_epochs=10,
    hif_hubs=5,
    hif_depth=12,
    rule_params=None,
    verbose=False,
    target_col=None,
):
    from tabular_polygraph.fidelity import fidelity_report

    info("Running fidelity report...")
    dataset_type = "cross_sectional"
    syn_body = syn.drop(columns=["syn_id"], errors="ignore")
    rp = rule_params or {}
    try:
        return fidelity_report(
            seed_df,
            syn_body,
            dataset_type=dataset_type,
            target_col=target_col,
            include_downstream=bool(target_col),
            hif_epochs=hif_epochs,
            hif_hubs=hif_hubs,
            hif_depth=hif_depth,
            random_state=seed,
            verbose=verbose,
            rule_min_confidence=rp.get("rule_min_confidence", 0.95),
            rule_min_support=rp.get("rule_min_support", 0.005),
            rule_max_rules=rp.get("rule_max_rules", 25),
            rule_min_lift=rp.get("rule_min_lift", 1.0),
            rule_max_antecedents=rp.get("rule_max_antecedents", 2),
        )
    except Exception as fe:
        warn(f"Fidelity report skipped: {fe}")
        import logging

        logging.debug("Fidelity report failure traceback:", exc_info=True)
        return None


def _print_generate_bars(report):
    section("Marginal Fidelity")

    mm_cols = report.get("moment_matching", {}).get("column_scores", {})
    if mm_cols:
        print(f"    {_c('Moment matching', C.GRAY)}")
        for col, score in mm_cols.items():
            print(
                f"    {col:<26}{bar(score)}  {_c(str(score) + '%', C.GREEN if score >= 90 else C.YELLOW)}"
            )
        print()

    ks_cols = report.get("distribution_fit", {}).get("column_scores", {})
    if ks_cols:
        print(f"    {_c('KS distribution', C.GRAY)}")
        for col, score in ks_cols.items():
            print(
                f"    {col:<26}{bar(score)}  {_c(str(score) + '%', C.GREEN if score >= 90 else C.YELLOW)}"
            )
        print()

    ap = report.get("coverage", {}).get("alpha_precision")
    br = report.get("coverage", {}).get("beta_recall")
    au = report.get("coverage", {}).get("authenticity")
    if ap is not None and br is not None and au is not None:
        print(f"    {_c('Alpha-precision (coverage)', C.GRAY):<34}{ap:.3f}")
        print(f"    {_c('Beta-recall (coverage)', C.GRAY):<34}{br:.3f}")
        print(f"    {_c('Authenticity', C.GRAY):<34}{au:.3f}")
        print()


def _print_generate_stylized(report):
    sf_summary = report.get("stylized_facts", {}).get("_summary", {})
    print()
    section("Stylized facts")
    if sf_summary.get("applicable", True):
        print(
            f"    {_c('Mean score:', C.GRAY):<34}{_c(str(sf_summary.get('mean_score')) + '%', C.GREEN)}"
        )
        print(
            f"    {_c('Columns tested:', C.GRAY):<34}{sf_summary.get('columns_tested', 0)}"
        )
        for col, item in report.get("stylized_facts", {}).items():
            if col == "_summary":
                continue
            print(f"    {col:<26}{item.get('score', '—')}%")
    else:
        print(f"    {sf_summary.get('note', 'Not evaluated.')}")


def _print_generate_logical(report):
    if "logical" not in report:
        return

    lg = report["logical"]
    section("Logical")
    if "error" in lg:
        print(f"    {_c('Error:', C.RED)} {lg['error']}")
        return

    print(
        f"    {_c('Unified violation rate ', C.GRAY):<28}{lg.get('hif_violation_rate_pct', '—')}%"
    )
    print(
        f"    {_c('NIC (Continuous) rate  ', C.GRAY):<28}{lg.get('nic_violation_rate_pct', '—')}%"
    )
    print(
        f"    {_c('Rule violation rate    ', C.GRAY):<28}{lg.get('rule_violation_rate_pct', '—')}%"
    )
    print(
        f"    {_c('Noise floor threshold  ', C.GRAY):<28}{lg.get('violation_threshold', '—')}"
    )
    print(
        f"    {_c('Violations found       ', C.GRAY):<28}{lg.get('num_hif_violations', '—')} (rules mined: {lg.get('num_rules_mined', '—')})"
    )


def _print_generate_downstream(report):
    ds = report.get("downstream")
    if not ds or "error" in ds or ds.get("status") == "skipped":
        return
    section("Downstream ML Utility (TSTR)")
    task = ds.get("task", "")
    metric = ds.get("metric", "")
    trr = ds.get("trr_score")
    tstr = ds.get("tstr_score")
    ratio = ds.get("ratio")
    target_col = ds.get("target_col", "")

    if trr is not None and tstr is not None:
        print(f"    {_c('Target column          ', C.GRAY):<28}{target_col} ({task})")
        print(f"    {_c('TRTR Real baseline     ', C.GRAY):<28}{trr:.4f} ({metric})")
        print(f"    {_c('TSTR Synthetic trained ', C.GRAY):<28}{tstr:.4f} ({metric})")
        ret_pct = ratio * 100.0 if ratio is not None else 0.0
        c_fmt = C.GREEN if ret_pct >= 90 else C.YELLOW
        print(
            f"    {_c('ML Retention ratio     ', C.GRAY):<28}{_c(f'{ret_pct:.1f}%', c_fmt)}"
        )
        print()


def _print_generate_report(report):
    if report is None:
        return

    _print_generate_bars(report)
    _print_generate_logical(report)
    _print_generate_downstream(report)
    _print_generate_stylized(report)


def _save_generated_output(syn, output_path: str):
    from tabular_polygraph.io import write

    final_path = write(syn, output_path)
    print()
    ok(
        f"Saved → {_c(str(final_path), C.CYAN)}  ({final_path.stat().st_size // 1024} KB)"
    )
    print()


# Commands


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
        err(str(e))
        sys.exit(1)

    header(f"Dataset: {args.dataset}", meta["name"])

    pairs = [
        ("Vertical", meta["vertical"]),
        ("Source", meta["source"]),
        ("Columns", str(meta["col_count"])),
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


def cmd_generate(args):
    set_seed(args.seed)
    input_file, dataset_id, filters, drop_cols = _prepare_generate_request(args)
    gen, seed_df, gen_type = _fit_generate_generator(
        input_file, dataset_id, args, drop_cols
    )

    # Generate
    info(f"Sampling {args.rows:,} rows...")
    t1 = time.time()
    syn = gen.generate(args.rows, filters=filters or None, seed=args.seed)
    ok(f"{len(syn):,} rows generated  ({time.time() - t1:.1f}s)")

    report = _compute_generate_report(
        seed_df,
        syn,
        gen_type,
        seed=args.seed,
        hif_epochs=getattr(args, "hif_epochs", 10),
        hif_hubs=getattr(args, "hif_hubs", 5),
        hif_depth=getattr(args, "hif_depth", 12),
        target_col=getattr(args, "target", None),
        verbose=getattr(args, "verbose", False),
    )
    _print_generate_report(report)

    _save_generated_output(syn, args.output)


def cmd_evaluate(args):
    set_seed(args.seed)
    from tabular_polygraph.fidelity import fidelity_report, format_report

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
        **rule_params,
    )

    print(format_report(report))

    if getattr(args, "json", False):
        import json as _json

        print(_json.dumps(_json_clean(report), indent=2))

    if getattr(args, "output", None):
        import json as _json

        Path(args.output).write_text(_json.dumps(report, indent=2, default=str))
        ok(f"Report saved → {args.output}")


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


def cmd_download(args):
    from tabular_polygraph.dataset.downloader import (
        DOWNLOADERS,
        download,
        is_cached,
        status,
    )

    if args.dataset == "status":
        header("Download status")
        df = status()
        for _, row in df.iterrows():
            icon = _c("✓", C.GREEN) if "cached" in row["status"] else _c("○", C.GRAY)
            print(f"    {icon} {row['dataset']:<28} {row['rows']:<12} {row['size']}")
        print()
        dim("  Run: tabular-polygraph download <id>   to download real data")
        dim("  Run: tabular-polygraph download all    to download everything")
        print()
        return

    dataset_id = args.dataset

    if dataset_id != "all" and dataset_id not in DOWNLOADERS:
        err(f"No downloader for '{dataset_id}'.")
        info(f"Available: {', '.join(DOWNLOADERS)}")
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


# Arg parser


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
        help="Built-in dataset ID (e.g. bls). Omit if using --input.",
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

    import logging

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
        dim("    tabular-polygraph generate bls --rows 500 --output syn.csv")
        dim("    tabular-polygraph evaluate real.csv synthetic.csv")
        dim("    tabular-polygraph validate my_data.csv")
        print()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
