"""
Commands:
    list                        List datasets (with vertical filter)
    info <dataset>              Full dataset metadata
    generate <dataset>          Generate synthetic data
    evaluate <real> <syn>       Full fidelity report
    audit <real> <syn>          Full privacy audit
    scenario list               List built-in scenarios
    scenario apply <scenario>   Apply scenario to a CSV
    validate <file>             Validate a real data file
    download                    Fetch real-world datasets
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path


# Terminal colours


class C:
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


def _critical_drop_cols(dataset_id: str | None, drop_cols: list[str]) -> list[str]:
    """Return the subset of requested drops that are structural for the dataset."""
    if not dataset_id or not drop_cols:
        return []

    critical_by_dataset = {
        "world_bank": {"country_code", "year", "region", "income_group"},
    }
    critical = critical_by_dataset.get(dataset_id, set())
    return [c for c in drop_cols if c in critical]


def _resolve_generator_type(dataset_id: str, generator_type: str) -> str:
    if generator_type != "auto":
        return generator_type

    time_series_datasets = {"fred_macro", "bls"}
    panel_datasets = {"world_bank"}
    if dataset_id in time_series_datasets:
        return "var"
    if dataset_id in panel_datasets:
        return "panel"
    return "copula"


def _drop_existing_columns(df, drop_cols: list[str] | None):
    if not drop_cols:
        return df
    drop_present = [c for c in drop_cols if c in df.columns]
    if not drop_present:
        return df
    return df.drop(columns=drop_present)


def _infer_panel_columns(df):
    entity_col = next(
        (c for c in ["country_code", "bank_id", "entity"] if c in df.columns),
        df.columns[0],
    )
    if "year" in df.columns:
        time_col = "year"
    elif "quarter" in df.columns:
        time_col = "quarter"
    else:
        time_col = None
    return entity_col, time_col


def _load_generator(
    dataset_id,
    generator_type="auto",
    drop_cols: list[str] | None = None,
    fit_rows: int | None = None,
):
    """Load and fit a generator for the given dataset."""
    from src.catalog import load_dataset
    from src.catalog.downloader import load_cached
    from src.generators import GaussianCopulaGenerator
    from src.generators.time_series import VARGenerator
    from src.generators.panel import FixedEffectsGenerator

    generator_type = _resolve_generator_type(dataset_id, generator_type)
    fit_n = fit_rows
    if fit_n is not None and fit_n < 1:
        raise ValueError("--fit-rows must be >= 1")

    if fit_n is None:
        seed_df = load_cached(dataset_id)
        if seed_df is None or len(seed_df) < 100:
            raise ValueError(
                f"Dataset '{dataset_id}' not cached. Download real data first: python main.py download {dataset_id}"
            )
        seed_df = seed_df.reset_index(drop=True)
    else:
        seed_df = load_dataset(dataset_id, n=fit_n)
    seed_df = _drop_existing_columns(seed_df, drop_cols)

    if generator_type == "var":
        time_col = "year" if "year" in seed_df.columns else None
        gen = VARGenerator(lags=2, time_col=time_col)
    elif generator_type == "panel":
        entity_col, time_col = _infer_panel_columns(seed_df)
        gen = FixedEffectsGenerator(entity_col=entity_col, time_col=time_col)
    else:
        gen = GaussianCopulaGenerator()

    gen.fit(seed_df)
    return gen, seed_df, generator_type


def _load_eval_frames(real_path: str, syn_path: str):
    for p in [Path(real_path), Path(syn_path)]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    from src.io import read

    info(f"Loading real:      {real_path}")
    real = read(real_path)
    info(f"Loading synthetic: {syn_path}")
    syn = read(syn_path)
    return real, syn


def _apply_eval_drop_cols(real, syn, drop_cols_arg: str | None):
    drop_cols = _parse_drop_cols(drop_cols_arg)
    if not drop_cols:
        return real, syn

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
        "rule_max_antecedents": int(getattr(args, "rule_max_antecedents", 1)),
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


def _json_clean(obj):
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _prepare_generate_request(args):
    input_file = getattr(args, "input", None)
    dataset_id = getattr(args, "dataset", None)

    if not input_file and not dataset_id:
        err("Provide a dataset ID (e.g. python3 main.py generate fred_macro)")
        err("or your own file (e.g. python3 main.py generate --input data.csv)")
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
        dropped_critical = _critical_drop_cols(dataset_id, drop_cols)
        if dropped_critical:
            err(
                "Refusing to drop structural world_bank columns: "
                f"{', '.join(dropped_critical)}"
            )
            err(
                "Keep country_code, year, region, and income_group to preserve panel structure."
            )
            sys.exit(1)

    return input_file, dataset_id, filters, drop_cols


def _fit_custom_input_generator(input_file: str, drop_cols: list[str]):
    if not Path(input_file).exists():
        raise FileNotFoundError(input_file)

    from src.io import read, validate as validate_df
    from src.generators import GaussianCopulaGenerator

    seed_df = read(input_file)
    seed_df = _drop_existing_columns(seed_df, drop_cols)
    result = validate_df(seed_df, min_rows=10)
    if not result.passed:
        raise ValueError("\n".join(result.errors))
    if result.warnings:
        for w_msg in result.warnings:
            warn(w_msg)

    gen = GaussianCopulaGenerator()
    gen.fit(seed_df)
    info(
        f"Loaded {len(seed_df):,} rows × {len(seed_df.columns)} columns from {input_file}"
    )
    return gen, seed_df, "copula"


def _fit_generate_generator(input_file, dataset_id, args, drop_cols):
    t0 = time.time()
    print()
    info("Fitting generator...")
    try:
        if input_file:
            gen, seed_df, gen_type = _fit_custom_input_generator(input_file, drop_cols)
        else:
            gen, seed_df, gen_type = _load_generator(
                dataset_id,
                args.generator,
                drop_cols=drop_cols,
                fit_rows=getattr(args, "fit_rows", None),
            )
    except FileNotFoundError as e:
        err(f"File not found: {e}")
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        err(str(e))
        sys.exit(1)

    ok(f"{gen}  [{gen_type}]  ({time.time() - t0:.1f}s)")
    return gen, seed_df, gen_type


def _run_scenario_if_requested(syn, args):
    if not getattr(args, "scenario", None):
        return syn

    from src.calibration import apply_scenario

    info(f"Applying scenario: {args.scenario}  (intensity={args.intensity})")
    syn = apply_scenario(syn, args.scenario, intensity=args.intensity)
    ok("Scenario applied")
    return syn


def _run_calibration_if_requested(syn, seed_df, args):
    if not getattr(args, "calibrate", False):
        return syn

    from src.calibration import match_moments

    info("Running moment calibration...")
    syn_body = syn.drop(columns=["syn_id"], errors="ignore")
    syn_body = match_moments(seed_df, syn_body)
    syn_body.insert(0, "syn_id", syn["syn_id"])
    ok("Moments calibrated")
    return syn_body


def _compute_generate_report(seed_df, syn, gen_type, no_eval: bool):
    if no_eval:
        return None

    from src.fidelity import fidelity_report

    info("Running fidelity report...")
    dataset_type = {"var": "time_series", "panel": "panel"}.get(
        gen_type, "cross_sectional"
    )
    syn_body = syn.drop(columns=["syn_id"], errors="ignore")
    try:
        return fidelity_report(seed_df, syn_body, dataset_type=dataset_type)
    except Exception as fe:
        warn(f"Fidelity report skipped: {fe}")
        return None


def _print_generate_main_scores(report):
    s = report["summary"]

    section("Fidelity report")
    mm_cols = report.get("moment_matching", {}).get("column_scores", {})
    if mm_cols:
        print("    Moment matching")
        for col, score in mm_cols.items():
            print(
                f"    {col:<26}{bar(score)}  {_c(str(score) + '%', C.GREEN if score >= 90 else C.YELLOW)}"
            )
        print()

    ks_cols = report.get("distribution_fit", {}).get("column_scores", {})
    if ks_cols:
        print("    KS distribution")
        for col, score in ks_cols.items():
            print(
                f"    {col:<26}{bar(score)}  {_c(str(score) + '%', C.GREEN if score >= 90 else C.YELLOW)}"
            )
        print()
    print(
        f"    {_c('Overall fidelity:', C.GRAY):<34}{_c(str(s['overall_fidelity']) + '%', C.GREEN)}"
    )
    print(
        f"    {_c('Moment matching:', C.GRAY):<34}{_c(str(s['moment_matching_score']) + '%', C.GREEN)}"
    )
    print(
        f"    {_c('KS distribution:', C.GRAY):<34}{_c(str(s['ks_score']) + '%', C.GREEN)}"
    )
    print(
        f"    {_c('Joint score:', C.GRAY):<34}{_c(str(s['joint_score']) + '%', C.GREEN)}"
    )
    print(
        f"    {_c('Privacy score:', C.GRAY):<34}{_c(str(s['privacy_score']) + '%', C.GREEN)}"
    )
    print(f"    {_c('Exact copies:', C.GRAY):<34}{s['exact_copies']}")


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


def _print_generate_temporal(report):
    if "temporal" not in report:
        return

    t = report["temporal"]
    section("Temporal fidelity")
    print(
        f"    Stationarity agreement  {t['stationarity']['_summary']['agreement_rate']}%"
    )
    print(
        f"    Cointegration agreement {t['cointegration']['_summary']['agreement_rate']}%"
    )
    print(f"    Break match rate        {t['breaks']['_summary']['break_match_rate']}%")
    print(
        f"    Causality agreement     {t['causality']['_summary']['agreement_rate']}%"
    )


def _print_generate_logical(report):
    if "logical" not in report:
        return

    lg = report["logical"]
    section("Logical (Rules + LCV)")
    if "error" in lg:
        print(f"    {_c('Error:', C.RED)} {lg['error']}")
        return

    print(f"    LCV score               {lg.get('lcv_score_pct', '—')}%")
    print(f"    LCV violation rate      {lg.get('lcv_violation_rate_pct', '—')}%")
    print(f"    Rule violation rate     {lg.get('rule_violation_rate_pct', '—')}%")
    print(f"    Mean penalty            {lg.get('mean_penalty_pct', '—')}%")
    print(
        f"    Rule violations         {lg.get('num_rule_violations', '—')} (rules mined: {lg.get('num_rules_mined', '—')})"
    )


def _print_generate_report(report):
    if report is None:
        return

    _print_generate_main_scores(report)
    _print_generate_stylized(report)
    _print_generate_temporal(report)
    _print_generate_logical(report)


def _save_generated_output(syn, output_path: str):
    from src.io import write

    output = Path(output_path)
    write(syn, output)
    print()
    ok(f"Saved → {_c(str(output), C.CYAN)}  ({output.stat().st_size // 1024} KB)")
    print()


# Commands


def cmd_list(args):
    from src.catalog import list_datasets

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
    dim("  python3 main.py info <id>    full metadata + columns")
    dim("  python3 main.py generate <id>    generate synthetic data")
    print()


def cmd_info(args):
    from src.catalog import get_dataset_info

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
    input_file, dataset_id, filters, drop_cols = _prepare_generate_request(args)
    gen, seed_df, gen_type = _fit_generate_generator(
        input_file, dataset_id, args, drop_cols
    )

    # Generate
    info(f"Sampling {args.rows:,} rows...")
    t1 = time.time()
    syn = gen.generate(args.rows, filters=filters or None, seed=args.seed)
    ok(f"{len(syn):,} rows generated  ({time.time() - t1:.1f}s)")

    syn = _run_scenario_if_requested(syn, args)
    syn = _run_calibration_if_requested(syn, seed_df, args)

    report = _compute_generate_report(
        seed_df, syn, gen_type, no_eval=getattr(args, "no_eval", False)
    )
    _print_generate_report(report)

    _save_generated_output(syn, args.output)


def cmd_evaluate(args):
    from src.fidelity import fidelity_report, format_report

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

    info("Running full fidelity report...")
    report = fidelity_report(
        real,
        syn,
        dataset_type=dataset_type,
        target_col=target_col,
        include_downstream=bool(target_col),
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


def cmd_audit(args):
    header("Privacy audit", f"{args.real}  vs  {args.synthetic}")

    for p in [Path(args.real), Path(args.synthetic)]:
        if not p.exists():
            err(f"File not found: {p}")
            sys.exit(1)

    from src.io import read
    from src.privacy import privacy_audit, format_audit

    info(f"Loading real:      {args.real}")
    real = read(args.real)
    info(f"Loading synthetic: {args.synthetic}")
    syn = read(args.synthetic)
    info(f"Rows — real: {len(real):,}  synthetic: {len(syn):,}")
    info(f"Running {args.attacks} attacks per test...")
    print()

    report = privacy_audit(
        real,
        syn,
        n_attacks=args.attacks,
        seed=42,
    )

    print(format_audit(report))

    v = report["verdict"]
    section("Detailed results")
    ec = report["exact_copies"]
    mi = report["membership_inference"]
    so = report["singling_out"]
    lk = report["linkability"]

    print(f"    {'Exact copies':<28}{ec['count']}  [{risk_colour(ec['risk_level'])}]")
    print(
        f"    {'Membership inference AUC':<28}{mi.get('attack_auc', '—')}  [{risk_colour(mi.get('risk_level', '—'))}]"
    )
    print(f"      {_c(mi.get('interpretation', ''), C.GRAY)}")
    print(
        f"    {'Singling-out rate':<28}{so.get('singling_out_rate', '—')}  [{risk_colour(so.get('risk_level', '—'))}]"
    )
    print(
        f"    {'Linkability rate':<28}{lk.get('linkability_rate', '—')}  [{risk_colour(lk.get('risk_level', '—'))}]"
    )
    print(f"      lift {lk.get('lift_over_baseline_pct', '—')}% over baseline")
    print()
    print(f"    {_c('Recommendation:', C.BOLD)}")
    print(f"    {v['recommendation']}")

    if getattr(args, "json", False):
        import json as _json

        print(_json.dumps(report, indent=2, default=str))

    if getattr(args, "output", None):
        import json as _json

        Path(args.output).write_text(_json.dumps(report, indent=2, default=str))
        ok(f"Audit saved → {args.output}")
    print()


def cmd_scenario_list(args):
    from src.calibration import list_scenarios

    header("Built-in scenarios")
    df = list_scenarios()
    for _, row in df.iterrows():
        print(f"\n    {_c(row['name'], C.BOLD)}")
        print(f"    {_c(row['description'], C.GRAY)}")
        print(f"    {row['columns_affected']} columns affected")
    print()
    dim(
        "  Usage: python3 main.py scenario apply <name> --input data.csv --output stressed.csv"
    )
    print()


def cmd_scenario_apply(args):
    from src.calibration import apply_scenario
    from src.io import read, write

    header(f"Applying scenario: {args.scenario}", f"intensity={args.intensity}")

    if not Path(args.input).exists():
        err(f"File not found: {args.input}")
        sys.exit(1)

    info(f"Loading: {args.input}")
    df = read(args.input)
    info(f"Rows: {len(df):,}  columns: {len(df.columns)}")

    try:
        result = apply_scenario(df, args.scenario, intensity=args.intensity)
    except ValueError as e:
        err(str(e))
        sys.exit(1)

    output = Path(args.output)
    write(result, output)
    ok(f"Saved → {_c(str(output), C.CYAN)}")

    # Show which columns shifted
    num_cols = [
        c for c in df.columns if df[c].dtype.kind in "if" and c in result.columns
    ]
    if num_cols:
        section("Column shifts")
        for col in num_cols[:8]:
            before = df[col].mean()
            after = result[col].mean()
            delta = after - before
            sign = "+" if delta >= 0 else ""
            print(
                f"    {col:<28}  {before:>10.2f} → {after:>10.2f}  ({sign}{delta:.2f})"
            )
    print()


def cmd_validate(args):
    from src.io import read, validate

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
    from src.catalog.downloader import download, status, DOWNLOADERS, is_cached

    if args.dataset == "status":
        header("Download status")
        df = status()
        for _, row in df.iterrows():
            icon = _c("✓", C.GREEN) if "cached" in row["status"] else _c("○", C.GRAY)
            print(f"    {icon} {row['dataset']:<28} {row['rows']:<12} {row['size']}")
        print()
        dim("  Run: python3 main.py download <id>   to download real data")
        dim("  Run: python3 main.py download all    to download everything")
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

    parser = argparse.ArgumentParser(prog="python3 main.py", add_help=False)
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
        help="Built-in dataset ID (e.g. fred_macro). Omit if using --input.",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to your own CSV. Fit the generator on it instead of a built-in dataset.",
    )
    p.add_argument("--rows", type=int, default=1000, metavar="N")
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
        metavar="TYPE",
        help="auto | copula | var | panel",
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
        "--scenario",
        type=str,
        default=None,
        metavar="NAME",
        help="Apply built-in scenario: recession | rate_shock | credit_crisis ...",
    )
    p.add_argument(
        "--intensity",
        type=float,
        default=1.0,
        metavar="F",
        help="Scenario intensity 0.0–1.0 (default: 1.0)",
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Run moment calibration after generation",
    )
    p.add_argument("--seed", type=int, default=None, metavar="INT")
    p.add_argument("--no-eval", action="store_true", help="Skip fidelity evaluation")
    p.add_argument(
        "--drop-cols",
        type=str,
        default=None,
        metavar="COLS",
        help="Comma-separated columns to drop before fit/eval, e.g. tract_id,customer_id",
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
        help="cross_sectional | time_series | panel",
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
        default=1,
        metavar="N",
        help="Maximum antecedent size for mined logical rules (default: 1)",
    )
    p.set_defaults(func=cmd_evaluate)

    # audit
    p = sub.add_parser("audit", help="Full privacy audit.")
    p.add_argument("real")
    p.add_argument("synthetic")
    p.add_argument(
        "--attacks",
        type=int,
        default=300,
        metavar="N",
        help="Number of attack attempts per test (default: 300)",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=str, default=None, metavar="FILE")
    p.set_defaults(func=cmd_audit)

    # scenario
    p_sc = sub.add_parser("scenario", help="Scenario management.")
    sc_sub = p_sc.add_subparsers(dest="scenario_cmd", metavar="<subcommand>")

    p_scl = sc_sub.add_parser("list", help="List built-in scenarios.")
    p_scl.set_defaults(func=cmd_scenario_list)

    p_sca = sc_sub.add_parser("apply", help="Apply a scenario to a CSV.")
    p_sca.add_argument("scenario")
    p_sca.add_argument("--input", type=str, required=True, metavar="FILE")
    p_sca.add_argument("--output", type=str, required=True, metavar="FILE")
    p_sca.add_argument("--intensity", type=float, default=1.0, metavar="F")
    p_sca.set_defaults(func=cmd_scenario_apply)

    def _scenario_help(_args):
        p_sc.print_help()
        print()

    p_sc.set_defaults(func=_scenario_help)

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

    if not args.command:
        parser.print_help()
        print()
        dim("  Examples:")
        dim("    python3 main.py list")
        dim("    python3 main.py generate fred_macro --rows 5000 --scenario recession")
        dim("    python3 main.py generate world_bank --rows 3000 --generator panel")
        dim(
            "    python3 main.py evaluate real.csv synthetic.csv --type time_series --target gdp_growth_yoy"
        )
        dim("    python3 main.py audit real.csv synthetic.csv --attacks 500")
        dim("    python3 main.py scenario list")
        dim(
            "    python3 main.py scenario apply recession --input syn.csv --output stressed.csv"
        )
        dim("    python3 main.py validate my_data.csv")
        print()
        sys.exit(0)

    # Handle scenario sub-subcommand routing
    if args.command == "scenario" and not getattr(args, "scenario_cmd", None):
        cmd_scenario_list(args)
        return

    args.func(args)


if __name__ == "__main__":
    main()
