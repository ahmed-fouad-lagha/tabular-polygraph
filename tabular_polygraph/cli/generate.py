"""Generate command and report printing helpers."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

from tabular_polygraph.utils import set_seed
from .utils import C, _c, bar, err, header, info, ok, section, warn


def _prepare_generate_request(args):
    input_file = getattr(args, "input", None)
    dataset_id = getattr(args, "dataset", None)

    if not input_file and not dataset_id:
        err("Provide a dataset ID (e.g. tabular-polygraph generate adult)")
        err("or your own file (e.g. tabular-polygraph generate --input data.csv)")
        sys.exit(1)

    if input_file:
        header(f"Generating from: {input_file}", f"rows={args.rows:,}")
    else:
        header(
            f"Generating: {dataset_id}",
            f"generator={args.generator}  rows={args.rows:,}",
        )

    from .helpers import _parse_drop_cols, _parse_filters

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

    from tabular_polygraph.io import read, validate as validate_df
    from tabular_polygraph.utils import DEFAULT_DROP_LIST
    from .helpers import _create_generator_instance, _drop_existing_columns

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
                verbose=not getattr(args, "quiet", False),
            )
        else:
            from .helpers import _load_generator

            gen_kwargs: dict[str, Any] = {"verbose": not getattr(args, "quiet", False)}
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
        logging.debug("Fidelity report failure traceback:", exc_info=True)
        return None


def _print_generate_bars(report):
    section("Marginal Fidelity")

    mm_cols = report.get("moment_matching", {}).get("column_scores", {})
    if mm_cols:
        print(f"    {_c('Continuous (Moment matching)', C.GRAY)}")
        for col, score in mm_cols.items():
            print(
                f"    {col:<26}{bar(score)}  {_c(str(score) + '%', C.GREEN if score >= 90 else C.YELLOW)}"
            )
        print()

    ks_cols = report.get("distribution_fit", {}).get("column_scores", {})
    if ks_cols:
        print(f"    {_c('Continuous (KS distribution)', C.GRAY)}")
        for col, score in ks_cols.items():
            print(
                f"    {col:<26}{bar(score)}  {_c(str(score) + '%', C.GREEN if score >= 90 else C.YELLOW)}"
            )
        print()

    tvd_cols = report.get("categorical_tvd", {}).get("column_scores", {})
    if tvd_cols:
        print(f"    {_c('Categorical (TVD)', C.GRAY)}")
        for col, score in tvd_cols.items():
            print(
                f"    {col:<26}{bar(score)}  {_c(str(score) + '%', C.GREEN if score >= 90 else C.YELLOW)}"
            )
        print()

    ap = report.get("coverage", {}).get("alpha_precision")
    br = report.get("coverage", {}).get("beta_recall")
    au = report.get("coverage", {}).get("authenticity")
    if ap is not None and br is not None and au is not None:
        section("Multidimensional Coverage")
        print(f"    {_c('Alpha-precision', C.GRAY):<34}{ap:.3f}")
        print(f"    {_c('Beta-recall', C.GRAY):<34}{br:.3f}")
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
        f"    {_c('Unified violation rate ', C.GRAY):<28}{lg.get('hif_violation_rate_pct', '—')}%  ({lg.get('num_hif_violations', '—')} total violations)"
    )
    print(
        f"    {_c('LSE (Categorical) rate ', C.GRAY):<28}{lg.get('lse_violation_rate_pct', '—')}%"
    )
    print(
        f"    {_c('NIC (Continuous) rate  ', C.GRAY):<28}{lg.get('nic_violation_rate_pct', '—')}%"
    )
    print(
        f"    {_c('Rule violation rate    ', C.GRAY):<28}{lg.get('rule_violation_rate_pct', '—')}%  ({lg.get('num_rule_violations', '—')} violations / {lg.get('num_rules_mined', '—')} rules mined)"
    )
    print(
        f"    {_c('Noise floor threshold  ', C.GRAY):<28}{lg.get('violation_threshold', '—')}"
    )


def _print_generate_joint(report):
    jt = report.get("joint")
    if not jt or "error" in jt:
        return
    section("Joint Fidelity")
    score = jt.get("correlation_distance_score", 0.0)
    print(f"    {_c('Correlation Distance Score ', C.GRAY):<34}{score:.2f}%")
    print()


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
    _print_generate_joint(report)
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


def cmd_generate(args):
    set_seed(args.seed)
    input_file, dataset_id, filters, drop_cols = _prepare_generate_request(args)
    gen, seed_df, gen_type = _fit_generate_generator(
        input_file, dataset_id, args, drop_cols
    )

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


__all__ = [
    "cmd_generate",
]
