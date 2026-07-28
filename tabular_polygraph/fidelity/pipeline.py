from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from tabular_polygraph._config import FidelityConfig
from tabular_polygraph._types import FidelityReport, Summary, PerColumnScore, Metric
from tabular_polygraph._utils import DEFAULT_DROP_LIST, numeric_columns, categorical_columns
from . import metrics as _metrics


logger = logging.getLogger(__name__)


def _shared_columns(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str] | None
) -> list[str]:
    drop_lower = {s.lower() for s in DEFAULT_DROP_LIST}
    return columns or [
        c
        for c in real.columns
        if c in synthetic.columns and c.lower() not in drop_lower
    ]


def _resolve_columns(real: pd.DataFrame, column_types: set[str]) -> list[str]:
    result: list[str] = []
    if "numeric" in column_types or "all" in column_types:
        result.extend(numeric_columns(real))
    if "categorical" in column_types or "all" in column_types:
        result.extend(c for c in categorical_columns(real) if c not in result)
    return result


_OLD_TO_NEW = {
    "moment_matching": "moment_matching",
    "ks_test": "distribution_fit",
    "tvd": "categorical_tvd",
    "correlation": "joint",
    "alpha_beta": "coverage",
    "stylized_facts": "stylized_facts",
    "downstream": "downstream",
    "hif": "logical",
}


def _build_summary(report: FidelityReport) -> Summary:
    mm = report.moment_matching
    ks = report.distribution_fit
    jt = report.joint
    cv = report.coverage
    lg = report.logical
    sf = report.stylized_facts
    ds = report.downstream

    stylized_mean = sf.mean_score if sf else None
    tstr_ratio = ds.ratio if ds else None

    return Summary(
        moment_matching_score=mm.mean if mm else 0.0,
        ks_score=ks.mean if ks else 0.0,
        joint_score=jt.correlation_distance_score if jt else 0.0,
        alpha_precision=cv.alpha_precision if cv else None,
        beta_recall=cv.beta_recall if cv else None,
        authenticity=cv.authenticity if cv else None,
        logic_score=lg.hif_score_pct if lg and lg.error is None else None,
        stylized_facts_score=stylized_mean,
        tstr_ratio=tstr_ratio,
        rows_real=0,
        rows_synthetic=0,
        elapsed_seconds=0.0,
    )


class FidelityPipeline:
    def __init__(self, config: FidelityConfig | None = None):
        self.config = config or FidelityConfig()

    def run(
        self, real: pd.DataFrame, synthetic: pd.DataFrame
    ) -> FidelityReport:
        t0 = time.time()
        cols = _shared_columns(real, synthetic, self.config.columns)
        real = real[cols].copy()
        syn = synthetic[cols].copy()

        report = FidelityReport(
            dataset_type=self.config.dataset_type,
            columns_evaluated=cols,
        )

        metric_instances: list[Metric] = []
        metric_targets: dict[str, list[str]] = {}

        for name in _metrics.list_metrics():
            cls = _metrics.get_metric_cls(name)
            if name == "downstream":
                inst = cls(target_col=self.config.target_col)
            elif name == "hif":
                inst = cls(config=self.config.hif)
            else:
                inst = cls()

            err = inst.validate(real, syn)
            if err is not None:
                logger.debug("Skipping metric '%s': %s", name, err)
                continue

            mcols = _resolve_columns(real, inst.required_column_types())
            if not mcols:
                logger.debug("Skipping metric '%s': no matching columns", name)
                continue

            metric_instances.append(inst)
            metric_targets[name] = mcols

        if self.config.parallel and len(metric_instances) > 1:
            self._run_parallel(real, syn, metric_instances, metric_targets, report)
        else:
            self._run_sequential(real, syn, metric_instances, metric_targets, report)

        report.summary = _build_summary(report)
        report.summary.rows_real = len(real)
        report.summary.rows_synthetic = len(syn)
        report.summary.elapsed_seconds = round(time.time() - t0, 3)

        return report

    def _run_sequential(
        self,
        real: pd.DataFrame,
        syn: pd.DataFrame,
        metrics: list[Metric],
        targets: dict[str, list[str]],
        report: FidelityReport,
    ) -> None:
        for inst in metrics:
            try:
                inst.fit(real, targets[inst.name])
                result = inst.compute(real, syn, targets[inst.name])
                self._assign_result(report, inst.name, result)
            except Exception as e:
                logger.warning("Metric '%s' failed: %s", inst.name, e)

    def _run_parallel(
        self,
        real: pd.DataFrame,
        syn: pd.DataFrame,
        metrics: list[Metric],
        targets: dict[str, list[str]],
        report: FidelityReport,
    ) -> None:
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
            fit_futures = {
                ex.submit(inst.fit, real, targets[inst.name]): inst
                for inst in metrics
            }
            for fut in as_completed(fit_futures):
                inst = fit_futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    logger.warning("Metric '%s'.fit() failed: %s", inst.name, e)

            compute_futures = {}
            for inst in metrics:
                fut = ex.submit(inst.compute, real, syn, targets[inst.name])
                compute_futures[fut] = inst

            for fut in as_completed(compute_futures):
                inst = compute_futures[fut]
                try:
                    result = fut.result()
                    self._assign_result(report, inst.name, result)
                except Exception as e:
                    logger.warning("Metric '%s'.compute() failed: %s", inst.name, e)

    def _assign_result(
        self, report: FidelityReport, metric_name: str, result: dict
    ) -> None:
        section = _OLD_TO_NEW.get(metric_name)
        if section == "moment_matching":
            cols = result.get("column_scores", {})
            report.moment_matching = PerColumnScore(
                columns=cols, mean=result.get("mean_score", 0.0)
            )
        elif section == "distribution_fit":
            cols = result.get("column_scores", {})
            report.distribution_fit = PerColumnScore(
                columns=cols, mean=result.get("mean_score", 0.0)
            )
        elif section == "categorical_tvd":
            cols = result.get("column_scores", {})
            report.categorical_tvd = PerColumnScore(
                columns=cols, mean=result.get("mean_score", 0.0)
            )
        elif section == "joint":
            from tabular_polygraph._types.report import JointScore
            report.joint = JointScore(
                correlation_distance_score=result.get("correlation_distance_score", 0.0),
                pairwise_deltas=result.get("pairwise_deltas", {}),
            )
        elif section == "coverage":
            from tabular_polygraph._types.report import CoverageScore
            report.coverage = CoverageScore(
                alpha_precision=result.get("alpha_precision"),
                beta_recall=result.get("beta_recall"),
                authenticity=result.get("authenticity"),
            )
        elif section == "stylized_facts":
            from tabular_polygraph._types.report import StylizedFactsScore
            report.stylized_facts = StylizedFactsScore(
                per_column=result.get("per_column", {}),
                mean_score=result.get("mean_score"),
                columns_tested=result.get("columns_tested", 0),
                applicable=result.get("applicable", True),
            )
        elif section == "downstream":
            from tabular_polygraph._types.report import DownstreamScore
            if "error" in result:
                report.downstream = DownstreamScore(status="skipped", reason=result["error"])
            else:
                report.downstream = DownstreamScore(
                    target_col=result.get("target_col", ""),
                    metric=result.get("metric", ""),
                    task=result.get("task", ""),
                    tstr_score=result.get("tstr_score", 0.0),
                    trr_score=result.get("trr_score", 0.0),
                    ratio=result.get("ratio", 0.0),
                )
        elif section == "logical":
            from tabular_polygraph._types.report import LogicalScore
            if "error" in result:
                report.logical = LogicalScore(error=result["error"])
            else:
                report.logical = LogicalScore(
                    hif_score_pct=result.get("hif_score_pct", 0.0),
                    hif_violation_rate_pct=result.get("hif_violation_rate_pct", 0.0),
                    mean_penalty_pct=result.get("mean_penalty_pct", 0.0),
                    num_hif_violations=result.get("num_hif_violations", 0),
                    violation_threshold=result.get("violation_threshold"),
                    nic_violation_rate_pct=result.get("nic_violation_rate_pct", 0.0),
                    lse_violation_rate_pct=result.get("lse_violation_rate_pct", 0.0),
                    rule_violation_rate_pct=result.get("rule_violation_rate_pct", 0.0),
                    num_rule_violations=result.get("num_rule_violations", 0),
                    num_rules_mined=result.get("num_rules_mined", 0),
                    columns_used=result.get("columns_used", []),
                    top_violated_rules=result.get("top_violated_rules", []),
                    violation_examples=result.get("violation_examples", []),
                )
