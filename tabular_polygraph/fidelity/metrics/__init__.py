# ruff: noqa: E402 — submodule imports must stay after register()

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from tabular_polygraph._types import Metric

_METRICS: dict[str, type[Metric]] = {}


def register(metric_cls: type[Metric]) -> type[Metric]:
    _METRICS[metric_cls.name] = metric_cls  # type: ignore[index]
    return metric_cls


def list_metrics() -> list[str]:
    return list(_METRICS)


def get_metric_cls(name: str) -> Type[Metric]:
    if name not in _METRICS:
        raise KeyError(f"Unknown metric: {name}. Available: {list_metrics()}")
    return _METRICS[name]


from . import (
    alpha_beta,  # noqa: F401
    correlation,  # noqa: F401
    downstream,  # noqa: F401
    hif,  # noqa: F401
    ks_test,  # noqa: F401
    moment_matching,  # noqa: F401
    stylized_facts,  # noqa: F401
    tvd,  # noqa: F401
)
