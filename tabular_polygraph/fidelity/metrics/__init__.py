from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from tabular_polygraph._types import Metric

_METRICS: dict[str, type[Metric]] = {}


def register(metric_cls: type[Metric]) -> type[Metric]:
    _METRICS[metric_cls.name] = metric_cls
    return metric_cls


def list_metrics() -> list[str]:
    return list(_METRICS)


def get_metric_cls(name: str) -> Type[Metric]:
    if name not in _METRICS:
        raise KeyError(f"Unknown metric: {name}. Available: {list_metrics()}")
    return _METRICS[name]


from . import moment_matching  # noqa: F401, E402
from . import ks_test  # noqa: F401, E402
from . import tvd  # noqa: F401, E402
from . import correlation  # noqa: F401, E402
from . import alpha_beta  # noqa: F401, E402
from . import stylized_facts  # noqa: F401, E402
from . import downstream  # noqa: F401, E402
from . import hif  # noqa: F401, E402
