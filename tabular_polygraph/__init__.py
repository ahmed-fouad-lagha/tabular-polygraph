"""
Quick start:
    from tabular_polygraph.generators import GaussianCopulaGenerator
    from tabular_polygraph.dataset    import load_dataset
    from tabular_polygraph.fidelity   import fidelity_report
"""

from .dataset import get_dataset_info, list_datasets, load_dataset
from .fidelity import fidelity_report, format_report
from .generators import (
    BaseGenerator,
    CTGANGenerator,
    GaussianCopulaGenerator,
    TVAEGenerator,
    VineCopulaGenerator,
)
from .io import read, validate, write
from .privacy import privacy_audit, format_audit

from ._version import __version__
__all__ = [
    "BaseGenerator",
    "CTGANGenerator",
    "GaussianCopulaGenerator",
    "TVAEGenerator",
    "VineCopulaGenerator",
    "list_datasets",
    "get_dataset_info",
    "load_dataset",
    "fidelity_report",
    "format_report",
    "read",
    "write",
    "validate",
    "privacy_audit",
    "format_audit",
]
