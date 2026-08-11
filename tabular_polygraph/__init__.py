"""
Quick start:
    from tabular_polygraph.generators import GaussianCopulaGenerator
    from tabular_polygraph.dataset    import load_dataset
    from tabular_polygraph.fidelity   import fidelity_report
"""

from .dataset import get_dataset_info, list_datasets, load_dataset
from .fidelity import fidelity_report, format_report
from .fidelity.hif import hif_score
from .generators import (
    BaseGenerator,
    CTGANGenerator,
    GaussianCopulaGenerator,
    TVAEGenerator,
    VineCopulaGenerator,
)
from .io import read, validate, write
from .privacy import format_audit, privacy_audit

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
    "hif_score",
]
