"""
Quick start:
    from tabular_polygraph.generators import GaussianCopulaGenerator
    from tabular_polygraph.catalog    import load_dataset
    from tabular_polygraph.fidelity   import fidelity_report
    from tabular_polygraph.privacy    import privacy_audit
    from tabular_polygraph.calibration import apply_scenario
"""

from .calibration import apply_scenario, list_scenarios
from .catalog import get_dataset_info, list_datasets, load_dataset
from .fidelity import fidelity_report, format_report
from .generators.base import BaseGenerator
from .generators.cross_sectional import GaussianCopulaGenerator
from .generators.panel import FixedEffectsGenerator
from .generators.time_series import VARGenerator
from .io import read, validate, write
from .privacy import format_audit, privacy_audit

__version__ = "1.0.0"
__all__ = [
    "BaseGenerator",
    "GaussianCopulaGenerator",
    "VARGenerator",
    "FixedEffectsGenerator",
    "list_datasets",
    "get_dataset_info",
    "load_dataset",
    "fidelity_report",
    "format_report",
    "privacy_audit",
    "format_audit",
    "apply_scenario",
    "list_scenarios",
    "read",
    "write",
    "validate",
]
