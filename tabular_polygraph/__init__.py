"""
Quick start:
    from tabular_polygraph.generators import GaussianCopulaGenerator
    from tabular_polygraph.catalog    import load_dataset
    from tabular_polygraph.fidelity   import fidelity_report
    from tabular_polygraph.privacy    import privacy_audit
    from tabular_polygraph.calibration import apply_scenario
"""

from .generators.base import BaseGenerator
from .generators.cross_sectional import GaussianCopulaGenerator
from .generators.time_series import VARGenerator
from .generators.panel import FixedEffectsGenerator
from .catalog import list_datasets, get_dataset_info, load_dataset
from .fidelity import fidelity_report, format_report
from .privacy import privacy_audit, format_audit
from .calibration import apply_scenario, list_scenarios
from .io import read, write, validate

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
