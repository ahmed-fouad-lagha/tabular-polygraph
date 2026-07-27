"""
tabular_polygraph.privacy.common
---------------------------------
Shared risk-level helpers for the TAMIS privacy suite.

Each privacy test uses domain-specific thresholds, so there are three
named variants rather than a single universal function.
"""

from __future__ import annotations


def risk_level_membership(auc: float) -> str:
    """Risk level for membership inference AUC scores."""
    if auc < 0.52:
        return "very_low"
    if auc < 0.60:
        return "low"
    if auc < 0.70:
        return "medium"
    if auc < 0.80:
        return "high"
    return "very_high"


def risk_level_singling_out(rate: float) -> str:
    """Risk level for singling-out attack rates."""
    if rate < 0.001:
        return "very_low"
    if rate < 0.01:
        return "low"
    if rate < 0.05:
        return "medium"
    if rate < 0.15:
        return "high"
    return "very_high"


def risk_level_linkability(rate: float) -> str:
    """Risk level for linkability attack rates."""
    if rate < 0.52:
        return "very_low"
    if rate < 0.60:
        return "low"
    if rate < 0.70:
        return "medium"
    if rate < 0.85:
        return "high"
    return "very_high"
