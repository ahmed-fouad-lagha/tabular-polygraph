"""Terminal output helpers for the CLI."""

from __future__ import annotations

import sys

import numpy as np


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
    if not np.isfinite(score):
        score = 0.0
    score = max(0.0, min(100.0, score))
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


def _json_clean(obj):
    import numpy as _np

    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


__all__ = [
    "C",
    "_c",
    "ok",
    "info",
    "warn",
    "err",
    "dim",
    "header",
    "section",
    "bar",
    "risk_colour",
    "_json_clean",
]
