from __future__ import annotations

import sys


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def err(msg: str) -> None:
    print(f"  [ERR] {msg}", file=sys.stderr)
