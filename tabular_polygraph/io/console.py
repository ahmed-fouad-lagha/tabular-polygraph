from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def ok(msg: str) -> None:
    logger.info("[OK] %s", msg)


def info(msg: str) -> None:
    logger.info("%s", msg)


def warn(msg: str) -> None:
    logger.warning("%s", msg)


def err(msg: str) -> None:
    logger.error("%s", msg)
