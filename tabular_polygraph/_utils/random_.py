from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.warning("torch not available — skipping torch seed")
