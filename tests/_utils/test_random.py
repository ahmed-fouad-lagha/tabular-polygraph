from __future__ import annotations

import random

import numpy as np

from tabular_polygraph.utils import set_seed


def test_set_seed_reproducibility():
    set_seed(42)
    val1_py = random.random()
    val1_np = np.random.randn()

    set_seed(42)
    val2_py = random.random()
    val2_np = np.random.randn()

    assert val1_py == val2_py
    assert val1_np == val2_np


def test_set_seed_none():
    set_seed(None)
