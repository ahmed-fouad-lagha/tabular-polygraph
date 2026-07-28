import numpy as np
import pandas as pd
import pytest


def test_vine_copula_smoke():
    try:
        import pyvinecopulib  # noqa: F401
    except ImportError:
        pytest.skip("pyvinecopulib not installed")

    from tabular_polygraph.generators import VineCopulaGenerator

    df = pd.DataFrame(
        {
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "c": np.random.choice(["X", "Y"], 100),
        }
    )
    gen = VineCopulaGenerator()
    gen.fit(df)
    syn = gen.generate(10)
    assert len(syn) == 10
    assert all(col in syn.columns for col in df.columns)
