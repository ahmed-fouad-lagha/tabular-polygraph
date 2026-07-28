import numpy as np
import pandas as pd
import pytest


def test_tvae_smoke():
    try:
        from sdv.single_table import TVAESynthesizer  # noqa: F401
    except ImportError:
        pytest.skip("sdv not installed")

    from tabular_polygraph.generators import TVAEGenerator

    df = pd.DataFrame(
        {
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "c": np.random.choice(["X", "Y"], 100),
        }
    )
    gen = TVAEGenerator(epochs=10)
    gen.fit(df)
    syn = gen.generate(10)
    assert len(syn) == 10
    assert all(col in syn.columns for col in df.columns)
