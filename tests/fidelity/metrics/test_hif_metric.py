import numpy as np
import pandas as pd

from tabular_polygraph._config import HIFConfig
from tabular_polygraph.fidelity.metrics.hif import HIFMetric


def test_hif_metric_basic():
    real = pd.DataFrame(
        {
            "cat1": ["A", "B"] * 50,
            "cat2": ["X", "Y"] * 50,
            "num1": np.random.normal(0, 1, 100),
        }
    )
    syn = real.copy()
    config = HIFConfig(epochs=2, hubs=2, depth=4)
    metric = HIFMetric(config=config)
    assert metric.validate(real, syn) is None
    res = metric.compute(real, syn, real.columns.tolist())
    assert "hif_score_pct" in res
    assert res["hif_score_pct"] is not None
