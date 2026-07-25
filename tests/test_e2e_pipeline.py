from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_polygraph.fidelity import fidelity_report
from tabular_polygraph.generators import GaussianCopulaGenerator


def test_e2e_generate_and_evaluate():
    np.random.seed(42)
    n = 100
    real_df = pd.DataFrame(
        {
            "age": np.random.randint(18, 70, size=n),
            "income": np.random.normal(50000, 15000, size=n),
            "approved": np.random.choice([0, 1], size=n),
        }
    )

    gen = GaussianCopulaGenerator()
    gen.fit(real_df)

    syn_df = gen.generate(n, seed=42)
    assert len(syn_df) == n
    assert "syn_id" in syn_df.columns

    syn_clean = syn_df.drop(columns=["syn_id"])
    report = fidelity_report(
        real_df,
        syn_clean,
        target_col="approved",
        include_downstream=True,
        hif_epochs=2,
    )

    assert "summary" in report
    assert "moment_matching" in report
    assert "distribution_fit" in report
    assert "downstream" in report
    assert report["summary"]["rows_real"] == n
    assert report["summary"]["rows_synthetic"] == n
