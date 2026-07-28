import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.hif import LogicalSentinelEnsemble


def test_lse_oracle_trains_and_audits():
    real = pd.DataFrame(
        {
            "a": ["X", "Y", "Z", "W"] * 50,
            "b": ["1", "2", "3", "4"] * 50,
            "c": ["M", "N", "O", "P"] * 50,
        }
    )
    syn = real.copy()
    syn.loc[0, "b"] = "99"

    oracle = LogicalSentinelEnsemble()
    oracle.fit(real)
    score, penalties, meta = oracle.audit(syn)

    assert len(penalties) == len(syn)
    assert penalties[0] > 0.0
    assert penalties[1] == 0.0


def test_lse_oracle_trains_and_audits_simple():
    df = pd.DataFrame(
        {
            "A": ["x", "x", "y", "y"] * 25,
            "B": ["1", "1", "2", "2"] * 25,
            "C": ["a", "b", "c", "d"] * 25,
        }
    )
    lse = LogicalSentinelEnsemble(top_n_hubs=2)
    lse.fit(df, hif_epochs=2, verbose=False)
    assert lse.is_trained
    score, penalties, meta = lse.audit(df)
    assert score >= 0
