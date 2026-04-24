import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.logical import (
    LogicalSentinelEnsemble,
    NeighborInvariantContinuity,
    _adaptive_binning,
    hif_score,
    mine_implication_rules,
)


def test_adaptive_binning():
    df = pd.DataFrame(
        {"a": [1, 1, 1, 1], "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "c": ["x", "y"] * 5}
    )
    binned = _adaptive_binning(df, ["a", "b"])
    assert binned["a"].iloc[0] == "bin_0"
    assert "bin_" in str(binned["b"].iloc[0])


def test_mine_implication_rules():
    df = pd.DataFrame({"A": ["x", "x", "y", "y"] * 25, "B": ["1", "1", "2", "2"] * 25})
    rules = mine_implication_rules(
        df, columns=["A", "B"], min_confidence=0.9, min_support=0.1
    )
    assert len(rules) > 0
    assert rules[0]["confidence"] >= 0.9


def test_lse_oracle():
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


def test_nic_auditor():
    cat_df = pd.DataFrame({"A": ["x", "y"] * 50})
    cont_df = pd.DataFrame({"B": np.random.normal(0, 1, 100)})
    nic = NeighborInvariantContinuity()
    nic.fit(cat_df, cont_df, verbose=False)
    score, penalties = nic.score(cat_df, cont_df)
    assert score >= 0


def test_hif_score_full_pipeline():
    real = pd.DataFrame(
        {
            "cat1": ["A", "B"] * 50,
            "cat2": ["X", "Y"] * 50,
            "num1": np.random.normal(0, 1, 100),
        }
    )
    syn = real.copy()
    # Perfect match
    res = hif_score(real, syn, verbose=False, hif_epochs=2)
    assert res["hif_score"] > 0.9

    # Violation
    syn.iloc[0, 0] = "B"  # Break cat1=A -> cat2=X logic
    res_v = hif_score(real, syn, verbose=False, hif_epochs=2)
    assert res_v["hif_score"] < res["hif_score"]
