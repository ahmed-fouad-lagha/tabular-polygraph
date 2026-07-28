import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.hif import NeighborInvariantContinuity


def test_nic_scorer_manifold_continuity():
    np.random.seed(42)
    n = 200
    groups = np.random.choice(["low", "mid", "high"], n)
    offsets = {"low": -2, "mid": 0, "high": 2}
    noise = np.random.normal(0, 0.3, n)
    vals = np.array([offsets[g] for g in groups]) + noise

    real_cat = pd.DataFrame({"g1": groups, "g2": np.random.choice(["A", "B", "C"], n)})
    real_num = pd.DataFrame({"val": vals})

    syn_cat = pd.DataFrame({"g1": ["low"], "g2": ["A"]})
    syn_num = pd.DataFrame({"val": [offsets["low"] + 0.1]})

    scorer = NeighborInvariantContinuity()
    scorer.fit(real_cat, real_num)
    score, penalties = scorer.score(syn_cat, syn_num)
    assert penalties[0] < 0.5

    syn_cat2 = pd.DataFrame({"g1": ["low"], "g2": ["A"]})
    syn_num2 = pd.DataFrame({"val": [50.0]})
    _, penalties2 = scorer.score(syn_cat2, syn_num2)
    assert penalties2[0] > 0.5


def test_nic_auditor_basic():
    cat_df = pd.DataFrame({"A": ["x", "y"] * 50})
    cont_df = pd.DataFrame({"B": np.random.normal(0, 1, 100)})
    nic = NeighborInvariantContinuity()
    nic.fit(cat_df, cont_df, verbose=False)
    score, penalties = nic.score(cat_df, cont_df)
    assert score >= 0


def test_nic_outlier_detection():
    np.random.seed(42)
    n = 200
    groups = np.random.choice(["low", "high"], n)
    vals = np.where(
        groups == "low", np.random.normal(0, 0.5, n), np.random.normal(5, 0.5, n)
    )

    real_cat = pd.DataFrame({"g": groups})
    real_num = pd.DataFrame({"v": vals})

    scorer = NeighborInvariantContinuity()
    scorer.fit(real_cat, real_num)

    syn_cat = pd.DataFrame({"g": ["low"]})
    syn_num = pd.DataFrame({"v": [0.1]})
    _, p_in = scorer.score(syn_cat, syn_num)

    syn_num_out = pd.DataFrame({"v": [100.0]})
    _, p_out = scorer.score(syn_cat, syn_num_out)

    assert p_out[0] > p_in[0]
