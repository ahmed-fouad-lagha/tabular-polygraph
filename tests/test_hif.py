import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.logical import (
    LogicalSentinelEnsemble,
    NeighborInvariantContinuity,
    _apply_binning,
    _fit_binning,
    hif_score,
    mine_implication_rules,
)


def test_adaptive_binning():
    df = pd.DataFrame(
        {
            "a": [1] * 10,
            "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "c": ["x", "y"] * 5,
        }
    )
    edges = _fit_binning(df, ["a", "b"])
    binned = _apply_binning(df, ["a", "b"], edges)
    assert binned["a"].iloc[0] == "bin_0"
    assert "bin_" in str(binned["b"].iloc[0])


def test_adaptive_binning_constant():
    df = pd.DataFrame({"a": [1, 1, 1, 1, 1]})
    edges = _fit_binning(df, ["a"])
    binned = _apply_binning(df, ["a"], edges)
    assert (binned["a"] == "bin_0").all()


def test_binning_consistency():
    """Bin edges fitted on real data must produce the same labels on synthetic."""
    np.random.seed(42)
    real = pd.DataFrame({"x": np.random.normal(0, 1, 500)})
    syn = pd.DataFrame({"x": np.random.normal(0.5, 1.2, 500)})

    edges = _fit_binning(real, ["x"])
    real_binned = _apply_binning(real, ["x"], edges)
    syn_binned = _apply_binning(syn, ["x"], edges)

    real_labels = set(real_binned["x"].unique())
    syn_labels = set(syn_binned["x"].unique())
    # Synthetic should use the same bin labels as real (same edges)
    assert syn_labels.issubset(real_labels | {"bin_0"}), (
        f"Synthetic has unexpected labels: {syn_labels - real_labels}"
    )


def test_mine_rules_numeric_quantization():
    # Test the branch where numeric columns with many unique values are quantized
    df = pd.DataFrame({"A": np.linspace(0, 100, 100), "B": ["x", "y"] * 50})
    rules = mine_implication_rules(
        df, columns=["A", "B"], min_confidence=0.1, min_support=0.01
    )
    # Check if 'A' was processed (it should be in some rules if correlations exist)
    assert any(
        "A" in r["antecedent_repr"] or r["consequent_feature"] == "A" for r in rules
    )


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


def test_hif_numeric_only():
    # Numeric-only datasets have no categorical manifold for LSE/NIC.
    # hif_score should return gracefully without crashing.
    real = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    # Provide in-distribution synthetic data so that it isn't flagged as an anomaly
    syn = pd.DataFrame({"x": [1, 3, 5, 2, 4], "y": [10, 30, 50, 20, 40]})
    result = hif_score(real, syn, verbose=False, hif_epochs=2)
    assert "hif_score" in result
    assert result["violation_rate"] == 0.0


def test_binning_cross_distribution():
    """Edges from a normal distribution should still label an exponential one."""
    np.random.seed(7)
    real = pd.DataFrame({"x": np.random.normal(0, 1, 1000)})
    syn = pd.DataFrame({"x": np.random.exponential(2, 1000)})

    edges = _fit_binning(real, ["x"])
    real_binned = _apply_binning(real, ["x"], edges)
    syn_binned = _apply_binning(syn, ["x"], edges)

    # Both should produce bin_ labels
    assert real_binned["x"].str.startswith("bin_").all()
    assert syn_binned["x"].str.startswith("bin_").all()
    # Real bins should cover 0..N (consecutive)
    real_bins = sorted(set(real_binned["x"]))
    assert real_bins == [f"bin_{i}" for i in range(len(real_bins))]


def test_nic_outlier_detection():
    """NIC should penalise values outside the real support, not hide them."""
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

    # In-range value
    syn_cat = pd.DataFrame({"g": ["low"]})
    syn_num = pd.DataFrame({"v": [0.1]})
    _, p_in = scorer.score(syn_cat, syn_num)

    # Outlier far beyond real range
    syn_num_out = pd.DataFrame({"v": [100.0]})
    _, p_out = scorer.score(syn_cat, syn_num_out)

    # Outlier should be penalised more than in-range
    assert p_out[0] > p_in[0]


def test_hif_geometric_mean_non_compensatory():
    """One bad component should dominate the geometric mean."""
    np.random.seed(42)
    real = pd.DataFrame(
        {
            "cat1": ["A", "B"] * 50,
            "cat2": ["X", "Y"] * 50,
            "num1": np.random.normal(0, 1, 100),
        }
    )
    syn = real.copy()
    res_clean = hif_score(real, syn, verbose=False, hif_epochs=2, component_floor=1e-4)

    # Corrupt only cat1 — should tank the whole score
    syn_corrupt = syn.copy()
    syn_corrupt["cat1"] = np.random.choice(["A", "B"], 100)
    res_corrupt = hif_score(
        real, syn_corrupt, verbose=False, hif_epochs=2, component_floor=1e-4
    )

    assert res_corrupt["hif_score"] < res_clean["hif_score"]
