import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.hif.binning import apply_binning, fit_binning


def test_adaptive_binning():
    df = pd.DataFrame(
        {
            "a": [1] * 10,
            "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "c": ["x", "y"] * 5,
        }
    )
    edges = fit_binning(df, ["a", "b"])
    binned = apply_binning(df, ["a", "b"], edges)
    assert binned["a"].iloc[0] == "bin_0"
    assert "bin_" in str(binned["b"].iloc[0])


def test_adaptive_binning_constant():
    df = pd.DataFrame({"a": [1, 1, 1, 1, 1]})
    edges = fit_binning(df, ["a"])
    binned = apply_binning(df, ["a"], edges)
    assert (binned["a"] == "bin_0").all()


def test_binning_consistency():
    np.random.seed(42)
    real = pd.DataFrame({"x": np.random.normal(0, 1, 500)})
    syn = pd.DataFrame({"x": np.random.normal(0.5, 1.2, 500)})

    edges = fit_binning(real, ["x"])
    real_binned = apply_binning(real, ["x"], edges)
    syn_binned = apply_binning(syn, ["x"], edges)

    real_labels = set(real_binned["x"].unique())
    syn_labels = set(syn_binned["x"].unique())
    assert syn_labels.issubset(real_labels | {"bin_0"}), (
        f"Synthetic has unexpected labels: {syn_labels - real_labels}"
    )


def test_binning_cross_distribution():
    np.random.seed(7)
    real = pd.DataFrame({"x": np.random.normal(0, 1, 1000)})
    syn = pd.DataFrame({"x": np.random.exponential(2, 1000)})

    edges = fit_binning(real, ["x"])
    real_binned = apply_binning(real, ["x"], edges)
    syn_binned = apply_binning(syn, ["x"], edges)

    assert real_binned["x"].str.startswith("bin_").all()
    assert syn_binned["x"].str.startswith("bin_").all()
    real_bins = sorted(set(real_binned["x"]))
    assert real_bins == [f"bin_{i}" for i in range(len(real_bins))]
