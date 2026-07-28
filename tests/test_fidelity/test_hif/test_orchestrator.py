import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.hif import hif_score


def test_hif_handles_single_category_feature():
    real = pd.DataFrame({"cat": ["A"] * 20})
    syn = pd.DataFrame({"cat": ["A"] * 20})
    result = hif_score(real, syn, verbose=False)
    assert result["hif_score"] == 1.0
    assert result["violation_rate"] == 0.0
    assert result["mean_penalty"] == 0.0


def test_hif_small_dataset_train_with_verbose():
    real = pd.DataFrame(
        {"a": ["x", "y"] * 20, "b": ["m", "n"] * 20}
    )
    syn = real.copy()
    result = hif_score(real, syn, verbose=True, random_state=42)
    assert 0.0 <= result["hif_score"] <= 1.0


def test_hif_score_full_pipeline():
    real = pd.DataFrame(
        {
            "cat1": ["A", "B"] * 50,
            "cat2": ["X", "Y"] * 50,
            "num1": np.random.normal(0, 1, 100),
        }
    )
    syn = real.copy()
    res = hif_score(real, syn, verbose=False, hif_epochs=2)
    assert res["hif_score"] > 0.9

    syn.iloc[0, 0] = "B"
    res_v = hif_score(real, syn, verbose=False, hif_epochs=2)
    assert res_v["hif_score"] < res["hif_score"]


def test_hif_numeric_only():
    real = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    syn = pd.DataFrame({"x": [1, 3, 5, 2, 4], "y": [10, 30, 50, 20, 40]})
    result = hif_score(real, syn, verbose=False, hif_epochs=2)
    assert "hif_score" in result
    assert result["violation_rate"] == 0.0


def test_hif_geometric_mean_non_compensatory():
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

    syn_corrupt = syn.copy()
    syn_corrupt["cat1"] = np.random.choice(["A", "B"], 100)
    res_corrupt = hif_score(
        real, syn_corrupt, verbose=False, hif_epochs=2, component_floor=1e-4
    )

    assert res_corrupt["hif_score"] < res_clean["hif_score"]


def test_hif_sensitivity_to_hallucination():
    data = {
        "job": ["Doctor"] * 100 + ["Student"] * 100,
        "education": ["PhD"] * 100 + ["BS"] * 100,
        "income_bin": ["High"] * 100 + ["Low"] * 100,
        "age": np.random.randint(20, 60, 200).astype(float),
    }
    real = pd.DataFrame(data)

    syn_valid = real.copy()
    syn_hallucination = real.copy()
    syn_hallucination.loc[0, "job"] = "Student"
    syn_hallucination.loc[0, "education"] = "PhD"
    syn_hallucination.loc[0, "income_bin"] = "High"

    res_valid = hif_score(real, syn_valid, random_state=42, verbose=False)
    res_hallucination = hif_score(
        real, syn_hallucination, random_state=42, verbose=False
    )

    assert res_hallucination["row_penalties"][0] > 0.5
    assert res_hallucination["hif_score"] < res_valid["hif_score"]
