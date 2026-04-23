import numpy as np
import pandas as pd

from tabular_polygraph.fidelity.logical import hif_score


def test_hif_sensitivity_to_hallucination():
    """
    Verifies that the hardened HIF detects combinatorial hallucinations
    (rows that are individually valid but logically inconsistent).
    """
    # Create a simple structured dataset: 'Income' depends on 'Job' and 'Education'
    # Job=Doctor -> Education=MD/PhD, Income=High
    # Job=Student -> Education=HighSchool/BS, Income=Low
    data = {
        "job": ["Doctor"] * 100 + ["Student"] * 100,
        "education": ["PhD"] * 100 + ["BS"] * 100,
        "income_bin": ["High"] * 100 + ["Low"] * 100,
        "age": np.random.randint(20, 60, 200).astype(float),
    }
    real = pd.DataFrame(data)

    # 1. Valid Synthetic Data (Perfect Sync)
    syn_valid = real.copy()

    # 2. Combinatorial Hallucination: Student with PhD and High Income
    # Statistically, all values are in the 'real' distribution, but the combination is rare/unseen.
    syn_hallucination = real.copy()
    syn_hallucination.loc[0, "job"] = "Student"
    syn_hallucination.loc[0, "education"] = "PhD"
    syn_hallucination.loc[0, "income_bin"] = "High"

    # Run HIF
    res_valid = hif_score(real, syn_valid, random_state=42, verbose=False)
    res_hallucination = hif_score(
        real, syn_hallucination, random_state=42, verbose=False
    )

    print(f"\nValid HIF Score: {res_valid['hif_score']}")
    print(f"Hallucination HIF Score: {res_hallucination['hif_score']}")

    # The hallucination row should have a high penalty
    # row_penalties[0] should be > 0
    assert res_hallucination["row_penalties"][0] > 0.5
    assert res_hallucination["hif_score"] < res_valid["hif_score"]


if __name__ == "__main__":
    test_hif_sensitivity_to_hallucination()
