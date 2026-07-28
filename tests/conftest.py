import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _fallback_seed(did: str, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    if did == "census_acs":
        return pd.DataFrame(
            {
                "age": rng.integers(18, 90, n),
                "income": rng.normal(60000.0, 25000.0, n).clip(1000.0, None),
                "education": rng.choice(["HS", "College", "Graduate"], n),
                "employment_status": rng.choice(
                    ["employed", "unemployed"], n, p=[0.92, 0.08]
                ),
            }
        )

    if did == "adult":
        return pd.DataFrame(
            {
                "age": rng.integers(17, 90, n),
                "workclass": rng.choice(["Private", "Self-emp", "Gov"], n),
                "education": rng.choice(
                    ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"], n
                ),
                "marital_status": rng.choice(
                    ["Married", "Never-married", "Divorced"], n
                ),
                "occupation": rng.choice(
                    ["Prof-specialty", "Exec-managerial", "Sales", "Craft-repair"], n
                ),
                "relationship": rng.choice(
                    ["Husband", "Not-in-family", "Own-child", "Unmarried"], n
                ),
                "race": rng.choice(
                    [
                        "White",
                        "Black",
                        "Asian-Pac-Islander",
                        "Amer-Indian-Eskimo",
                        "Other",
                    ],
                    n,
                ),
                "sex": rng.choice(["Male", "Female"], n),
                "capital_gain": rng.lognormal(0, 1, n) * 1000,
                "capital_loss": rng.lognormal(0, 1, n) * 100,
                "hours_per_week": rng.normal(40, 5, n).clip(1, 99),
                "native_country": rng.choice(
                    ["United-States", "Mexico", "Philippines", "Germany"], n
                ),
                "income": rng.choice(["<=50K", ">50K"], n),
            }
        )

    raise ValueError(f"Unsupported fallback dataset: {did}")


@pytest.fixture(scope="module")
def all_seeds():
    from tabular_polygraph.dataset import DATASETS, load_dataset

    result = {}
    for did in DATASETS:
        try:
            result[did] = load_dataset(did)
        except ValueError:
            result[did] = _fallback_seed(did)
    return result


@pytest.fixture(scope="module")
def census_acs(all_seeds):
    return all_seeds.get("census_acs")
