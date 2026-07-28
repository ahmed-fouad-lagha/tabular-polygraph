import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def all_seeds():
    from tabular_polygraph.dataset import DATASETS, load_dataset

    result = {}
    for dataset_id in DATASETS:
        try:
            result[dataset_id] = load_dataset(dataset_id)
        except ValueError:
            pytest.skip(
                f"Dataset '{dataset_id}' not cached. Run: tabular-polygraph download {dataset_id}"
            )
    return result


@pytest.fixture(scope="module")
def census_acs(all_seeds):
    return all_seeds.get("census_acs")
