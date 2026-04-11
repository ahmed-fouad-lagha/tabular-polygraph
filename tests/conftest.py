import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def all_seeds():
    from src.catalog import load_dataset, DATASETS

    return {did: load_dataset(did) for did in DATASETS}


@pytest.fixture(scope="module")
def hmda(all_seeds):
    return all_seeds["hmda"]


@pytest.fixture(scope="module")
def fred_macro(all_seeds):
    return all_seeds["fred_macro"]


@pytest.fixture(scope="module")
def world_bank(all_seeds):
    return all_seeds["world_bank"]


@pytest.fixture(scope="module")
def credit_risk(all_seeds):
    return all_seeds["credit_risk"]


@pytest.fixture(scope="module")
def edgar(all_seeds):
    return all_seeds["edgar"]


@pytest.fixture(scope="module")
def syn_hmda(hmda):
    from src.generators import GaussianCopulaGenerator

    gen = GaussianCopulaGenerator()
    gen.fit(hmda)
    return gen.generate(500, seed=42)


@pytest.fixture(scope="module")
def syn_macro(fred_macro):
    from src.generators.time_series import VARGenerator

    gen = VARGenerator(lags=2, time_col="year")
    gen.fit(fred_macro)
    return gen.generate(300, seed=42)


@pytest.fixture(scope="module")
def syn_wb(world_bank):
    from src.generators.panel import FixedEffectsGenerator

    gen = FixedEffectsGenerator(entity_col="country_code", time_col="year")
    gen.fit(world_bank)
    return gen.generate(300, seed=42)
