"""Regression tests for checkpoint/resume in 13_nonhub_target_recovery.py.

Guards against the bug where a resumed run skipped combos already in the raw
CSV but then overwrote the file with only newly computed rows, silently
dropping every previously completed seed.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

_spec = importlib.util.spec_from_file_location(
    "nonhub_target_recovery",
    SCRIPTS_DIR / "13_nonhub_target_recovery.py",
)
module = importlib.util.module_from_spec(_spec)
sys.modules["nonhub_target_recovery"] = module
_spec.loader.exec_module(module)

TARGETS = module.CONFIGS[0][2]


def _row(ds_id: str, gen_name: str, seed: int, tgt: str) -> dict:
    return {
        "dataset": ds_id,
        "generator": gen_name,
        "target": tgt,
        "target_is_hub": tgt in TARGETS[:5],
        "seed": seed,
        "retention": 60.0,
        "violation_rate": 0.1,
        "f1_full": 0.4,
        "f1_filtered": 0.5,
    }


def _make_real(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {tgt: rng.integers(0, 4, size=n).astype(float) for tgt in TARGETS}
    )


class _FakeAuditor:
    def __init__(self, config):
        self.oracle = SimpleNamespace(hubs=list(TARGETS[:5]))

    def fit(self, real):
        pass

    def score(self, syn):
        return {
            "row_penalties": np.full(len(syn), 0.1),
            "violation_rate": 0.0,
        }


def _install_fakes(monkeypatch):
    monkeypatch.setattr(module, "load_real", lambda ds, n, seed: _make_real(n))
    monkeypatch.setattr(module, "generate", lambda real, n, seed, gen: real.copy())
    monkeypatch.setattr(module, "HIFAuditor", _FakeAuditor)
    monkeypatch.setattr(
        module,
        "utility_metrics",
        lambda real, syn, tgt, seed: {"f1": 0.4 + 0.05 * (seed % 3)},
    )


def test_load_existing_rows_roundtrip(tmp_path):
    rows = [_row("census_acs", "ctgan", 42, tgt) for tgt in TARGETS]
    module._save_checkpoint(tmp_path, rows)
    loaded = module._load_existing_rows(tmp_path)
    assert len(loaded) == len(rows)
    assert {(r["dataset"], r["generator"], int(r["seed"])) for r in loaded} == {
        ("census_acs", "ctgan", 42)
    }
    assert module._load_existing_rows(tmp_path / "missing") == []


def test_resume_preserves_previously_completed_seeds(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    # Simulate an interrupted first process: seed 42 was fully checkpointed.
    module._save_checkpoint(
        tmp_path, [_row("census_acs", "ctgan", 42, tgt) for tgt in TARGETS]
    )

    module.run(n_seeds=2, output_dir=tmp_path)

    raw = pd.read_csv(tmp_path / "nonhub_target_recovery_raw.csv")
    ctgan_seeds = set(raw.loc[raw["generator"] == "ctgan", "seed"].astype(int))
    # Regression: the pre-existing seed must survive the resumed run's saves.
    assert ctgan_seeds == {42, 43}

    summ = pd.read_csv(tmp_path / "nonhub_target_recovery_summary.csv")
    ctgan = summ[summ["generator"] == "ctgan"]
    assert len(ctgan) == len(TARGETS)
    assert (ctgan["n_valid"] == 2).all()
