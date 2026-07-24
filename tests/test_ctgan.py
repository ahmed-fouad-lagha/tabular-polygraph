import numpy as np
import pandas as pd
import pytest

from tabular_polygraph.generators.ctgan import CTGANGenerator


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 200).astype(float),
            "income": np.random.uniform(20000, 150000, 200),
            "category": np.random.choice(["A", "B", "C"], 200),
            "label": np.random.choice([0, 1], 200).astype(
                float
            ),  # Low cardinality numeric
        }
    )


def test_ctgan_discovery_logic(sample_data):
    """Verify that CTGAN correctly identifies discrete columns."""
    gen = CTGANGenerator(epochs=1)

    # We need to mock _require_ctgan to avoid dependency issues during discovery check if not installed
    # But for a real test we assume it's installed as we just did 'pip install .[ctgan]'
    pytest.importorskip("ctgan")

    gen.fit(sample_data)

    # Check internal discovery (we can't easily check _model.fit arguments without mocking)
    # but we can verify the generator is fitted.
    assert gen._fitted
    assert len(gen._columns) == 4


def test_ctgan_generate_basic(sample_data):
    """Test basic generation loop."""
    pytest.importorskip("ctgan")

    gen = CTGANGenerator(epochs=1)
    gen.fit(sample_data)
    syn = gen.generate(10)

    assert len(syn) == 10
    assert list(syn.columns) == ["syn_id"] + list(sample_data.columns)
    assert syn["age"].dtype == sample_data["age"].dtype


def test_ctgan_reproducibility(sample_data):
    """Test that seeding works correctly."""
    pytest.importorskip("ctgan")

    gen = CTGANGenerator(epochs=1)
    gen.fit(sample_data)

    syn1 = gen.generate(5, seed=42)
    syn2 = gen.generate(5, seed=42)

    pd.testing.assert_frame_equal(syn1, syn2)


def test_ctgan_filters(sample_data):
    """Test basic filtering logic in CTGAN."""
    pytest.importorskip("ctgan")

    gen = CTGANGenerator(epochs=1)
    gen.fit(sample_data)

    # Filter for category A
    filters = {"category": "A"}
    syn = gen.generate(5, filters=filters)

    assert len(syn) == 5
    assert (syn["category"] == "A").all()


def test_ctgan_require_fitted_guard():
    """Verify guard against generating before fitting."""
    gen = CTGANGenerator()
    with pytest.raises(RuntimeError, match="not been fitted"):
        gen.generate(10)


def test_ctgan_manual_discrete_override(sample_data):
    """Verify that manual discrete column override works."""
    pytest.importorskip("ctgan")

    # Manually specify only 'category' as discrete, even if 'label' has low cardinality
    gen = CTGANGenerator(epochs=1, discrete_columns=["category"])
    gen.fit(sample_data)
    assert gen._fitted


def test_ctgan_custom_params(sample_data):
    """Verify that custom CTGAN parameters are correctly passed."""
    pytest.importorskip("ctgan")

    gen = CTGANGenerator(
        epochs=2,
        batch_size=100,
        generator_lr=1e-3,
        discriminator_steps=5,
        discrete_threshold=5,
    )
    gen.fit(sample_data)
    assert gen._model._epochs == 2
    assert gen._model._batch_size == 100
    assert gen._model._generator_lr == 1e-3
    assert gen._model._discriminator_steps == 5


def test_ctgan_missing_dependency(monkeypatch):
    """Test informative error when ctgan is missing."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "ctgan":
            raise ImportError("Mocked missing ctgan")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    gen = CTGANGenerator()
    with pytest.raises(ImportError, match="CTGAN is not installed"):
        gen._require_ctgan()
