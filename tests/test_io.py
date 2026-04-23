import pandas as pd
import pytest
from pathlib import Path


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_amount": [100000, 120000, 98000, 135000],
            "interest_rate": [3.5, 3.8, 3.2, 4.1],
            "segment": ["A", "B", "A", "C"],
        }
    )


def test_csv_roundtrip(tmp_path):
    from tabular_polygraph.io import read, write

    df = _sample_df()
    path = tmp_path / "sample.csv"
    write(df, path)
    reloaded = read(path)
    assert len(reloaded) == len(df)
    assert set(reloaded.columns) == set(df.columns)


def test_json_roundtrip(tmp_path):
    from tabular_polygraph.io import read, write

    df = _sample_df()
    path = tmp_path / "sample.json"
    write(df, path)
    reloaded = read(path)
    assert len(reloaded) == len(df)
    assert set(reloaded.columns) == set(df.columns)


def test_stata_roundtrip(tmp_path):
    from tabular_polygraph.io import read, write

    df = _sample_df()
    path = tmp_path / "sample.dta"
    write(df, path)
    reloaded = read(path)
    assert len(reloaded) == len(df)
    assert set(reloaded.columns) == set(df.columns)


def test_unsupported_format_raises(tmp_path):
    from tabular_polygraph.io import write

    with pytest.raises(ValueError):
        write(_sample_df(), Path(tmp_path) / "sample.xyz")


def test_validate_catches_nulls():
    from tabular_polygraph.io import validate

    df = pd.DataFrame({"a": [1, None, None, None, None], "b": [1, 2, 3, 4, 5]})
    result = validate(df, min_rows=3)
    assert not result.passed
    assert any("a" in e for e in result.errors)


def test_validate_warns_constant_column():
    from tabular_polygraph.io import validate

    df = pd.DataFrame({"a": [1] * 100, "b": range(100)})
    result = validate(df)
    assert any("constant" in w.lower() for w in result.warnings)


def test_validate_warns_high_cardinality():
    from tabular_polygraph.io import validate

    df = pd.DataFrame(
        {
            "id": [f"id_{i}" for i in range(200)],
            "val": range(200),
        }
    )
    result = validate(df, max_cardinality=50, min_rows=100)
    assert any("cardinality" in w.lower() for w in result.warnings)


def test_supported_formats_list():
    from tabular_polygraph.io import supported_formats

    fmts = supported_formats()
    assert "csv" in fmts
    assert "json" in fmts
    assert "stata" in fmts
