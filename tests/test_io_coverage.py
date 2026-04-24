import numpy as np
import pandas as pd
import pytest

from tabular_polygraph.io.formats import read, supported_formats, write
from tabular_polygraph.io.validators import validate


def test_io_json_roundtrip(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    path = tmp_path / "test.json"
    write(df, path)
    df_read = read(path)
    pd.testing.assert_frame_equal(df, df_read)


def test_io_unsupported_format():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(ValueError, match="Cannot infer format"):
        write(df, "test.unknown_ext")


def test_validate_edge_cases():
    # Constant column warning
    df = pd.DataFrame({"A": [1, 1, 1, 1, 1]})
    res = validate(df, min_rows=1)
    assert any("is constant" in w for w in res.warnings)

    # High cardinality warning
    df = pd.DataFrame({"A": [str(i) for i in range(1000)]})
    res = validate(df, min_rows=1, max_cardinality=10)
    assert any("high cardinality" in w for w in res.warnings)

    # No columns error
    df = pd.DataFrame()
    res = validate(df)
    assert not res.passed
    assert any("no columns" in e for e in res.errors)


def test_validate_inf_values():
    df = pd.DataFrame({"A": [1.0, 2.0, np.inf, 4.0, 5.0] * 10})
    res = validate(df)
    assert not res.passed
    assert any("contains Inf or NaN" in e for e in res.errors)


def test_supported_formats():
    formats = supported_formats()
    assert "csv" in formats
    assert "json" in formats
    assert "parquet" in formats
