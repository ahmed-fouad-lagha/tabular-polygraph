import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabular_polygraph.io.formats import _infer_format, read, supported_formats, write
from tabular_polygraph.io.validators import ValidationResult, validate


def _sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})


# ── FORMATS & IO ─────────────────────────────────────────────────────────────


def test_write_read_csv(tmp_path):
    df = _sample_df()
    path = tmp_path / "test.csv"
    write(df, path)
    reloaded = read(path)
    assert len(reloaded) == 3


def test_write_read_parquet(tmp_path):
    df = _sample_df()
    path = tmp_path / "test.parquet"
    write(df, path)
    reloaded = read(path)
    pd.testing.assert_frame_equal(reloaded, df)


def test_write_explicit_format(tmp_path):
    df = _sample_df()
    target = os.path.join(str(tmp_path), "manual_save")
    path = write(df, target, fmt="parquet")
    assert str(path).lower().endswith((".parquet", ".pq"))
    assert os.path.exists(str(path))


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


def test_write_no_extension(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "testfile"  # No extension
    written_path = write(df, path)
    assert written_path.suffix == ".csv"
    assert written_path.exists()


def test_write_fmt_override(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "testfile"
    written_path = write(df, path, fmt="parquet")
    assert written_path.suffix == ".pq"


def test_write_stata_long_strings(tmp_path):
    df = pd.DataFrame({"a": ["x" * 300, "y" * 10]})
    path = tmp_path / "test.dta"
    write(df, path)
    assert path.exists()
    res = read(path)
    assert len(res.iloc[0, 0]) == 244


def test_read_not_found():
    with pytest.raises(FileNotFoundError):
        read("non_existent_file_xyz.csv")


def test_infer_format_unsupported():
    with pytest.raises(ValueError, match="Cannot infer format"):
        _infer_format(Path("test.txt"))


def test_supported_formats():
    fmts = supported_formats()
    assert "csv" in fmts
    assert "parquet" in fmts
    assert "arrow" in fmts
    assert "json" in fmts


# ── VALIDATORS ─────────────────────────────────────────────────────────────


def test_validation_result_str():
    res = ValidationResult(passed=False, errors=["Error 1"], warnings=["Warning 1"])
    s = str(res)
    assert "Passed  : False" in s
    assert "✗ Error 1" in s
    assert "! Warning 1" in s


def test_validate_catches_nulls():
    df = pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]})
    report = validate(df, min_rows=1)
    assert report.passed is False


def test_validate_min_rows():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    report = validate(df, min_rows=10)
    assert report.passed is False


def test_validate_required_columns():
    df = pd.DataFrame({"a": [1, 2, 3] * 20})
    res = validate(df, required_columns=["a", "b"])
    assert not res.passed
    assert any("Missing required columns: ['b']" in e for e in res.errors)


def test_validate_dtypes():
    df = pd.DataFrame({"a": ["x", "y"] * 30, "b": [1, 2] * 30})
    res = validate(df, expected_dtypes={"a": "numeric", "b": "categorical"})
    assert not res.passed
    assert any("Column 'a' expected numeric" in e for e in res.errors)
    assert any("Column 'b' expected categorical" in w for w in res.warnings)


def test_validate_unrecognized_dtype_silent():
    df = pd.DataFrame({"a": [1, 2] * 30})
    res = validate(df, expected_dtypes={"a": "boolean"})
    assert res.passed


def test_validate_nulls():
    df = pd.DataFrame({"a": [None] * 50 + [1] * 10})
    res = validate(df, null_threshold=0.3)
    assert not res.passed
    assert any("Column 'a' has 83.3% nulls" in e for e in res.errors)


def test_validate_const_and_cardinality():
    df = pd.DataFrame(
        {
            "const": [1] * 60,
            "high_card": [str(i) for i in range(60)],
            "other": [1, 2, 3] * 20,
        }
    )
    res = validate(df, max_cardinality=10)
    assert any("Column 'const' is constant" in w for w in res.warnings)
    assert any("high_card' has very high cardinality" in w for w in res.warnings)


def test_validate_duplicates():
    df = pd.DataFrame({"a": [1, 2] * 30})
    res = validate(df, duplicate_threshold=0.1)
    assert any("58 duplicate rows (96.7%)" in w for w in res.warnings)


def test_validate_inf():
    df = pd.DataFrame({"a": [1.0, 2.0, np.inf, 4.0] * 15})
    res = validate(df)
    assert not res.passed
    assert any("contains Inf or NaN" in e for e in res.errors)


def test_validate_no_cols():
    df = pd.DataFrame()
    res = validate(df)
    assert not res.passed
    assert any("no columns" in e for e in res.errors)
