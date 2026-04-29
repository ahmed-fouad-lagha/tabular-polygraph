from pathlib import Path

import pandas as pd
import pytest

from tabular_polygraph.io.formats import _infer_format, read, supported_formats, write


def test_write_no_extension(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "testfile"  # No extension
    written_path = write(df, path)
    assert written_path.suffix == ".csv"
    assert written_path.exists()


def test_write_fmt_override(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "testfile"
    # 'parquet' canonical extension in _EXT_MAP is '.pq' (last one wins)
    written_path = write(df, path, fmt="parquet")
    assert written_path.suffix == ".pq"


def test_write_stata_long_strings(tmp_path):
    # Stata has a 244 char limit for strings in older versions/certain dtypes
    df = pd.DataFrame({"a": ["x" * 300, "y" * 10]})
    path = tmp_path / "test.dta"
    write(df, path)
    assert path.exists()
    # Check that it truncated if needed (internal logic does this)
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
