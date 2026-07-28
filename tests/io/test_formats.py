import os
from pathlib import Path

import pandas as pd
import pytest

from tabular_polygraph.io.formats import _infer_format, read, supported_formats, write


def _sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})


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
    path = tmp_path / "testfile"
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
