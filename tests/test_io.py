import os

import pandas as pd


def _sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})


def test_write_read_csv(tmp_path):
    from tabular_polygraph.io import read, write

    df = _sample_df()
    path = tmp_path / "test.csv"
    write(df, path)
    reloaded = read(path)
    assert len(reloaded) == 3


def test_write_read_parquet(tmp_path):
    from tabular_polygraph.io import read, write

    df = _sample_df()
    path = tmp_path / "test.parquet"
    write(df, path)
    reloaded = read(path)
    pd.testing.assert_frame_equal(reloaded, df)


def test_write_explicit_format(tmp_path):
    from tabular_polygraph.io import write

    df = _sample_df()
    target = os.path.join(str(tmp_path), "manual_save")
    path = write(df, target, fmt="parquet")
    # Path might end in .parquet or .pq
    assert str(path).lower().endswith((".parquet", ".pq"))
    assert os.path.exists(str(path))


def test_validate_catches_nulls():
    from tabular_polygraph.io import validate

    df = pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]})
    report = validate(df, min_rows=1)
    assert report.passed is False


def test_validate_min_rows():
    from tabular_polygraph.io import validate

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    report = validate(df, min_rows=10)
    assert report.passed is False
