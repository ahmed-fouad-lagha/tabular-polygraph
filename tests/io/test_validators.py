import numpy as np
import pandas as pd

from tabular_polygraph.io.validators import ValidationResult, validate


def test_validation_result_str():
    res = ValidationResult(passed=False, errors=["Error 1"], warnings=["Warning 1"])
    s = str(res)
    assert "Passed  : False" in s
    assert "Error 1" in s
    assert "Warning 1" in s


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
