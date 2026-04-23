import numpy as np
import pandas as pd
import pytest

from tabular_polygraph.utils import (
    categorical_columns,
    denormalize,
    format_score_bar,
    normalize,
    numeric_columns,
    shared_columns,
    to_numeric_array,
)


def test_column_filters():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.0, 2.0, 3.0]})
    assert numeric_columns(df) == ["a", "c"]
    assert categorical_columns(df) == ["b"]

    df2 = pd.DataFrame({"a": [4, 5], "b": ["w", "v"], "d": [True, False]})
    assert shared_columns(df, df2) == ["a", "b"]
    assert shared_columns(df, df2, exclude=["a"]) == ["b"]


def test_to_numeric_array():
    s = pd.Series([1.0, 2.0, np.nan, 4.0])

    # dropna
    assert len(to_numeric_array(s, fill_method="dropna")) == 3

    # mean
    arr_mean = to_numeric_array(s, fill_method="mean")
    assert arr_mean[2] == pytest.approx(7.0 / 3.0)

    # median
    arr_median = to_numeric_array(s, fill_method="median")
    assert arr_median[2] == 2.0

    # zero
    arr_zero = to_numeric_array(s, fill_method="zero")
    assert arr_zero[2] == 0.0

    # forward
    s_ts = pd.Series([1.0, np.nan, 3.0])
    arr_fwd = to_numeric_array(s_ts, fill_method="forward")
    assert arr_fwd[1] == 1.0

    # value
    arr_val = to_numeric_array(s, fill_method="value", fill_value=99.0)
    assert arr_val[2] == 99.0

    with pytest.raises(ValueError, match="fill_value required"):
        to_numeric_array(s, fill_method="value")

    with pytest.raises(ValueError, match="Unknown fill_method"):
        to_numeric_array(s, fill_method="invalid")


def test_normalization():
    arr = np.array([1.0, 2.0, 3.0])
    norm, mu, sigma = normalize(arr, return_params=True)
    assert mu == 2.0
    assert sigma > 0
    assert np.allclose(denormalize(norm, mu, sigma), arr)

    # Constant array
    const = np.array([5.0, 5.0, 5.0])
    norm_const = normalize(const)
    assert np.allclose(norm_const, 0.0)


def test_format_score_bar():
    bar = format_score_bar(50, width=10)
    assert bar == "█████░░░░░"
    bar_full = format_score_bar(100, width=5)
    assert bar_full == "█████"
