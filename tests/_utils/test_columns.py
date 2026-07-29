import pandas as pd

from tabular_polygraph._utils import categorical_columns, numeric_columns


def test_column_filters():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.0, 2.0, 3.0]})
    assert numeric_columns(df) == ["a", "c"]
    assert categorical_columns(df) == ["b"]
