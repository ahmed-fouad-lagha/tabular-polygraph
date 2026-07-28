from __future__ import annotations

import pandas as pd

DEFAULT_DROP_LIST: set[str] = {
    "syn_id",
    "id",
    "row_id",
    "index",
    "uid",
    "uuid",
    "tract_id",
    "serial_no",
    "fips_code",
    "ip_address",
}


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
    ]
