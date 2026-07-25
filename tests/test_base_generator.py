from __future__ import annotations

import pandas as pd
import pytest

from tabular_polygraph.generators.base import BaseGenerator


class DummyGenerator(BaseGenerator):
    def _init(self, **kwargs):
        pass

    def fit(self, data: pd.DataFrame) -> DummyGenerator:
        self._record_schema(data)
        self._fitted = True
        return self

    def _generate(
        self, n: int, filters: dict | None = None, seed: int | None = None
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "age": [20 + i for i in range(n)],
                "score": [1.23456 for _ in range(n)],
                "state": ["CA" if i % 2 == 0 else "NY" for i in range(n)],
            }
        )
        df = self._cast_types(df)
        if filters:
            df = self._apply_filters(df, filters)
        return self._add_syn_id(df.head(n))


def test_base_generator_methods():
    df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "score": [1.1, 2.2, 3.3],
            "state": ["CA", "NY", "CA"],
        }
    )

    gen = DummyGenerator()
    with pytest.raises(RuntimeError):
        gen.generate(5)

    gen.fit(df)
    syn1 = gen.generate(3)
    assert "syn_id" in syn1.columns
    assert len(syn1) == 3
    assert syn1["syn_id"].iloc[0] == "SYN-100000"

    syn2 = gen.generate(2)
    assert syn2["syn_id"].iloc[0] == "SYN-100003"


def test_base_generator_filters_and_aliases():
    df = pd.DataFrame(
        {
            "debt_to_income": [10.0, 20.0, 30.0, 40.0],
            "state": ["CA", "NY", "CA", "TX"],
        }
    )

    gen = DummyGenerator()
    gen.fit(df)
    gen._columns = ["debt_to_income", "state"]

    filtered = gen._apply_filters(df, {"dti_min": 25.0})
    assert len(filtered) == 2
    assert (filtered["debt_to_income"] >= 25.0).all()
