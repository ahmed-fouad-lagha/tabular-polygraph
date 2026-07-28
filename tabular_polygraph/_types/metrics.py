from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def required_column_types(self) -> set[str]:
        return {"all"}

    def validate(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> str | None:
        return None

    def fit(self, real: pd.DataFrame, columns: list[str]) -> None:
        pass

    @abstractmethod
    def compute(
        self, real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]
    ) -> dict:
        ...
