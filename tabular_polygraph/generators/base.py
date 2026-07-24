"""
tabular_polygraph.generators.base
--------------------------
Abstract base class that every generator must implement.
Enforces a consistent interface across cross-sectional and deep generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseGenerator(ABC):
    """
    Abstract generator. Subclasses implement fit() and generate().

    Lifecycle
    ---------
        gen = MyGenerator(**kwargs)
        gen.fit(real_df)               # learn from data
        syn = gen.generate(n=1000)       # draw synthetic rows
        syn = gen.generate(n=500, filters={"state": ["CA"]})

    Or shorthand (fit + generate in one call):
        syn = gen.fit_generate(real_df, n=1000)
    """

    # Subclasses declare which dataset types they support
    # e.g. ["cross_sectional"]
    supported_types: list[str] = []

    def __init__(self, **kwargs: Any):
        self._fitted = False
        self._n_fit = 0  # rows seen during fit
        self._columns: list[str] = []  # column order from fit
        self._dtypes: dict[str, Any] = {}  # original dtypes
        self._meta: dict = {}  # arbitrary metadata subclasses may store
        self._init(**kwargs)

    @abstractmethod
    def _init(self, **kwargs: Any) -> None:
        """Optional hook for subclass __init__ logic without overriding __init__."""
        pass

    # ── Core interface ────────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseGenerator":
        """
        Learn the statistical structure of real data.
        Must set self._fitted = True and self._columns before returning.
        Returns self (fluent interface).
        """

    def generate(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Draw n synthetic rows.

        Handles global seeding automatically before calling subclass _generate().
        """
        self._require_fitted()

        if seed is not None:
            from tabular_polygraph.utils import set_seed

            set_seed(seed)

        return self._generate(n, filters=filters, seed=seed)

    @abstractmethod
    def _generate(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Internal implementation of sampling. Subclasses override this.

        Parameters
        ----------
        n       : number of rows to return
        filters : optional column constraints
        seed    : random seed (already applied globally by generate())

        Returns a DataFrame without 'syn_id' (BaseGenerator adds it if needed).
        """

    # ── Convenience ───────────────────────────────────────────────────────────

    def fit_generate(
        self,
        data: pd.DataFrame,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Fit on data then immediately generate n rows."""
        return self.fit(data).generate(n, filters=filters, seed=seed)

    def sample(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Backward-compatible alias for generate()."""
        return self.generate(n, filters=filters, seed=seed)

    def fit_sample(
        self,
        data: pd.DataFrame,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Backward-compatible alias for fit_generate()."""
        return self.fit_generate(data, n, filters=filters, seed=seed)

    # ── Guard ─────────────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. Call .fit(df) first."
            )

    # ── Shared utilities ──────────────────────────────────────────────────────

    def _record_schema(self, df: pd.DataFrame) -> None:
        """Store column order and dtypes from the training DataFrame."""
        self._columns = list(df.columns)
        self._dtypes = dict(df.dtypes)
        self._n_fit = len(df)

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast generated columns back to original dtypes where safe."""
        import pandas as _pd

        for col, dtype in self._dtypes.items():
            if col not in df.columns:
                continue
            try:
                if _pd.api.types.is_integer_dtype(dtype):
                    df[col] = df[col].round(0).astype(int)
                elif _pd.api.types.is_float_dtype(dtype):
                    df[col] = df[col].round(4)
            except Exception:
                pass
        return df

    def _add_syn_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepend a unique synthetic row identifier."""
        df = df.reset_index(drop=True)
        df.insert(0, "syn_id", [f"SYN-{100_000 + i}" for i in range(len(df))])
        return df

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = f"fitted on {self._n_fit:,} rows" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}({status})"
