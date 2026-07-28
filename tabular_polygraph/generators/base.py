"""
tabular_polygraph.generators.base
---------------------------------
Abstract base class that every generator must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

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

    def __init__(self, **kwargs: Any):
        self._fitted = False
        self._n_fit = 0  # rows seen during fit
        self._columns: list[str] = []  # column order from fit
        self._dtypes: dict[str, Any] = {}  # original dtypes
        self._syn_id_counter = 0
        self._init(**kwargs)

    @abstractmethod
    def _init(self, **kwargs: Any) -> None:
        """Hook for subclass __init__ logic without overriding __init__."""
        pass

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseGenerator":
        """
        Learn the statistical structure of real data.
        Must call self._record_schema(data) and set self._fitted = True.
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
            self._syn_id_counter = 0

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
        for col, dtype in self._dtypes.items():
            if col not in df.columns:
                continue
            if pd.api.types.is_integer_dtype(dtype):
                try:
                    df[col] = df[col].round(0).astype("Int64")
                except (ValueError, TypeError):
                    pass
        return df

    def _add_syn_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepend a unique synthetic row identifier."""
        df = df.reset_index(drop=True)
        start_id = self._syn_id_counter
        self._syn_id_counter += len(df)
        df.insert(0, "syn_id", [f"SYN-{start_id + i}" for i in range(len(df))])
        return df

    def _generate_with_retry(
        self,
        n: int,
        filters: dict | None,
        sample_fn: Callable[[int], pd.DataFrame],
        max_attempts: int = 5,
    ) -> pd.DataFrame:
        """Generate rows with retry when filters reduce the count below n.

        Parameters
        ----------
        n          : desired row count
        filters    : optional column constraints (None skips retry)
        sample_fn  : callable that takes a row count and returns a DataFrame
        max_attempts: maximum retry iterations
        """
        if not filters:
            return self._add_syn_id(sample_fn(n).head(n))

        df = self._apply_filters(sample_fn(n * 10), filters)
        attempts = 0
        while len(df) < n and attempts < max_attempts:
            attempts += 1
            df_more = self._apply_filters(sample_fn(n * 10), filters)
            df = pd.concat([df, df_more], ignore_index=True)

        return self._add_syn_id(df.head(n))

    def __repr__(self) -> str:
        status = f"fitted on {self._n_fit:,} rows" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}({status})"

    # ── Filters ───────────────────────────────────────────────────────────────

    def _resolve_col(self, key: str, columns: list[str] | None = None) -> str | None:
        """Resolve a filter key to an actual column name."""
        cols = columns if columns is not None else self._columns
        if not cols:
            return None
        if key in cols:
            return key
        matches = [c for c in cols if c.startswith(key + "_")]
        return matches[0] if len(matches) == 1 else None

    def _apply_filters(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """Apply column filters to a generated DataFrame.

        Supports exact match, ``_min``, and ``_max`` suffixes with prefix matching.
        """
        cols = list(df.columns)
        for key, val in filters.items():
            if key.endswith("_min"):
                base_key = key[:-4]
                col = self._resolve_col(base_key, cols)
                if col and col in df.columns:
                    df = df[df[col] >= val]
            elif key.endswith("_max"):
                base_key = key[:-4]
                col = self._resolve_col(base_key, cols)
                if col and col in df.columns:
                    df = df[df[col] <= val]
            else:
                col = self._resolve_col(key, cols)
                if col and col in df.columns:
                    if isinstance(val, list):
                        df = df[df[col].isin(val)]
                    else:
                        df = df[df[col] == val]
        return df
