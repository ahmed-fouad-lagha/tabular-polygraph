"""
tabular_polygraph.generators.base
---------------------------------
Abstract base class that every generator must implement.
"""

from __future__ import annotations

import warnings
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
    """

    def __init__(self, **kwargs: Any):
        self._fitted = False
        self._n_fit = 0  # rows seen during fit
        self._columns: list[str] = []  # column order from fit
        self._dtypes: dict[str, Any] = {}  # original dtypes
        self._syn_id_counter = 0
        self._model: Any = None
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
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. Call .fit(df) first."
            )

        if seed is not None:
            from tabular_polygraph.utils import set_seed

            set_seed(seed)

        df = self._generate(n, filters=filters, seed=seed)
        return self._add_syn_id(df)

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
            try:
                if pd.api.types.is_integer_dtype(dtype):
                    df[col] = df[col].round(0).astype("Int64")
                elif pd.api.types.is_bool_dtype(dtype):
                    df[col] = df[col].round(0).astype("boolean")
                elif isinstance(dtype, pd.CategoricalDtype):
                    df[col] = df[col].astype(dtype)
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

    def __repr__(self) -> str:
        status = f"fitted on {self._n_fit:,} rows" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}({status})"

    def _sdv_generate(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Shared generate logic for SDV-based generators (CTGAN, TVAE)."""
        from sdv.sampling import Condition

        if seed is not None:
            self._model.set_random_state(seed)

        if not filters:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                df = self._model.sample(n)
            return self._cast_types(df)

        exact_filters: dict[str, Any] = {}
        post_filters: dict[str, Any] = {}
        for k, v in filters.items():
            if k in self._columns and not isinstance(v, list):
                exact_filters[k] = v
            else:
                post_filters[k] = v

        n_sample = n * 10 if post_filters else n

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            if exact_filters:
                conditions = [Condition(column_values=exact_filters, num_rows=n_sample)]
                df = self._model.sample_from_conditions(conditions=conditions)
            else:
                df = self._model.sample(n_sample)

        if post_filters:
            df = self._apply_filters(df, post_filters)
            if len(df) < n:
                warnings.warn(
                    f"Requested {n} rows but filters yielded only {len(df)}",
                    stacklevel=3,
                )

        df = df.head(n)
        return self._cast_types(df)

    def _apply_filters(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """Apply column filters to a generated DataFrame.

        Supports exact matching and _min/_max suffix range filtering.
        """
        import warnings

        cols = set(df.columns)
        for key, val in filters.items():
            # Exact column match
            if key in cols:
                if isinstance(val, list):
                    df = df[df[key].isin(val)]
                else:
                    df = df[df[key] == val]
                continue

            # _min / _max suffix → range filter
            if key.endswith("_min") and key[:-4] in cols:
                df = df[df[key[:-4]] >= val]
                continue

            if key.endswith("_max") and key[:-4] in cols:
                df = df[df[key[:-4]] <= val]
                continue

            warnings.warn(f"Filter key '{key}' did not match any column", stacklevel=2)

        return df
