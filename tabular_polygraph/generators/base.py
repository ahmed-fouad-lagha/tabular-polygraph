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

    # Default shorthand column aliases (subclasses or instances may override)
    _ALIASES: dict[str, str] = {
        "dti": "debt_to_income",
        "income": "applicant_income",
        "loan": "loan_amount",
        "gdp": "gdp_growth_yoy",
        "ffr": "fed_funds_rate",
        "assets": "total_assets",
    }
    _float_decimals: int = 4

    def __init__(self, **kwargs: Any):
        self._fitted = False
        self._n_fit = 0  # rows seen during fit
        self._columns: list[str] = []  # column order from fit
        self._dtypes: dict[str, Any] = {}  # original dtypes
        self._meta: dict = {}  # arbitrary metadata subclasses may store
        self._syn_id_counter = 100_000
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
            self._syn_id_counter = 100_000

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
                    # Use nullable Int64 to handle NaN without IntCastingNaNError
                    df[col] = df[col].round(0).astype("Int64")
                elif _pd.api.types.is_float_dtype(dtype):
                    df[col] = df[col].round(self._float_decimals)
            except Exception:
                pass
        return df

    def _add_syn_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepend a unique synthetic row identifier."""
        df = df.reset_index(drop=True)
        start_id = self._syn_id_counter
        self._syn_id_counter += len(df)
        df.insert(0, "syn_id", [f"SYN-{start_id + i}" for i in range(len(df))])
        return df

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = f"fitted on {self._n_fit:,} rows" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}({status})"

    # ── Filters ───────────────────────────────────────────────────────────────

    def _resolve_col(self, key: str, columns: list[str] | None = None) -> str | None:
        """Resolve abbreviated key → actual column name."""
        cols = columns if columns is not None else self._columns
        if not cols:
            return None
        if key in cols:
            return key
        if key in self._ALIASES and self._ALIASES[key] in cols:
            return self._ALIASES[key]
        matches = [c for c in cols if c == key or c.startswith(key + "_")]
        return matches[0] if len(matches) == 1 else None

    def _apply_filters(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """Apply column filters to a generated DataFrame.

        Supports exact match, ``_min``, and ``_max`` suffixes with alias resolution.
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
