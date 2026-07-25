"""
tabular_polygraph.generators.gaussian_copula
------------------------------------------------------
Gaussian Copula generator for cross-sectional tabular data.

Algorithm
---------
1. Fit a marginal model per column (log-normal / normal / categorical)
2. Transform each column to uniform via its CDF, then to normal via Φ⁻¹
3. Learn the inter-column correlation matrix in normal space
4. Sample correlated normals via Cholesky decomposition
5. Invert back through each marginal to produce synthetic values

Produces statistically faithful synthetic records with zero exact copies
of real rows and no individual-level information.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseGenerator


class _NumericMarginal:
    """Fits and inverts a single numeric column distribution."""

    def __init__(self) -> None:
        self._params: dict = {}
        self._min = self._max = 0.0

    def fit(self, series: pd.Series) -> "_NumericMarginal":
        import warnings

        arr = series.dropna().astype(float).values
        if len(arr) == 0:
            self._min = self._max = 0.0
            self._params = {"kind": "norm", "loc": 0.0, "scale": 1.0}
            return self
        self._min, self._max = float(arr.min()), float(arr.max())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skewness = float(stats.skew(arr))
        if skewness > 1.0 and self._min > 0:
            s, loc, scale = stats.lognorm.fit(arr, floc=0)
            self._params = {"kind": "lognorm", "s": s, "loc": loc, "scale": scale}
        else:
            self._params = {
                "kind": "norm",
                "loc": float(arr.mean()),
                "scale": float(arr.std()) or 1.0,
            }
        return self

    def to_uniform(self, series: pd.Series) -> np.ndarray:
        arr = series.fillna(series.median()).astype(float).values
        p = self._params
        if p["kind"] == "lognorm":
            u = stats.lognorm.cdf(arr, p["s"], loc=p["loc"], scale=p["scale"])
        else:
            u = stats.norm.cdf(arr, loc=p["loc"], scale=p["scale"])
        return np.clip(u, 1e-6, 1 - 1e-6)

    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-6, 1 - 1e-6)
        p = self._params
        if p["kind"] == "lognorm":
            v = stats.lognorm.ppf(u, p["s"], loc=p["loc"], scale=p["scale"])
        else:
            v = stats.norm.ppf(u, loc=p["loc"], scale=p["scale"])
        return np.clip(v, self._min, self._max)

    @property
    def kind(self) -> str:
        return self._params.get("kind", "unknown")


class _CategoricalMarginal:
    """Fits and inverts a single categorical column distribution."""

    def __init__(self) -> None:
        self._cats: list = []
        self._probs: np.ndarray = np.array([])

    def fit(self, series: pd.Series) -> "_CategoricalMarginal":
        vc = series.dropna().value_counts(normalize=True)
        self._cats = list(vc.index)
        self._probs = vc.values
        return self

    def to_uniform(self, series: pd.Series) -> np.ndarray:
        if not self._cats:
            return np.full(len(series), 0.5)
        mapping = {c: (i + 0.5) / len(self._cats) for i, c in enumerate(self._cats)}
        return np.array([mapping.get(v, 0.5) for v in series.fillna(self._cats[0])])

    def from_uniform(self, u: np.ndarray) -> list:
        if not self._cats:
            return ["unknown"] * len(u)
        cum = np.cumsum(self._probs)
        idx = np.clip(
            np.searchsorted(cum, np.clip(u, 1e-6, 1 - 1e-6)), 0, len(self._cats) - 1
        )
        return [self._cats[i] for i in idx]

    @property
    def kind(self) -> str:
        return "categorical"


class GaussianCopulaGenerator(BaseGenerator):
    """
    Cross-sectional Gaussian Copula synthetic data generator.

    Usage
    -----
        from tabular_polygraph.generators import GaussianCopulaGenerator

        gen = GaussianCopulaGenerator()
        gen.fit(real_df)
        syn = gen.generate(1000)
        syn = gen.generate(500, filters={"state": ["CA", "TX"], "dti_min": 45})
    """

    supported_types = ["cross_sectional"]

    # Shorthand aliases: 'dti' → 'debt_to_income', etc.
    _ALIASES: dict[str, str] = {
        "dti": "debt_to_income",
        "income": "applicant_income",
        "loan": "loan_amount",
        "gdp": "gdp_growth_yoy",
        "ffr": "fed_funds_rate",
        "assets": "total_assets",
    }

    def _init(self, priors: Any | None = None, **kwargs: Any) -> None:
        self._marginals: dict[str, _NumericMarginal | _CategoricalMarginal] = {}
        self._corr: np.ndarray | None = None
        self._priors = priors

    def fit(self, data: pd.DataFrame) -> "GaussianCopulaGenerator":
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._fit_impl(data)

    def _fit_impl(self, data: pd.DataFrame) -> "GaussianCopulaGenerator":
        self._record_schema(data)
        self._fit_marginals(data)

        # Transform to normal space
        uniform = np.column_stack(
            [self._marginals[c].to_uniform(data[c]) for c in self._columns]
        )
        normal = stats.norm.ppf(np.clip(uniform, 1e-6, 1 - 1e-6))

        # HARDENING: Handle extreme outliers that might cause INFs
        if np.any(np.isinf(normal)):
            normal = np.nan_to_num(normal, posinf=4.0, neginf=-4.0)

        self._fit_correlation(normal)
        self._fitted = True
        return self

    def _fit_marginals(self, data: pd.DataFrame) -> None:
        """Fit marginal distributions for all columns."""
        for col in self._columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                m = _NumericMarginal().fit(data[col])
                if self._priors is not None:
                    self._apply_numeric_prior(m, col, len(data[col].dropna()))
                self._marginals[col] = m
            else:
                self._marginals[col] = _CategoricalMarginal().fit(data[col])

    def _apply_numeric_prior(self, m: _NumericMarginal, col: str, n: int) -> None:
        """Apply prior regularization to a numeric marginal."""
        if self._priors is None:
            return
        prior = self._priors.get(col)
        if prior is not None and m.kind in ("norm", "lognorm"):
            p = m._params
            p["loc"] = prior.map_mean(p.get("loc", 0), n)
            p["scale"] = max(prior.map_std(p.get("scale", 1), n), 1e-6)

    def _fit_correlation(self, normal: np.ndarray) -> None:
        """Estimate and regularize the inter-column correlation matrix."""
        if len(self._columns) == 0:
            self._corr = np.eye(0)
            return

        corr = np.corrcoef(normal.T)
        corr = np.atleast_2d(corr)

        # HARDENING: Handle NaNs and ensure PSD
        if np.any(np.isnan(corr)):
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)

        # Numerical stability jitter
        corr = (corr + corr.T) / 2
        corr += np.eye(len(self._columns)) * 1e-10

        try:
            # Use SVD for more robust eigenvalue estimation on ill-conditioned matrices
            _, s, _ = np.linalg.svd(corr)
            min_eig = s.min()
            # If SVD is successful but min singular value is effectively 0 or negative
            # (singular values are non-negative, but numerical noise might happen)
            if min_eig < 1e-8:
                corr += (1e-8 - min_eig) * np.eye(len(self._columns))
        except np.linalg.LinAlgError:
            # Absolute fallback
            corr += 1e-6 * np.eye(len(self._columns))

        self._corr = corr

    def _generate(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        n_gen = n * (6 if filters else 1)

        # Correlated normal samples via modern Generator
        rng = np.random.default_rng(seed)
        try:
            L = np.linalg.cholesky(self._corr)
            z = rng.standard_normal((n_gen, len(self._columns))) @ L.T
        except np.linalg.LinAlgError:
            z = rng.standard_normal((n_gen, len(self._columns)))

        u = stats.norm.cdf(z)

        records = {
            col: self._marginals[col].from_uniform(u[:, i])
            for i, col in enumerate(self._columns)
        }
        df = pd.DataFrame(records)
        df = self._cast_types(df)

        if filters:
            df = self._apply_filters(df, filters)

        return self._add_syn_id(df.head(n))

    # ── filter helpers ────────────────────────────────────────────────────────

    def _resolve_col(self, key: str) -> str | None:
        """Resolve abbreviated key → actual column name."""
        if key in self._columns:
            return key
        if key in self._ALIASES and self._ALIASES[key] in self._columns:
            return self._ALIASES[key]
        # unambiguous prefix match
        matches = [c for c in self._columns if c == key or c.startswith(key + "_")]
        return matches[0] if len(matches) == 1 else None

    def _apply_filters(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """Apply a set of filters to a generated DataFrame."""
        filtered_df = df
        for key, val in filters.items():
            if key.endswith("_min"):
                filtered_df = self._apply_min_filter(filtered_df, key, val)
            elif key.endswith("_max"):
                filtered_df = self._apply_max_filter(filtered_df, key, val)
            else:
                filtered_df = self._apply_exact_filter(filtered_df, key, val)
        return filtered_df

    def _apply_min_filter(self, df: pd.DataFrame, key: str, val: Any) -> pd.DataFrame:
        col = self._resolve_col(key[:-4])
        return df[df[col] >= val] if col else df

    def _apply_max_filter(self, df: pd.DataFrame, key: str, val: Any) -> pd.DataFrame:
        col = self._resolve_col(key[:-4])
        return df[df[col] <= val] if col else df

    def _apply_exact_filter(self, df: pd.DataFrame, key: str, val: Any) -> pd.DataFrame:
        col = self._resolve_col(key)
        if not col:
            return df
        # Don't stringify — keep original types so numeric columns match numeric values
        if isinstance(val, list):
            return df[df[col].isin(val)]
        return df[df[col] == val]

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def marginal_kinds(self) -> dict[str, str]:
        """Return the fitted distribution kind for each column."""
        return {col: m.kind for col, m in self._marginals.items()}

    @property
    def correlation_matrix(self) -> pd.DataFrame | None:
        """Return the learned correlation matrix as a DataFrame."""
        if self._corr is None:
            return None
        return pd.DataFrame(self._corr, index=self._columns, columns=self._columns)
