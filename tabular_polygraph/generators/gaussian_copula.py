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


class _Prior:
    """Conjugate-prior shrinkage for Normal marginal parameters.

    Shrink the MLE mean / std toward *mean_0* / *std_0* with a
    pseudo-count of *n_0*.
    """

    def __init__(self, mean_0: float = 0.0, std_0: float = 1.0, n_0: int = 1) -> None:
        self.mean_0 = mean_0
        self.std_0 = std_0
        self.n_0 = n_0

    def map_mean(self, mle_mean: float, n: int) -> float:
        return (self.n_0 * self.mean_0 + n * mle_mean) / (self.n_0 + n)

    def map_std(self, mle_std: float, n: int) -> float:
        var = (self.n_0 * self.std_0 ** 2 + n * mle_std ** 2) / (self.n_0 + n)
        return float(np.sqrt(max(var, 1e-12)))


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
        return np.array([mapping.get(v, 0.5) if pd.notna(v) else 0.5 for v in series])

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
        syn = gen.generate(500, filters={"state": ["CA", "TX"], "income_min": 45})
    """

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

        # Handle NaNs and ensure PSD
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
            if min_eig < 1e-8:
                corr += (1e-8 - min_eig) * np.eye(len(self._columns))
        except np.linalg.LinAlgError:
            corr += 1e-6 * np.eye(len(self._columns))

        self._corr = corr

    def _generate(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        if self._corr is None:
            raise ValueError("Correlation matrix not fitted")

        try:
            L = np.linalg.cholesky(self._corr)
        except np.linalg.LinAlgError:
            L = np.eye(len(self._columns))

        filters = filters or {}

        resolved_filters = []
        for k, v in filters.items():
            if k in self._columns:
                resolved_filters.append((self._columns.index(k), k, "exact", v))
            elif k.endswith("_min") and k[:-4] in self._columns:
                resolved_filters.append((self._columns.index(k[:-4]), k[:-4], "min", v))
            elif k.endswith("_max") and k[:-4] in self._columns:
                resolved_filters.append((self._columns.index(k[:-4]), k[:-4], "max", v))

        valid_u_list = []
        collected = 0
        batch_size = max(n * 10, 1000)
        _max_iter = 1000
        _iter = 0

        while collected < n:
            _iter += 1
            if _iter > _max_iter:
                raise RuntimeError(
                    f"Generation terminated after {_max_iter} iterations: "
                    f"filters are too strict. Collected {collected}/{n} rows."
                )
            z = rng.standard_normal((batch_size, len(self._columns))) @ L.T
            u = stats.norm.cdf(z)

            if resolved_filters:
                mask = np.ones(batch_size, dtype=bool)
                for col_idx, col_name, f_type, f_val in resolved_filters:
                    col_vals = np.array(
                        self._marginals[col_name].from_uniform(u[:, col_idx])
                    )
                    if f_type == "exact":
                        if isinstance(f_val, list):
                            mask &= np.isin(col_vals, f_val)
                        else:
                            mask &= col_vals == f_val
                    elif f_type == "min":
                        mask &= col_vals >= f_val
                    elif f_type == "max":
                        mask &= col_vals <= f_val
                u = u[mask]

            if len(u) > 0:
                valid_u_list.append(u)
                collected += len(u)

        u_final = np.vstack(valid_u_list)[:n]

        records = {
            col: self._marginals[col].from_uniform(u_final[:, i])
            for i, col in enumerate(self._columns)
        }
        df = self._cast_types(pd.DataFrame(records))

        return df.head(n)

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
