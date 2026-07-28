"""
tabular_polygraph.generators.vine_copula
--------------------------------------------------
Vine (Pair) Copula generator for cross-sectional data.

Improves on Gaussian Copula in two key ways:
1. Each column pair gets its own copula family (Clayton, Gumbel, Frank, Joe, Gaussian)
   chosen by AIC — capturing asymmetric dependence structures
2. Tail dependence: Clayton captures lower tail dependence (joint crashes),
   Gumbel captures upper tail dependence (joint booms)

This matters for financial data where correlations are higher in downturns
than in normal periods — the Gaussian Copula misses this completely.

Limitations
-----------
Categorical columns are sampled independently from their marginal distributions.
The vine copula models only the continuous features' joint structure. Joint
dependence between categorical and numeric features is NOT captured.

Requirements
-----------
    pip install pyvinecopulib

Usage
-----
    from tabular_polygraph.generators import VineCopulaGenerator

    gen = VineCopulaGenerator()
    gen.fit(real_df)
    syn = gen.generate(10_000)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseGenerator


def _require_pyvine():
    try:
        import pyvinecopulib  # noqa
    except ImportError:
        raise ImportError(
            "pyvinecopulib is required for VineCopulaGenerator.\n"
            "Install it with: pip install .[vine]\n"
            "  or: pip install pyvinecopulib"
        ) from None


def _resolve_family_set(family_set):
    """Convert a user-facing family_set spec to a list of BicopFamily enums."""
    _require_pyvine()
    import pyvinecopulib as pv

    if isinstance(family_set, list):
        return family_set

    if family_set == "parametric":
        return [
            pv.BicopFamily.gaussian,  # type: ignore
            pv.BicopFamily.student,  # type: ignore
            pv.BicopFamily.clayton,  # type: ignore
            pv.BicopFamily.gumbel,  # type: ignore
            pv.BicopFamily.frank,  # type: ignore
            pv.BicopFamily.joe,  # type: ignore
        ]

    if family_set == "all":
        return list(pv.BicopFamily.__members__.values())

    if family_set == "tll":
        return [pv.BicopFamily.tll]

    # Try direct enum lookup
    members = pv.BicopFamily.__members__
    if family_set in members:
        return [members[family_set]]

    raise ValueError(
        f"Unknown family_set '{family_set}'. "
        f"Choose from: 'parametric', 'all', 'tll', or a list of BicopFamily enums."
    )


def _tail_dependence_from_family(family, params: list[float]) -> tuple[float, float]:
    """Compute lower and upper tail dependence coefficients analytically.

    Returns (lower_tail, upper_tail).
    """
    import pyvinecopulib as pv

    if family == pv.BicopFamily.indep:  # type: ignore
        return 0.0, 0.0
    elif family == pv.BicopFamily.gaussian:  # type: ignore
        return 0.0, 0.0
    elif family == pv.BicopFamily.frank:  # type: ignore
        return 0.0, 0.0
    elif family == pv.BicopFamily.clayton:  # type: ignore
        theta = params[0] if params else 0.0
        return (2.0 ** (-1.0 / theta), 0.0) if theta > 0 else (0.0, 0.0)
    elif family == pv.BicopFamily.gumbel:  # type: ignore
        theta = params[0] if params else 1.0
        return (0.0, 2.0 - 2.0 ** (1.0 / theta)) if theta > 1 else (0.0, 0.0)
    elif family == pv.BicopFamily.joe:  # type: ignore
        theta = params[0] if params else 1.0
        return (0.0, 2.0 - 2.0 ** (1.0 / theta)) if theta > 1 else (0.0, 0.0)
    elif family == pv.BicopFamily.student:  # type: ignore
        rho = params[0] if params else 0.0
        nu = params[1] if len(params) > 1 else 30.0
        lam = 2.0 * stats.t.cdf(
            -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho)), df=nu + 1
        )
        return lam, lam
    elif family == pv.BicopFamily.bb1:  # type: ignore
        theta = params[0] if params else 0.0
        delta = params[1] if len(params) > 1 else 0.0
        if theta > 0 and 0 < delta <= 1:
            return 2.0 ** (-1.0 / theta), 2.0 - 2.0 ** (delta / theta)
        return 0.0, 0.0
    else:
        return 0.0, 0.0


class VineCopulaGenerator(BaseGenerator):
    """
    Vine (pair) copula generator.

    Fits a C-vine or R-vine structure where each pair of variables
    gets its own copula family, capturing asymmetric tail dependence.

    Usage
    -----
        gen = VineCopulaGenerator(family_set="all")
        gen.fit(real_df)
        syn = gen.generate(10_000)

    Parameters
    ----------
    family_set : copula families to consider
        "parametric"   — Gaussian, Student-t, Clayton, Gumbel, Frank, Joe
        "all"          — all parametric + non-parametric
        "tll"          — transformation kernel (most flexible, slowest)
        list           — e.g. ["gaussian", "clayton", "gumbel"]
    trunc_lvl : int
        Truncation level for the vine (0 = full vine, 1 = fast approximation)
    """

    def _init(self, family_set: str = "parametric", trunc_lvl: int = 0, **kwargs):
        self._family_set = family_set
        self._trunc_lvl = trunc_lvl
        self._vine: Any = None
        self._marginals: dict[str, dict[str, Any]] = {}
        self._numeric_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._cat_marginals: dict[str, dict[str, Any]] = {}

    def fit(self, data: pd.DataFrame) -> "VineCopulaGenerator":
        _require_pyvine()
        import pyvinecopulib as pv

        self._record_schema(data)

        self._numeric_cols = [
            c for c in self._columns if pd.api.types.is_numeric_dtype(data[c])
        ]
        self._cat_cols = [
            c for c in self._columns if not pd.api.types.is_numeric_dtype(data[c])
        ]

        # Fit marginals for each numeric column (empirical CDF)
        for col in self._numeric_cols:
            arr = data[col].dropna().astype(float).values
            if len(arr) == 0:
                self._marginals[col] = {
                    "sorted": np.array([]),
                    "n": 0,
                    "min": 0.0,
                    "max": 0.0,
                }
            else:
                self._marginals[col] = {
                    "sorted": np.sort(arr),
                    "n": len(arr),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }

        # Fit categorical distributions
        for col in self._cat_cols:
            vc = data[col].dropna().value_counts(normalize=True)
            if vc.empty:
                self._cat_marginals[col] = {"cats": [], "probs": np.array([])}
            else:
                self._cat_marginals[col] = {"cats": list(vc.index), "probs": vc.values}

        # Transform to uniform via empirical CDF
        n = len(data)
        U = np.zeros((n, len(self._numeric_cols)))
        for i, col in enumerate(self._numeric_cols):
            arr = data[col].fillna(data[col].median()).astype(float).values
            ranks = stats.rankdata(arr)
            U[:, i] = ranks / (n + 1)

        U = np.clip(U, 1e-4, 1 - 1e-4)

        # Fit vine copula
        families = _resolve_family_set(self._family_set)
        controls = pv.FitControlsVinecop(
            family_set=families,
            trunc_lvl=self._trunc_lvl,
            num_threads=1,
        )
        self._vine = pv.Vinecop.from_data(data=U, controls=controls)
        self._fitted = True
        return self

    def _generate(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        _require_pyvine()
        rng = np.random.default_rng(seed)
        filters = filters or {}

        resolved_filters = []
        for k, v in filters.items():
            if k in self._columns:
                resolved_filters.append((k, "exact", v))
            elif k.endswith("_min") and k[:-4] in self._columns:
                resolved_filters.append((k[:-4], "min", v))
            elif k.endswith("_max") and k[:-4] in self._columns:
                resolved_filters.append((k[:-4], "max", v))

        valid_records: dict[str, list[np.ndarray]] = {col: [] for col in self._columns}
        collected = 0
        batch_size = max(n * 10, 1000)

        while collected < n:
            U_syn = self._vine.simulate(batch_size)
            U_syn = np.clip(U_syn, 1e-4, 1 - 1e-4)

            records = {}
            for i, col in enumerate(self._numeric_cols):
                m = self._marginals[col]
                if m["n"] == 0:
                    records[col] = np.full(batch_size, 0.0)
                    continue
                quantile_idx = U_syn[:, i] * (m["n"] - 1)
                lower = np.floor(quantile_idx).astype(int)
                upper = np.minimum(lower + 1, m["n"] - 1)
                frac = quantile_idx - lower
                values = m["sorted"][lower] * (1 - frac) + m["sorted"][upper] * frac
                records[col] = np.clip(values, m["min"], m["max"])

            for col in self._cat_cols:
                m = self._cat_marginals[col]
                if not m["cats"]:
                    records[col] = np.full(batch_size, "unknown", dtype=object)
                    continue
                records[col] = rng.choice(m["cats"], size=batch_size, p=m["probs"])

            if resolved_filters:
                mask = np.ones(batch_size, dtype=bool)
                for col_name, f_type, f_val in resolved_filters:
                    col_vals = records[col_name]
                    if f_type == "exact":
                        if isinstance(f_val, list):
                            mask &= np.isin(col_vals, f_val)
                        else:
                            mask &= col_vals == f_val
                    elif f_type == "min":
                        mask &= col_vals >= f_val
                    elif f_type == "max":
                        mask &= col_vals <= f_val

                for col in self._columns:
                    records[col] = records[col][mask]

            num_valid = len(records[self._columns[0]])
            if num_valid > 0:
                for col in self._columns:
                    valid_records[col].append(records[col])
                collected += num_valid

        final_records = {}
        for col in self._columns:
            final_records[col] = np.concatenate(valid_records[col])[:n]

        df = pd.DataFrame(final_records)[self._columns]
        df = self._cast_types(df)
        return self._apply_filters(df, filters).head(n)

    def tail_dependence_report(self) -> dict:
        """Return upper and lower tail dependence coefficients for each pair.

        High values (> 0.1) indicate the variables tend to move together
        in extremes (e.g. joint crashes or joint booms).

        For tree-0 (unconditional) pairs, reports variable names.
        For higher trees (conditional pairs), reports tree/edge positions.
        """
        _require_pyvine()

        if self._vine is None:
            return {}

        report = {}
        d = self._vine.dim

        for tree in range(d - 1):
            for edge in range(d - 1 - tree):
                try:
                    pc = self._vine.get_pair_copula(tree, edge)
                except (IndexError, RuntimeError):
                    continue

                family = pc.family
                params = pc.parameters.tolist() if pc.parameters.size > 0 else []
                tau = float(pc.tau)

                # Compute tail dependence analytically from family parameters
                lower_tail, upper_tail = _tail_dependence_from_family(family, params)

                # Build label: tree-0 uses variable names, higher trees use position
                if tree == 0 and edge + 1 < len(self._vine.order):
                    order = self._vine.order
                    v1 = self._numeric_cols[order[edge] - 1]
                    v2 = self._numeric_cols[order[edge + 1] - 1]
                    label = f"{v1} x {v2}"
                else:
                    label = f"tree{tree}_edge{edge}"

                report[label] = {
                    "family": str(family),
                    "parameters": params,
                    "tau": round(tau, 3),
                    "lower_tail": round(lower_tail, 3),
                    "upper_tail": round(upper_tail, 3),
                }

        return report
