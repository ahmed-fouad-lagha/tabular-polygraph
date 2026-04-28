"""
tabular_polygraph.generators.deep.forest_diffusion
---------------------------------
ForestDiffusion generator: A modern, high-quality, and computationally efficient
diffusion model for tabular data based on XGBoost.

Installation:
    pip install .[deep]   # installs ForestDiffusion, torch
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..base import BaseGenerator


class ForestDiffusionGenerator(BaseGenerator):
    """
    ForestDiffusion generator based on XGBoost-based diffusion.

    Requires: pip install .[deep]

    Usage:
        gen = ForestDiffusionGenerator()
        gen.fit(real_df)
        syn = gen.generate(1000)
    """

    supported_types = ["cross_sectional"]

    def _init(self, n_t: int = 50, **kwargs):
        self._n_t = n_t
        self._model: Any | None = None

    def _require_forest_diffusion(self):
        try:
            import ForestDiffusion  # noqa: F401
        except ImportError:
            raise ImportError(
                "ForestDiffusion is not installed.\nRun: pip install .[deep]\n"
            ) from None

    def fit(self, data: pd.DataFrame) -> "ForestDiffusionGenerator":
        self._require_forest_diffusion()
        from ForestDiffusion import ForestDiffusionModel

        self._record_schema(data)

        # ForestDiffusion requires a numpy array of floats.
        # It handles categorical data if we provide X with labels or if we pre-encode.
        # For simplicity and robust integration, we pre-encode categorical columns.
        X = data.copy()
        self._cat_mappings = {}
        for col in self._columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype("category")
                self._cat_mappings[col] = dict(enumerate(X[col].cat.categories))
                X[col] = X[col].cat.codes.astype(float)
            else:
                X[col] = X[col].astype(float)

        self._model = ForestDiffusionModel(
            X.to_numpy(), n_t=self._n_t, duplicate_K=1, binarize=False
        )
        self._fitted = True
        return self

    def generate(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        self._require_fitted()
        self._require_forest_diffusion()

        if seed is not None:
            import numpy as np

            np.random.seed(seed)

        if self._model is None:
            raise RuntimeError("ForestDiffusion model is not fitted.")
        df_array = self._model.generate(batch_size=n)
        df = pd.DataFrame(df_array, columns=self._columns)

        # Decode categorical columns
        for col, mapping in self._cat_mappings.items():
            df[col] = df[col].round().clip(0, len(mapping) - 1).map(mapping)

        df = self._cast_types(df)

        if filters:
            df = self._apply_basic_filters(df, filters)
        return self._add_syn_id(df.head(n))

    def _apply_basic_filters(self, df, filters):
        for key, val in filters.items():
            if key in df.columns:
                if isinstance(val, list):
                    df = df[df[key].isin([str(v) for v in val])]
                else:
                    df = df[df[key] == val]
        return df
