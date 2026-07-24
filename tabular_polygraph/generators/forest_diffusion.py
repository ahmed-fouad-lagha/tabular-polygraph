"""
tabular_polygraph.generators.forest_diffusion
---------------------------------
ForestDiffusion generator: A modern, high-quality, and computationally efficient
diffusion model for tabular data based on XGBoost.

Installation:
    pip install .[forest]   # installs ForestDiffusion, torch
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseGenerator


class ForestDiffusionGenerator(BaseGenerator):
    """
    ForestDiffusion generator based on XGBoost-based diffusion.

    Requires: pip install .[forest]

    Usage:
        gen = ForestDiffusionGenerator()
        gen.fit(real_df)
        syn = gen.generate(1000)
    """

    supported_types = ["cross_sectional"]

    def _init(self, n_t: int = 10, max_train_rows: int = 5000, **kwargs):
        self._n_t = n_t
        self._max_train_rows = max_train_rows
        self._model: Any | None = None
        self._seed: int = 42

    def _require_forest_diffusion(self):
        try:
            import ForestDiffusion  # noqa: F401
        except ImportError:
            raise ImportError(
                "ForestDiffusion is not installed.\nRun: pip install .[forest]\n"
            ) from None

    def fit(self, data: pd.DataFrame) -> "ForestDiffusionGenerator":
        self._require_forest_diffusion()
        from ForestDiffusion import ForestDiffusionModel

        self._record_schema(data)

        X = data.copy()
        self._cat_mappings = {}
        cat_indexes = []
        int_indexes = []

        for i, col in enumerate(self._columns):
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype("category")
                self._cat_mappings[col] = dict(enumerate(X[col].cat.categories))
                X[col] = X[col].cat.codes.astype(float)
                cat_indexes.append(i)
            elif pd.api.types.is_integer_dtype(X[col]):
                X[col] = X[col].astype(float)
                int_indexes.append(i)
            else:
                X[col] = X[col].astype(float)

        X_np = X.to_numpy().astype("float32")

        # Subsample large datasets to keep ForestDiffusion tractable
        if len(X_np) > self._max_train_rows:
            import numpy as np

            rng = np.random.RandomState(self._seed)
            idx = rng.choice(len(X_np), self._max_train_rows, replace=False)
            X_np = X_np[idx]
            print(
                f"  [ForestDiffusion] Subsampled {len(data)} → {self._max_train_rows} rows for training"
            )

        print(
            f"  [ForestDiffusion] Fitting {self._n_t} diffusion steps on {len(X_np)} rows × {len(self._columns)} cols..."
        )

        # We use n_batch=0 for maximum speed on small/medium datasets
        # and n_estimators=50 for a faster audit.
        self._model = ForestDiffusionModel(
            X_np,
            n_t=self._n_t,
            duplicate_K=1,
            cat_indexes=cat_indexes,
            int_indexes=int_indexes,
            seed=self._seed,
            n_jobs=1,
            n_batch=0,  # Turbo mode: Use standard fast DMatrix
            n_estimators=50,  # Faster training for audit
        )
        self._fitted = True
        return self

    def _generate(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        self._require_forest_diffusion()

        if self._model is None:
            raise RuntimeError("ForestDiffusion model is not fitted.")

        if seed is not None:
            import numpy as np
            import torch

            np.random.seed(seed)
            torch.manual_seed(seed)

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
                    df = df[df[key].isin(val)]
                else:
                    df = df[df[key] == val]
        return df
