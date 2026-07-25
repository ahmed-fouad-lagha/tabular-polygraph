"""
tabular_polygraph.generators.ctgan
---------------------------------
Stub for CTGAN (Conditional Tabular GAN) deep generator.

CTGAN significantly outperforms Gaussian Copula on:
  - Multi-modal numeric distributions
  - Imbalanced categorical columns
  - Complex non-linear inter-column relationships

Requirements:
    pip install .[ctgan]   # installs ctgan, torch
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseGenerator


class CTGANGenerator(BaseGenerator):
    """
    CTGAN deep tabular generator.

    Usage (same interface as GaussianCopulaGenerator):
        gen = CTGANGenerator(epochs=300, batch_size=500)
        gen.fit(real_df)
        syn = gen.generate(1000)
    """

    supported_types = ["cross_sectional"]

    def _init(
        self,
        epochs: int = 50,
        batch_size: int = 500,
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        discriminator_steps: int = 1,
        log_frequency: bool = True,
        verbose: bool = False,
        discrete_columns: list[str] | None = None,
        discrete_threshold: int = 20,
        **kwargs,
    ):
        self._epochs = epochs
        self._batch_size = batch_size
        self._generator_lr = generator_lr
        self._discriminator_lr = discriminator_lr
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._user_discrete_columns = discrete_columns
        self._discrete_threshold = discrete_threshold
        self._model: Any | None = None

    def _require_ctgan(self):
        try:
            import ctgan  # noqa: F401
        except ImportError:
            raise ImportError(
                "CTGAN is not installed.\n"
                "Run: pip install ctgan torch\n\n"
                "This installs ctgan and its PyTorch dependencies."
            ) from None

    def fit(self, data: pd.DataFrame) -> "CTGANGenerator":
        self._require_ctgan()
        from ctgan import CTGAN

        self._record_schema(data)

        if self._user_discrete_columns is not None:
            discrete_cols = self._user_discrete_columns
        else:
            # Adaptive discovery: non-numeric OR low-cardinality numeric
            discrete_cols = []
            for c in self._columns:
                if not pd.api.types.is_numeric_dtype(data[c]):
                    discrete_cols.append(c)
                elif data[c].nunique() < self._discrete_threshold:
                    discrete_cols.append(c)

        self._model = CTGAN(
            epochs=self._epochs,
            batch_size=self._batch_size,
            generator_lr=self._generator_lr,
            discriminator_lr=self._discriminator_lr,
            discriminator_steps=self._discriminator_steps,
            log_frequency=self._log_frequency,
            verbose=self._verbose,
        )
        self._model.fit(data, discrete_columns=discrete_cols)
        self._fitted = True
        return self

    def _generate(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        self._require_fitted()
        self._require_ctgan()
        if self._model is None:
            raise RuntimeError("CTGAN model is not initialised. Call fit() first.")
        if seed is not None:
            if hasattr(self._model, "set_random_state"):
                self._model.set_random_state(seed)

        df = self._model.sample(n * (4 if filters else 1))
        df = self._cast_types(df)
        if filters:
            df = self._apply_filters(df, filters)
        return self._add_syn_id(df.head(n))
