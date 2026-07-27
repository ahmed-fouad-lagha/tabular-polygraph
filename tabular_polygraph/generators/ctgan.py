"""
tabular_polygraph.generators.ctgan
---------------------------------
CTGAN (Conditional Tabular GAN) generator.
"""

from __future__ import annotations

import warnings

import pandas as pd

from .base import BaseGenerator


class CTGANGenerator(BaseGenerator):
    """
    CTGAN deep tabular generator (wraps SDV's CTGANSynthesizer).
    """

    supported_types = ["cross_sectional"]

    def _init(
        self,
        epochs: int = 50,
        batch_size: int = 500,
        verbose: bool = False,
        **kwargs,
    ):
        self._epochs = epochs
        self._batch_size = batch_size
        self._verbose = verbose
        self._model = None

    def _require_sdv(self):
        try:
            from sdv.single_table import CTGANSynthesizer  # noqa: F401
        except ImportError:
            raise ImportError("SDV is not installed.") from None

    def fit(self, data: pd.DataFrame) -> "CTGANGenerator":
        self._require_sdv()
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer

        self._record_schema(data)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data[self._columns])

            self._model = CTGANSynthesizer(
                metadata=metadata,
                epochs=self._epochs,
                batch_size=self._batch_size,
                verbose=self._verbose,
            )
            assert self._model is not None
            self._model.fit(data[self._columns])

        self._fitted = True
        return self

    def _generate(
        self,
        n: int,
        filters: dict | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        self._require_sdv()
        self._require_fitted()
        if self._model is None:
            raise RuntimeError("CTGAN model is not initialised.")

        if seed is not None:
            if hasattr(self._model, "set_random_state"):
                self._model.set_random_state(seed)
            else:
                import random

                import numpy as np
                import torch

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            df = self._model.sample(n * (10 if filters else 1))

        if filters:
            df = self._apply_filters(df, filters)
            attempts = 0
            while len(df) < n and attempts < 5:
                attempts += 1
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    df_more = self._model.sample(n * 10)
                df_more = self._apply_filters(df_more, filters)
                df = pd.concat([df, df_more], ignore_index=True)

        return self._add_syn_id(df.head(n))
