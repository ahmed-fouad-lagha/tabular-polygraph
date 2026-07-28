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
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*SingleTableMetadata.*")
            warnings.filterwarnings("ignore", message=".*save_to_json.*")

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data[self._columns])

            self._model = CTGANSynthesizer(
                metadata=metadata,
                epochs=self._epochs,
                batch_size=self._batch_size,
                verbose=self._verbose,
            )
            if self._model is None:
                raise RuntimeError("CTGAN model failed to initialise.")
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

        def _sample(count: int) -> pd.DataFrame:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                df = self._model.sample(count)
            return self._cast_types(df)

        return self._generate_with_retry(n, filters, _sample)
