"""
tabular_polygraph.generators.tvae
---------------------------------
TVAE (Tabular Variational Autoencoder) generator.

TVAE is a VAE-based tabular data generator from the SDV ecosystem.
It typically produces smoother distributions than CTGAN and is more
stable to train, though it may under-represent minority modes.

Requirements:
    pip install sdv   # installs sdv which includes TVAE
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from .base import BaseGenerator


class TVAEGenerator(BaseGenerator):
    """
    TVAE deep tabular generator (wraps SDV's TVAESynthesizer).

    Usage (same interface as GaussianCopulaGenerator):
        gen = TVAEGenerator(epochs=300, batch_size=500)
        gen.fit(real_df)
        syn = gen.generate(1000)
    """

    def _init(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        embedding_dim: int = 128,
        compress_dims: tuple[int, ...] = (128, 128),
        decompress_dims: tuple[int, ...] = (128, 128),
        l2scale: float = 1e-5,
        loss_factor: float = 2,
        verbose: bool = False,
        **kwargs,
    ):
        self._epochs = epochs
        self._batch_size = batch_size
        self._embedding_dim = embedding_dim
        self._compress_dims = compress_dims
        self._decompress_dims = decompress_dims
        self._l2scale = l2scale
        self._loss_factor = loss_factor
        self._verbose = verbose
        self._model: Any = None

    def _require_sdv(self):
        try:
            from sdv.single_table import TVAESynthesizer  # noqa: F401
        except ImportError:
            raise ImportError("SDV is not installed.") from None

    def fit(self, data: pd.DataFrame) -> "TVAEGenerator":
        self._require_sdv()
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import TVAESynthesizer

        self._record_schema(data)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*SingleTableMetadata.*")
            warnings.filterwarnings("ignore", message=".*save_to_json.*")

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data[self._columns])

            self._model = TVAESynthesizer(
                metadata=metadata,
                epochs=self._epochs,
                batch_size=self._batch_size,
                embedding_dim=self._embedding_dim,
                compress_dims=self._compress_dims,
                decompress_dims=self._decompress_dims,
                l2scale=self._l2scale,
                loss_factor=self._loss_factor,
                verbose=self._verbose,
            )
            if self._verbose:
                from tabular_polygraph.io.console import info

                info(
                    "    Pre-processing data (fitting Gaussian Mixtures)... this can take a few minutes before the progress bar appears."
                )
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
        from sdv.sampling import Condition

        if seed is not None:
            self._model.set_random_state(seed)

        if not filters:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                df = self._model.sample(n)
            return self._cast_types(df)

        exact_filters = {}
        post_filters = {}
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
