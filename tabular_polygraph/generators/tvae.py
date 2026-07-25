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

    supported_types = ["cross_sectional"]

    def _init(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        embedding_dim: int = 128,
        compress_dims: tuple[int, ...] = (128, 128),
        decompress_dims: tuple[int, ...] = (128, 128),
        l2scale: float = 1e-5,
        loss_factor: float = 2,
        discrete_threshold: int = 20,
        **kwargs,
    ):
        self._epochs = epochs
        self._batch_size = batch_size
        self._embedding_dim = embedding_dim
        self._compress_dims = compress_dims
        self._decompress_dims = decompress_dims
        self._l2scale = l2scale
        self._loss_factor = loss_factor
        self._discrete_threshold = discrete_threshold
        self._model = None

    def _require_sdv(self):
        try:
            from sdv.single_table import TVAESynthesizer  # noqa: F401
        except ImportError:
            raise ImportError(
                "SDV is not installed.\n"
                "Run: pip install sdv\n\n"
                "This installs sdv and its dependencies."
            ) from None

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
                verbose=False,
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
            raise RuntimeError("TVAE model is not initialised. Call fit() first.")

        if seed is not None:
            if hasattr(self._model, "set_random_state"):
                self._model.set_random_state(seed)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            df = self._model.sample(n * (10 if filters else 1))

        df = self._cast_types(df)
        if filters:
            df = self._apply_filters(df, filters)
            attempts = 0
            while len(df) < n and attempts < 5:
                attempts += 1
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    df_more = self._model.sample(n * 10)
                df_more = self._cast_types(df_more)
                df_more = self._apply_filters(df_more, filters)
                df = pd.concat([df, df_more], ignore_index=True)

        return self._add_syn_id(df.head(n))
