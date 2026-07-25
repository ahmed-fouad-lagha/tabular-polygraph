from .base import BaseGenerator
from .ctgan import CTGANGenerator
from .gaussian_copula import GaussianCopulaGenerator
from .tvae import TVAEGenerator


class _LazyVine(BaseGenerator):
    """Lazy-loading wrapper for VineCopulaGenerator."""

    supported_types = ["cross_sectional"]

    def _init(self, **kwargs):
        self._kwargs = kwargs
        self._impl = None
        self._delegate = None

    def _load_impl(self):
        if self._impl is None:
            try:
                from .vine_copula import VineCopulaGenerator

                self._impl = VineCopulaGenerator
            except ImportError:
                raise ImportError(
                    "VineCopulaGenerator requires: pip install .[vine]"
                ) from None
        return self._impl

    def fit(self, data):
        cls = self._load_impl()
        self._delegate = cls(**self._kwargs)
        self._delegate.fit(data)
        self._fitted = self._delegate._fitted
        self._columns = self._delegate._columns
        self._dtypes = self._delegate._dtypes
        self._n_fit = self._delegate._n_fit
        return self

    def _generate(self, n, filters=None, seed=None):
        if self._delegate is None:
            raise RuntimeError("VineCopulaGenerator has not been fitted.")
        return self._delegate._generate(n, filters=filters, seed=seed)


VineCopulaGenerator = _LazyVine


__all__ = [
    "BaseGenerator",
    "GaussianCopulaGenerator",
    "VineCopulaGenerator",
    "CTGANGenerator",
    "TVAEGenerator",
]
