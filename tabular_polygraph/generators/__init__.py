from .base import BaseGenerator
from .ctgan import CTGANGenerator
from .gaussian_copula import GaussianCopulaGenerator
from .tvae import TVAEGenerator


def VineCopulaGenerator(*args, **kwargs):
    try:
        from .vine_copula import VineCopulaGenerator as _V

        return _V(*args, **kwargs)
    except ImportError:
        raise ImportError("VineCopulaGenerator requires: pip install .[vine]") from None


__all__ = [
    "BaseGenerator",
    "GaussianCopulaGenerator",
    "VineCopulaGenerator",
    "CTGANGenerator",
    "TVAEGenerator",
]
