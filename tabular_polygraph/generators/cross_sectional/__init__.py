from .dp_gaussian_copula import DPGaussianCopulaGenerator
from .gaussian_copula import GaussianCopulaGenerator


def VineCopulaGenerator(*args, **kwargs):
    try:
        from .vine_copula import VineCopulaGenerator as _V

        return _V(*args, **kwargs)
    except ImportError:
        raise ImportError(
            "VineCopulaGenerator requires: pip install src[vine]"
        ) from None


__all__ = [
    "GaussianCopulaGenerator",
    "DPGaussianCopulaGenerator",
    "VineCopulaGenerator",
]
