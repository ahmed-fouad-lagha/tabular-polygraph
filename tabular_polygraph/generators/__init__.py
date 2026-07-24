from .base import BaseGenerator
from .ctgan import CTGANGenerator
from .dp_gaussian_copula import DPGaussianCopulaGenerator
from .forest_diffusion import ForestDiffusionGenerator
from .gaussian_copula import GaussianCopulaGenerator


def VineCopulaGenerator(*args, **kwargs):
    try:
        from .vine_copula import VineCopulaGenerator as _V

        return _V(*args, **kwargs)
    except ImportError:
        raise ImportError("VineCopulaGenerator requires: pip install .[vine]") from None


__all__ = [
    "BaseGenerator",
    "GaussianCopulaGenerator",
    "DPGaussianCopulaGenerator",
    "VineCopulaGenerator",
    "CTGANGenerator",
    "ForestDiffusionGenerator",
]
