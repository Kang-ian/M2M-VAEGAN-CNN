"""Synthesizers module."""

from .ctgan import CTGAN
from .PreCVGMtvaeganConv1d import PRECVGMTVAEGANConv1d


__all__ = ('CTGAN','PRECVGMTVAEGANConv1d')

def get_all_synthesizers():
    return {name: globals()[name] for name in __all__}
