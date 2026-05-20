"""
prysm's interchangeable backend system for supporting numpy-like backends:
    https://github.com/brandondube/prysm/blob/master/prysm/mathops.py
"""

import warnings

import numpy as np
import scipy as sp
from scipy import optimize


class BackendShim:
    """A shim that allows a backend to be swapped at runtime."""

    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == "_srcmodule":
            return self._srcmodule

        return getattr(self._srcmodule, key)


_np = np
_sp = sp
_optimize = optimize

np = BackendShim(np)
sp = BackendShim(sp)
optimize = BackendShim(optimize)


def set_backend_to_cupy():
    """Convenience method to automatically configure prysm's backend to cupy."""

    try:
        import cupy as cp
        import cupyx.scipy as csp
        import cupyx.scipy.optimize as csp_optimize
    except ImportError:
        warnings.warn("cupy not installed, backend remains numpy")
        return

    np._srcmodule = cp
    sp._srcmodule = csp
    optimize._srcmodule = csp_optimize

    return


def set_backend_to_defaults():
    """Convenience method to restore prysm's default backend options."""
    np._srcmodule = _np
    sp._srcmodule = _sp
    optimize._srcmodule = _optimize
    return
