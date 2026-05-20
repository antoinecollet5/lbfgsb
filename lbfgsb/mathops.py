"""
prysm's interchangeable backend system for supporting numpy-like backends:
    https://github.com/brandondube/prysm/blob/master/prysm/mathops.py
"""

import warnings

import numpy as np
import scipy as sp
import scipy.linalg as la
from scipy import optimize
from scipy.linalg import cholesky


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
_la = la


# Go through alias to handle that scipy and cupy have different defaults,
# lbfgsb requires that lower=True, which is cupy's default, but not scipy's


# This was failing with lambda functions
_cholesky = lambda x: cholesky(x, lower=True)


np = BackendShim(np)
sp = BackendShim(sp)
optimize = BackendShim(optimize)
la = BackendShim(la)
cholesky_factorization = _cholesky


def set_backend_to_cupy():
    """Convenience method to automatically configure prysm's backend to cupy."""

    try:
        import cupy as cp
        import cupy.linalg as cla
        import cupyx.scipy as csp
        from cupy.linalg import cholesky as ccholesky

    except ImportError:
        warnings.warn("cupy not installed, backend remains numpy")
        return

    np._srcmodule = cp
    sp._srcmodule = csp
    la._srcmodule = cla

    # Create a wrapper that passes lower=True to cupy's cholesky
    cholesky_factorization = ccholesky

    return


def set_backend_to_defaults():
    """Convenience method to restore prysm's default backend options."""
    np._srcmodule = _np
    sp._srcmodule = _sp
    optimize._srcmodule = _optimize
    la._srcmodule = _la
    cholesky = _cholesky  # Restore scipy's cholesky with lower=True

    return
