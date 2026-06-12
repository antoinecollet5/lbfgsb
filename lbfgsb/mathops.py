"""
prysm's interchangeable backend system for supporting numpy-like backends:
    https://github.com/brandondube/prysm/blob/master/prysm/mathops.py
"""

import warnings
from ast import arg

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

np = BackendShim(np)
sp = BackendShim(sp)
optimize = BackendShim(optimize)
la = BackendShim(la)


# The numpy wrapper
def _np_cholesky_factorization(x):
    """Wrapper around cholesky factorization that passes lower=True to cupy's cholesky."""
    return cholesky(x, lower=True)


def _np_errstate(*args, **kwargs):
    return np.errstate(*args, **kwargs)


_backend = {
    "cholesky": _np_cholesky_factorization,
    "errstate": _np_errstate,
}


# Public facing functions
def cholesky_factorization(x):
    return _backend["cholesky"](x)


# Unused - currently linesearch uses tnp.errorstate
def errorstate(*args, **kwargs):
    return _backend["errstate"](*args, **kwargs)


def set_backend_to_cupy():
    """Convenience method to automatically configure prysm's backend to cupy."""

    try:
        import cupy as cp
        import cupy.linalg as cla
        import cupyx as cpx
        import cupyx.scipy as csp
        from cupy.linalg import cholesky as ccholesky

    except ImportError:
        warnings.warn("cupy not installed, backend remains numpy")
        return

    np._srcmodule = cp
    sp._srcmodule = csp
    la._srcmodule = cla

    # Define the cupy wrapper
    def _cp_cholesky_factorization(x):
        """
        Handling that cupy doesn't support lower=True, just uses by default
        """
        return ccholesky(x)

    def _cp_errstate(*args, **kwargs):
        return cpx.errstate(*args, **kwargs)

    # modify the backend to use cupy's cholesky wrapper
    _backend["cholesky"] = _cp_cholesky_factorization
    _backend["errstate"] = _cp_errstate

    return


def set_backend_to_defaults():
    """Convenience method to restore prysm's default backend options."""
    np._srcmodule = _np
    sp._srcmodule = _sp
    optimize._srcmodule = _optimize
    la._srcmodule = _la

    # modify the backend to use cupy's cholesky wrapper
    _backend["cholesky"] = _np_cholesky_factorization
    _backend["errstate"] = _np_errstate

    return
