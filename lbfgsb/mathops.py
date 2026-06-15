# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

"""
mathops.py — compatibility shim (deprecated).

The global BackendShim has been replaced by the POT-style
:mod:`lbfgsb.backend` module.  This file now simply re-exports the numpy
backend's underlying module references so that any remaining internal calls
that still use the old ``from lbfgsb.mathops import np`` pattern continue
to work during the migration.

.. deprecated::
    Import from :mod:`lbfgsb.backend` instead:

    .. code-block:: python

        from lbfgsb.backend import get_backend
        nx = get_backend(x)
        result = nx.zeros(x.shape)
"""

import warnings

from scipy import optimize
from scipy.linalg import cholesky

warnings.warn(
    "lbfgsb.mathops is deprecated and will be removed in a future release. "
    "Use lbfgsb.backend.get_backend() instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Keep these names importable so old internal code doesn't break immediately.
optimize = optimize
cholesky_factorization = lambda x: cholesky(x, lower=True)  # noqa: E731


def set_backend_to_cupy() -> None:
    raise RuntimeError(
        "set_backend_to_cupy() has been removed. "
        "Pass CuPy arrays directly to minimize_lbfgsb(); "
        "the backend is inferred automatically."
    )


def set_backend_to_defaults() -> None:
    raise RuntimeError(
        "set_backend_to_defaults() has been removed. "
        "The backend is inferred automatically from your array types."
    )
