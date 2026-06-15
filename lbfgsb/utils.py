# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide optimization utilities."""

from __future__ import annotations

from typing import Optional, overload

import numpy as np  # real numpy — LbfgsInvHessProduct.matvec always returns numpy
from scipy.optimize import LbfgsInvHessProduct

from lbfgsb.backend import Backend, get_backend
from lbfgsb.types import AnyArray, NDArrayFloat


def extract_hess_inv_diag(hess_inv: LbfgsInvHessProduct) -> NDArrayFloat:
    """Extract efficiently the diagonal of the L-BFGS approximate inverse Hessian.

    Relies on the linear operator ``matvec`` operation — no dense matrix is
    formed, so it remains tractable for large-scale problems.

    ``LbfgsInvHessProduct`` is a scipy object whose ``sk`` / ``yk`` correction
    pairs are always plain NumPy arrays (they are stored as numpy inside the
    solver regardless of the active backend).  The result is therefore always
    a ``NDArrayFloat``.

    Parameters
    ----------
    hess_inv : LbfgsInvHessProduct
        Linear operator for the L-BFGS approximate inverse Hessian.

    Returns
    -------
    NDArrayFloat
        Diagonal of the L-BFGS approximated inverse Hessian.
    """
    n_params: int = hess_inv.shape[0]
    hess_inv_diag: NDArrayFloat = np.zeros(n_params)
    for i in range(n_params):
        v: NDArrayFloat = np.zeros(n_params)
        v[i] = 1.0
        hess_inv_diag[i] = hess_inv.matvec(v)[i]
    return hess_inv_diag


@overload
def get_grad_projection_inf_norm(
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lbounds: NDArrayFloat,
    ubounds: NDArrayFloat,
    nx: None = ...,
) -> float: ...


@overload
def get_grad_projection_inf_norm(
    x: AnyArray,
    grad: AnyArray,
    lbounds: AnyArray,
    ubounds: AnyArray,
    nx: Backend,
) -> float: ...


def get_grad_projection_inf_norm(
    x,
    grad,
    lbounds,
    ubounds,
    nx: Optional[Backend] = None,
) -> float:
    """Return the infinity norm of the projected gradient.

    Computes ``‖x − P[x − g]‖∞`` where ``P`` is the projection onto
    ``[lbounds, ubounds]``.  This is the standard convergence criterion
    used by L-BFGS-B.

    Works with any backend (NumPy, CuPy, JAX) — the backend is inferred
    from ``x`` when ``nx`` is not supplied.

    Parameters
    ----------
    x : AnyArray
        Current parameter vector.
    grad : AnyArray
        Gradient at ``x``.
    lbounds : AnyArray
        Lower bounds (same shape as ``x``).
    ubounds : AnyArray
        Upper bounds (same shape as ``x``).
    nx : Backend, optional
        Backend to use.  Inferred from ``x`` when ``None``.

    Returns
    -------
    float
        ``max |x - clip(x - grad, lbounds, ubounds)|``.
    """
    if nx is None:
        nx = get_backend(x)
    return float(nx.max(nx.abs(x - nx.clip(x - grad, lbounds, ubounds))))
