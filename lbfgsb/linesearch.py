# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

r"""
Moré-Thuente line search for L-BFGS-B.

All array operations go through the ``nx`` backend instance, so this module
works with NumPy, CuPy, and JAX arrays without modification.
The scalar engine (:mod:`lbfgsb.dcsrch`) operates purely on Python floats,
eliminating the last hard NumPy dependency.

Functions
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    max_allowed_steplength
    line_search
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from lbfgsb._numba_helpers import njit
from lbfgsb.backend import Backend, get_backend
from lbfgsb.dcsrch import DcsrchState, dcsrch
from lbfgsb.scalar_function import ScalarFunction
from lbfgsb.types import AnyArray, NDArrayFloat

# ---------------------------------------------------------------------------
# Helpers — all scalar output, work on any backend via nx
# ---------------------------------------------------------------------------


def _safe_sumsq(nx: Backend, x: AnyArray) -> float:
    """Return ``sum(x**2)`` safely; ``inf`` on overflow or non-finite input."""
    if x.size == 0:
        return 0.0
    if not nx.all(nx.isfinite(x)):
        return math.inf

    max_abs = nx.max(nx.abs(x))
    if max_abs == 0.0:
        return 0.0

    with nx.errstate(over="ignore", invalid="ignore"):
        y = x / max_abs
        scaled_sum = float(nx.dot(y, y))

    if not math.isfinite(scaled_sum) or scaled_sum <= 0.0:
        return math.inf

    fmax = nx.finfo_max()
    limit = math.sqrt(fmax / scaled_sum)
    if max_abs > limit:
        return math.inf

    return float(max_abs * max_abs * scaled_sum)


def _safe_dot(nx: Backend, a: AnyArray, b: AnyArray) -> float:
    """Return ``dot(a, b)`` as a Python float; ``inf`` on overflow."""
    if not nx.all(nx.isfinite(a)) or not nx.all(nx.isfinite(b)):
        return math.inf
    with nx.errstate(over="ignore", invalid="ignore"):
        value = float(nx.dot(a, b))
    return value if math.isfinite(value) else math.inf


def _trial_point(
    nx: Backend,
    x0: AnyArray,
    alpha: float,
    d: AnyArray,
) -> Optional[AnyArray]:
    """Return ``x0 + alpha * d`` if finite, else ``None``."""
    if not math.isfinite(alpha):
        return None
    with nx.errstate(over="ignore", invalid="ignore"):
        x = x0 + d * alpha
    return x if nx.all(nx.isfinite(x)) else None  # ty:ignore[invalid-return-type]


# ---------------------------------------------------------------------------
# Max admissible step length
# ---------------------------------------------------------------------------


def max_allowed_steplength(
    nx: Backend,
    x: AnyArray,
    d: AnyArray,
    lb: AnyArray,
    ub: AnyArray,
    max_steplength: float,
    n_iter: int,
) -> float:
    """Compute the largest step keeping ``x + alpha * d`` within [lb, ub]."""
    if n_iter == 0:
        return 1.0
    if not math.isfinite(max_steplength) or max_steplength <= 0.0:
        return 0.0
    if not nx.all(nx.isfinite(d)):
        return 0.0

    # Boolean mask where d != 0
    mask = d != 0.0
    if not nx.any(mask):
        return float(max_steplength)

    with nx.errstate(divide="ignore", invalid="ignore", over="ignore"):
        candidates = nx.where(
            d[mask] > 0.0,
            (ub[mask] - x[mask]) / d[mask],
            (lb[mask] - x[mask]) / d[mask],
        )

    finite_mask = nx.isfinite(candidates)
    candidates = candidates[finite_mask]
    candidates = candidates[candidates >= 0.0]

    if candidates.size == 0:
        return float(max_steplength)

    return float(min(max_steplength, nx.max(candidates)))  # type: ignore[arg-type]


@njit(cache=True)
def _max_allowed_steplength_numba(
    x: NDArrayFloat,
    d: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    max_steplength: float,
    n_iter: int,
) -> float:
    """Numba-JIT version of :func:`max_allowed_steplength`."""
    import numpy as np  # numba sees real numpy

    if n_iter == 0:
        return 1.0
    if not np.isfinite(max_steplength) or max_steplength <= 0.0:
        return 0.0
    alpha = max_steplength
    found = False
    for i in range(x.size):
        di = d[i]
        if not np.isfinite(di):
            return 0.0
        if di != 0.0:
            tmp = (ub[i] - x[i]) / di if di > 0.0 else (lb[i] - x[i]) / di
            if np.isfinite(tmp) and tmp >= 0.0:
                if not found or tmp < alpha:
                    alpha = tmp
                    found = True
    return alpha


# ---------------------------------------------------------------------------
# Main line search
# ---------------------------------------------------------------------------


def line_search(
    x0: AnyArray,
    f0: float,
    g0: AnyArray,
    d: AnyArray,
    lb: AnyArray,
    ub: AnyArray,
    above_iter: int,
    max_steplength_user: float,
    is_boxed: bool,
    sf: ScalarFunction,
    nx: Optional[Backend] = None,
    ftol: float = 1e-3,
    gtol: float = 0.9,
    xtol: float = 1e-1,
    max_iter: int = 30,
    iprint: int = 10,
    logger: Optional[logging.Logger] = None,
    is_use_numba_jit: bool = False,
) -> Optional[float]:
    r"""Find a step length satisfying the strong Wolfe conditions.

    Parameters
    ----------
    x0 : AnyArray
        Starting point.
    f0 : float
        Objective value at ``x0``.
    g0 : AnyArray
        Gradient at ``x0``.
    d : AnyArray
        Search direction.
    lb, ub : AnyArray
        Bound vectors.
    above_iter : int
        Current outer iteration number.
    max_steplength_user : float
        Caller-imposed step-length ceiling.
    is_boxed : bool
        Whether all variables have finite bounds on both sides.
    sf : ScalarFunction
        Memoized objective/gradient wrapper.
    nx : Backend, optional
        Backend to use. Inferred from ``x0`` when ``None``.
    ftol : float
        Sufficient-decrease (Armijo) parameter c₁.
    gtol : float
        Curvature (Wolfe) parameter c₂.
    xtol : float
        Relative step-interval width tolerance.
    max_iter : int
        Maximum line-search iterations.
    iprint : int
        Verbosity level.
    logger : logging.Logger, optional
        Logger for diagnostic output.
    is_use_numba_jit : bool
        Use Numba-compiled ``max_allowed_steplength`` kernel.

    Returns
    -------
    float or None
        Accepted step length, or ``None`` if the line search fails.
    """
    if nx is None:
        nx = get_backend(x0)

    if max_iter <= 0:
        return None
    if not math.isfinite(f0):
        return None
    if not nx.all(nx.isfinite(x0)):
        return None
    if not nx.all(nx.isfinite(g0)):
        return None
    if not nx.all(nx.isfinite(d)):
        return None

    # --- Max admissible step ---
    if is_use_numba_jit:
        # Numba path: arrays must be numpy (guaranteed when NumbaBackend is used)
        max_steplength = _max_allowed_steplength_numba(
            x0,
            d,
            lb,
            ub,
            max_steplength_user,
            above_iter,  # type: ignore[arg-type]
        )
    else:
        max_steplength = max_allowed_steplength(
            nx, x0, d, lb, ub, max_steplength_user, above_iter
        )

    if not math.isfinite(max_steplength) or max_steplength <= 0.0:
        return None

    dphi0 = _safe_dot(nx, g0, d)
    if not math.isfinite(dphi0) or dphi0 >= 0.0:
        return None

    if above_iter == 0 and not is_boxed:
        dd = _safe_sumsq(nx, d)
        if not math.isfinite(dd) or dd <= 0.0:
            return None
        steplength_0 = min(1.0 / math.sqrt(dd), max_steplength)
    else:
        steplength_0 = min(1.0, max_steplength)

    if not math.isfinite(steplength_0) or steplength_0 <= 0.0:
        return None

    # --- Pure-Python Moré-Thuente engine (no array library inside) ---
    # Protocol (mirrors scipy's DCSRCH._iterate):
    #   1. First call with (stp0, f0, dphi0) → initialization, returns (stp0, "FG")
    #   2. Caller evaluates phi(stp0), dphi(stp0)
    #   3. Subsequent calls with (stp, phi(stp), dphi(stp)) → new stp or CONV/WARN
    state = DcsrchState()

    # Step 1: initialise (returns stp unchanged and "FG")
    stp, f_cur, dphi_cur, task = dcsrch(
        steplength_0,
        f0,
        dphi0,
        ftol,
        gtol,
        xtol,
        0.0,
        max_steplength,
        state,
    )
    if task != "FG":
        return None  # ERROR on initialization

    steplength: Optional[float] = None
    best_stp: Optional[float] = None
    f_old = f0

    for _iter in range(max_iter):
        # Evaluate phi and dphi at the proposed stp
        x_trial = _trial_point(nx, x0, stp, d)
        if x_trial is None:
            return None

        f_new, g_new = sf.fun_and_grad(x_trial)
        if not math.isfinite(f_new):
            return None
        if not nx.all(nx.isfinite(g_new)):
            return None

        dphi_new = _safe_dot(nx, g_new, d)
        if not math.isfinite(dphi_new):
            return None

        stp_prev = stp
        stp, f_cur, dphi_cur, task = dcsrch(
            stp,
            float(f_new),
            float(dphi_new),
            ftol,
            gtol,
            xtol,
            0.0,
            max_steplength,
            state,
        )

        steplength = stp_prev
        if float(f_new) < f_old:
            best_stp = stp_prev
        f_old = float(f_new)

        if task in ("CONV", "WARN"):
            break
        elif task == "ERROR":
            return None
        # task == "FG": continue loop with new stp

    if steplength is None or not math.isfinite(steplength) or steplength <= 0.0:
        return None

    if best_stp is not None:
        steplength = best_stp

    if iprint >= 99 and logger is not None:
        dd = _safe_sumsq(nx, d)
        norm_step = steplength * math.sqrt(dd) if math.isfinite(dd) else math.inf
        logger.info(f"LINE SEARCH {_iter + 1} times; norm of step = {norm_step}")

    return float(steplength)
