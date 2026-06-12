# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

r"""
Implement the line search algorithm by Moré and Thuente (1994),
currently used for the L-BFGS-B algorithm.

The target of this line search algorithm is to find a step size :math:`\alpha`
that satisfies the strong Wolfe conditions

.. math::

    f(x + \alpha d) \leq f(x) + \alpha \mu g(x)^T d

and

.. math::

    |g(x + \alpha d)^T d| \leq \eta |g(x)^T d|.

Functions
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    max_allowed_steplength
    line_search

References
----------
[1] Moré, J. J., & Thuente, D. J. (1994). Line search algorithms with guaranteed
    sufficient decrease.
"""

import logging
import warnings
from typing import Optional

import numpy as tnp
from packaging.version import Version
from scipy import __version__ as spversion
from scipy.optimize import _dcsrch

from lbfgsb._numba_helpers import njit
from lbfgsb.mathops import errorstate, np, sp
from lbfgsb.scalar_function import ScalarFunction
from lbfgsb.types import NDArrayFloat

_FLOAT64_MAX = np.finfo(np.float64).max


def _safe_sumsq(x: NDArrayFloat) -> float:
    """Return ``sum(x**2)`` without emitting overflow warnings.

    If the squared norm cannot be represented as a finite ``float64``, return
    ``np.inf``.

    Parameters
    ----------
    x : NDArrayFloat
        Input vector.

    Returns
    -------
    float
        Squared Euclidean norm, or ``np.inf`` if it cannot be represented
        finitely.
    """
    if x.size == 0:
        return 0.0

    if not np.all(np.isfinite(x)):
        return float("inf")

    max_abs = float(np.max(np.abs(x)))

    if max_abs == 0.0:
        return 0.0

    if not np.isfinite(max_abs):
        return float("inf")

    y = x / max_abs

    with np.errstate(over="ignore", invalid="ignore"):
        scaled_sum = float(np.dot(y, y))

    if not np.isfinite(scaled_sum) or scaled_sum <= 0.0:
        return float("inf")

    limit = np.sqrt(_FLOAT64_MAX / scaled_sum)

    if max_abs > limit:
        return float("inf")

    return float(max_abs * max_abs * scaled_sum)


def _safe_dot(a: NDArrayFloat, b: NDArrayFloat) -> float:
    """Return ``dot(a, b)`` while avoiding floating-point warnings.

    Parameters
    ----------
    a, b : NDArrayFloat
        Input vectors.

    Returns
    -------
    float
        Dot product if finite, otherwise ``np.inf``.
    """
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("inf")

    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.dot(a, b))

    if not np.isfinite(value):
        return float("inf")

    return value


def _trial_point(
    x0: NDArrayFloat,
    alpha: float,
    d: NDArrayFloat,
) -> Optional[NDArrayFloat]:
    """Return ``x0 + alpha * d`` if finite, otherwise ``None``."""
    if not np.isfinite(alpha):
        return None

    with np.errstate(over="ignore", invalid="ignore"):
        x = x0 + alpha * d

    if not np.all(np.isfinite(x)):
        return None

    return x


def max_allowed_steplength(
    x: NDArrayFloat,
    d: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    max_steplength: float,
    n_iter: int,
) -> float:
    r"""
    Compute the largest admissible nonnegative step length.

    The returned step length ``alpha`` satisfies

    .. math::

        lb \leq x + \alpha d \leq ub

    and

    .. math::

        0 \leq \alpha \leq \mathrm{max\_steplength}.

    Parameters
    ----------
    x : NDArrayFloat
        Starting point.
    d : NDArrayFloat
        Search direction.
    lb : NDArrayFloat
        Lower bounds.
    ub : NDArrayFloat
        Upper bounds.
    max_steplength : float
        Maximum step length allowed by the caller.
    n_iter : int
        Current number of outer iterations.

    Returns
    -------
    float
        Maximum admissible step length.

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing, 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, 23, 4, pp. 550-560.
    * J. L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778:
      L-BFGS-B, FORTRAN routines for large scale bound constrained optimization
      (2011), ACM Transactions on Mathematical Software, 38, 1.
    """
    if n_iter == 0:
        return 1.0

    if not np.isfinite(max_steplength) or max_steplength <= 0.0:
        return 0.0

    if not np.all(np.isfinite(d)):
        return 0.0

    mask = d != 0.0

    if not np.any(mask):
        return float(max_steplength)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        candidates = np.where(
            d[mask] > 0.0,
            (ub[mask] - x[mask]) / d[mask],
            (lb[mask] - x[mask]) / d[mask],
        )

    candidates = candidates[np.isfinite(candidates)]
    candidates = candidates[candidates >= 0.0]

    if candidates.size == 0:
        return float(max_steplength)

    return float(min(max_steplength, float(np.min(candidates))))


@njit(cache=True)
def max_allowed_steplength_numba(
    x: NDArrayFloat,
    d: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    max_steplength: float,
    n_iter: int,
) -> float:
    """See :func:`max_allowed_steplength`."""
    if n_iter == 0:
        return 1.0

    if not np.isfinite(max_steplength) or max_steplength <= 0.0:
        return 0.0

    alpha = max_steplength
    found = False

    n = x.size
    for i in range(n):
        di = d[i]

        if not np.isfinite(di):
            return 0.0

        if di != 0.0:
            if di > 0.0:
                tmp = (ub[i] - x[i]) / di
            else:
                tmp = (lb[i] - x[i]) / di

            if np.isfinite(tmp) and tmp >= 0.0:
                if not found or tmp < alpha:
                    alpha = tmp
                    found = True

    return alpha


def line_search(
    x0: NDArrayFloat,
    f0: float,
    g0: NDArrayFloat,
    d: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    above_iter: int,
    max_steplength_user: float,
    is_boxed: bool,
    sf: ScalarFunction,
    ftol: float = 1e-3,
    gtol: float = 0.9,
    xtol: float = 1e-1,
    max_iter: int = 30,
    iprint: int = 10,
    logger: Optional[logging.Logger] = None,
    isave: Optional[np.typing.NDArray[np.intc]] = None,
    dsave: Optional[NDArrayFloat] = None,
    is_use_numba_jit: bool = False,
) -> Optional[float]:
    r"""
    Find a step satisfying sufficient decrease and curvature conditions.

    The line search attempts to find a step length ``alpha`` such that

    .. math::

        f(x_0 + \alpha d)
        \leq
        f(x_0) + c_1 \alpha \nabla f(x_0)^T d

    and

    .. math::

        |\nabla f(x_0 + \alpha d)^T d|
        \leq
        c_2 |\nabla f(x_0)^T d|.

    These are the strong Wolfe conditions.

    Notes
    -----
    For SciPy versions older than 1.12, this routine calls the historical
    Minpack2 ``dcsrch`` implementation. For SciPy 1.12 and above, it uses
    SciPy's Python implementation.

    Parameters
    ----------
    x0 : NDArrayFloat
        Starting point.
    f0 : float
        Objective value at ``x0``.
    g0 : NDArrayFloat
        Gradient at ``x0``.
    d : NDArrayFloat
        Search direction.
    lb : NDArrayFloat
        Lower bounds.
    ub : NDArrayFloat
        Upper bounds.
    above_iter : int
        Current outer iteration number.
    max_steplength_user : float
        User-provided maximum step length.
    is_boxed : bool
        Whether all variables have both lower and upper bounds.
    sf : ScalarFunction
        Objective and gradient wrapper.
    ftol : float, optional
        Sufficient-decrease Wolfe parameter :math:`c_1`. Default is ``1e-3``.
    gtol : float, optional
        Curvature Wolfe parameter :math:`c_2`. Default is ``0.9``.
    xtol : float, optional
        Relative tolerance for acceptable steps in the line-search procedure.
        Default is ``1e-1``.
    max_iter : int, optional
        Maximum number of line-search iterations. Default is ``30``.
    iprint : int, optional
        Verbosity level. Default is ``10``.
    logger : logging.Logger, optional
        Logger used for textual output.
    isave : ndarray, optional
        Integer workspace used by the legacy Minpack2 implementation.
    dsave : ndarray, optional
        Floating-point workspace used by the legacy Minpack2 implementation.
    is_use_numba_jit : bool, optional
        Whether to use Numba-compiled helper functions. Default is ``False``.

    Returns
    -------
    float or None
        Step length if the line search succeeds or returns a usable warning
        state. ``None`` otherwise.
    """
    if isave is None:
        isave = np.zeros((2,), dtype=np.intc)

    if dsave is None:
        dsave = np.zeros((13,), dtype=np.float64)

    if max_iter <= 0:
        return None

    if not np.isfinite(f0):
        return None

    if not np.all(np.isfinite(x0)):
        return None

    if not np.all(np.isfinite(g0)):
        return None

    if not np.all(np.isfinite(d)):
        return None

    if is_use_numba_jit:
        max_steplength = max_allowed_steplength_numba(
            x0,
            d,
            lb,
            ub,
            max_steplength_user,
            above_iter,
        )
    else:
        max_steplength = max_allowed_steplength(
            x0,
            d,
            lb,
            ub,
            max_steplength_user,
            above_iter,
        )

    if not np.isfinite(max_steplength) or max_steplength <= 0.0:
        return None

    dphi0 = _safe_dot(g0, d)

    if not np.isfinite(dphi0):
        return None

    # Moré-Thuente line search requires a descent direction.
    if dphi0 >= 0.0:
        return None

    if above_iter == 0 and not is_boxed:
        dd = _safe_sumsq(d)

        if not np.isfinite(dd) or dd <= 0.0:
            return None

        steplength_0 = min(1.0 / np.sqrt(dd), max_steplength)
    else:
        steplength_0 = min(1.0, max_steplength)

    if not np.isfinite(steplength_0) or steplength_0 <= 0.0:
        return None

    # Support for Python 3.7 and 3.8: the minpack2 wrapper was removed from
    # SciPy 1.12 and replaced by a Python implementation. Python 3.7 and 3.8
    # cannot use SciPy 1.12, so the legacy Minpack2 path is still needed.
    is_use_minpack2: bool = Version(spversion) < Version("1.12")

    def phi(alpha: float) -> float:
        """Return objective value for step length ``alpha``."""
        x_trial = _trial_point(x0, alpha, d)

        if x_trial is None:
            return float("inf")

        value = sf.fun(x_trial)

        if not np.isfinite(value):
            return float("inf")

        return float(value)

    def dphi(alpha: float) -> float:
        """Return directional derivative for step length ``alpha``."""
        x_trial = _trial_point(x0, alpha, d)

        if x_trial is None:
            return float("nan")

        grad = sf.grad(x_trial)

        if not np.all(np.isfinite(grad)):
            return float("nan")

        value = _safe_dot(grad, d)

        if not np.isfinite(value):
            return float("nan")

        return float(value)

    task = b"START"
    f_m1 = float(f0)
    dphi_m1 = float(dphi0)
    _iter = 0

    steplength: Optional[float] = None
    best_stp: Optional[float] = None

    if not is_use_minpack2:
        # careful, there is an issue in the DCSRRCH.__call__ function. It returns
        # steplength = None when task is a warning while it should not be the case
        # for instance when steplength = max_steplength
        # So we must implement a while loop again.
        # steplength, f0, _, task = dcsrch(
        #     steplength_0, phi0=f0, derphi0=dphi0, maxiter=max_iter
        # )
        dcsrch = _dcsrch.DCSRCH(phi, dphi, ftol, gtol, xtol, 0.0, max_steplength)

    while _iter < max_iter:
        if is_use_minpack2:
            # SciPy older than 1.12 uses the Fortran Minpack2 implementation.
            with warnings.catch_warnings():
                # scipy.optimize.minpack2 may be deprecated on newer SciPy
                # versions, but this path is needed for Python 3.7/3.8
                # compatibility.
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                steplength_raw, f0_raw, dphi0_raw, task = sp.optimize.minpack2.dcsrch(
                    steplength_0,
                    f_m1,
                    dphi_m1,
                    ftol,
                    gtol,
                    xtol,
                    task,
                    0,
                    max_steplength,
                    isave,
                    dsave,
                )

            steplength = None if steplength_raw is None else float(steplength_raw)
            f0 = float(f0_raw)
            dphi0 = float(dphi0_raw)

        else:
            # SciPy >= 1.12 uses the Python implementation.
            steplength_raw, f0_raw, dphi0_raw, task = dcsrch._iterate(
                steplength_0,
                f_m1,
                dphi_m1,
                task,
            )

            steplength = None if steplength_raw is None else float(steplength_raw)

        if task[:2] == b"FG":
            if steplength is None:
                return None

            if not np.isfinite(steplength) or steplength <= 0.0:
                return None

            x_trial = _trial_point(x0, steplength, d)

            if x_trial is None:
                return None

            stp_old = float(steplength_0)
            f_m1_old = float(f_m1)

            steplength_0 = float(steplength)

            f_new, g_new = sf.fun_and_grad(x_trial)

            if not np.isfinite(f_new):
                return None

            if not np.all(np.isfinite(g_new)):
                return None

            dphi_new = _safe_dot(g_new, d)

            if not np.isfinite(dphi_new):
                return None

            f_m1 = float(f_new)
            dphi_m1 = float(dphi_new)

            if f_m1 < f_m1_old:
                best_stp = steplength_0
            else:
                best_stp = stp_old

        else:
            break

        _iter += 1

    else:
        # max_iter reached: the line search did not converge.
        task = b"WARNING: dcsrch did not converge within max iterations"

    if steplength is None:
        return None

    if not np.isfinite(steplength) or steplength <= 0.0:
        return None

    if task[:4] != b"CONV" and task[:4] != b"WARN":
        return None

    if best_stp is not None:
        steplength = best_stp

    if not np.isfinite(steplength) or steplength <= 0.0:
        return None

    if iprint >= 99 and logger is not None:
        dd = _safe_sumsq(d)

        if np.isfinite(dd):
            norm_step = steplength * np.sqrt(dd)
        else:
            norm_step = float("inf")

        logger.info(f"LINE SEARCH {_iter} times; norm of step = {norm_step}")

    return float(steplength)
