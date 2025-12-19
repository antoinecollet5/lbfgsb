# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

"""
Implement a function to compute the generalized Cauchy point (GCP) for the L-BFGS-B
algorithm, mainly for internal use.

The target of the GCP procedure is to find a step size t such that
x(t) = x0 - t * g is a local minimum of the quadratic function m(x),
where m(x) is a local approximation to the objective function.

First determine a sequence of break points t0=0, t1, t2, ..., tn.
On each interval [t[i-1], t[i]], x is changing linearly.
After passing a break point, one or more coordinates of x will be fixed at the bounds.
We search the first local minimum of m(x) by examining the intervals [t[i-1], t[i]]
sequentially.

Functions
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    get_cauchy_point

Reference:
[1] R. H. Byrd, P. Lu, and J. Nocedal (1995). A limited memory algorithm for bound
constrained optimization.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from lbfgsb._numba_helpers import njit
from lbfgsb.bfgsmats import LBFGSB_MATRICES, bmv, bmv_numba
from lbfgsb.types import NDArrayFloat, NDArrayInt


def display_start_point(
    nseg: int,
    f_prime: float,
    f_second: float,
    delta_t: Optional[float],
    delta_t_min: float,
    iprint: int,
    logger: Optional[logging.Logger],
) -> None:
    """
    Display the start point status.

    Parameters
    ----------
    nseg : int
        Number of explored segment.
    f_prime : float
        First derivative.
    f_second : float
        Second derivative.
    delta_t : float
        See Algorithm CP: Computation of the generalized Cauchy point in [1].
    delta_t_min : float
        See Algorithm CP: Computation of the generalized Cauchy point in [1].
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99``   print details of every iteration except n-vectors;
    logger: Optional[Logger], optional
        :class:`logging.Logger` instance. If None, nothing is displayed, no matter the
        value of `iprint`, by default None.

    """
    if iprint < 100:
        return
    if logger is None:
        return
    logger.info(f"Piece    , {nseg},  --f1, f2 at start point , {f_prime} , {f_second}")
    if delta_t is not None:
        logger.info(f"Distance to the next break point =  {delta_t}")
    logger.info(f"Distance to the stationary point =  {delta_t_min}")


def _get_cauchy_point_numpy(
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    W: NDArrayFloat,
    theta: float,
    invMfactors: Tuple[NDArrayFloat, NDArrayFloat],
    use_factor: bool,
    iprint: int,
    logger: Optional[logging.Logger] = None,
):
    eps_f_sec = 1e-30
    x_cp: NDArrayFloat = x.copy()

    # To define the breakpoints in each coordinate direction, we compute
    t = np.empty_like(grad)
    t.fill(np.inf)
    # masks
    neg = grad < 0
    pos = grad > 0
    # update breakpoints
    t[neg] = (x[neg] - ub[neg]) / grad[neg]
    t[pos] = (x[pos] - lb[pos]) / grad[pos]

    # used to store the Cauchy direction `P(x-tg)-x`.
    d = np.where(t == 0, 0.0, -grad)

    # In the end, F is the list of ordered breakpoint indices
    # sort {t;,i = 1,. ..,n} in increasing order to obtain the ordered
    # set {tj :tj <= tj+1 ,j = 1, ...,n}.
    # Keep only the indices where t > 0
    # Note: sorts only positive breakpoints to reduces sort cost from O(n log n)
    # to O(k log k) where k ≪ n in practice
    pos_idx = np.flatnonzero(t > 0)
    sorted_t_idx: NDArrayInt = pos_idx[np.argsort(t[pos_idx])]

    # Initialization
    p = W.T @ d  # 2mn operations

    # Initialize c = W'(xcp - x) = 0.
    c: NDArrayFloat = np.zeros(p.size)

    # Initialize f1
    f_prime: float = -d.dot(d)  # n operations

    # Initialize derivative f2.
    f_second: float = -theta * f_prime
    f2_org: float = f_second + 0.0  # make a copy

    # Update f2 with - d^{T} @ W @ M @ W^{T} @ d = - p^{T} @ M @ p
    # old way: f2 = f2 - p.dot(M.dot(p))  # O(m^{2}) operations
    # new_way: not at first iteration -> invMfactors and M are worse zero.
    # And cho_solve produces nan so we use bmv
    if use_factor:
        f_second = f_second - p.dot(bmv(invMfactors, p))  # O(m^{2}) operations

    # dtm in the fortran code
    delta_t_min: float = -f_prime / f_second

    # Number of breakpoints
    nbreak = len(sorted_t_idx)
    # Handler the case where there are no breakpoints
    if nbreak == 0:
        # is a zero vector, return with the initial xcp as GCP.
        return x_cp, c

    # iter in the fortran code and b in [1]
    _i = 0
    # break point index (b in section 4 [1])
    ibp: int = sorted_t_idx[_i]
    # value of the smallest breakpoint, t in section 4 [1]
    t_cur: float = t[ibp]
    # previous breakpoint value
    t_old = 0.0

    delta_t: float = t_cur - 0.0

    # Number of the breakpoint segment -> Nseg in Fortran
    nseg: int = 1

    if iprint >= 99 and logger is not None:
        logger.info(f"There are {nbreak} breakpoints ")

    # flag
    is_gpc_found = False

    nbreak = len(sorted_t_idx)
    while _i < nbreak:
        display_start_point(
            nseg, f_prime, f_second, delta_t, delta_t_min, iprint, logger
        )

        if delta_t_min < delta_t:
            is_gpc_found = True
            break

        # Fix one variable and reset the corresponding component of d to zero.
        if d[ibp] > 0:
            x_cp[ibp] = ub[ibp]
        elif d[ibp] < 0:
            x_cp[ibp] = lb[ibp]
        zb = x_cp[ibp] - x[ibp]

        if iprint >= 100 and logger is not None:
            # ibp +1 to match the Fortran code (because index starts at 1)
            logger.info(f"Variable  {ibp + 1} is fixed.")

        c += delta_t * p
        W_b = W[ibp, :]
        g_b = grad[ibp]

        # Update the derivative information
        # 1) Old way
        # f1 += delta_t * f2 + g_b * (g_b + theta * zb - W_b.dot(M.dot(c)))
        # f2 -= g_b * (g_b * theta + W_b.dot(M.dot(2 * p + g_b * W_b)))
        # 2) New way with the cholesky factorization
        f_prime += delta_t * f_second + g_b * (g_b + theta * zb)
        f_second -= g_b * g_b * theta

        # First iteration -> invMfactors and M are worse zero.
        # And cho_solve produces nan
        if use_factor:
            invMWb = bmv(invMfactors, W_b)
            f_prime -= g_b * invMWb.dot(c)
            f_second -= g_b * (2.0 * invMWb.dot(p) + g_b * invMWb.dot(W_b))

        # this is a trick of the original FORTRAN code that prevents very low
        # values of f2
        f_second = max(f_second, eps_f_sec * f2_org)

        # Fix one variable and reset the corresponding component of d to zero.
        p += g_b * W_b
        d[ibp] = 0
        delta_t_min = -f_prime / f_second
        t_old = t_cur + 0.0  # copy

        _i += 1
        if _i + 1 < nbreak:
            ibp = sorted_t_idx[_i + 1]
            t_cur = t[ibp]
        else:
            # to ensure that delta_t > delta_t_min and break the while
            t_cur = np.inf

        delta_t = t_cur - t_old
        nseg += 1

    if iprint >= 99 and logger is not None:
        if is_gpc_found:
            logger.info("GCP found in this segment")
            display_start_point(
                nseg, f_prime, f_second, None, delta_t_min, iprint, logger
            )

    delta_t_min = 0 if delta_t_min < 0 else delta_t_min
    t_old += delta_t_min

    mask = t >= t_cur
    x_cp[mask] = x[mask] + t_old * d[mask]

    return x_cp, c + delta_t_min * p


@njit(cache=True)
def _get_cauchy_point_numba(
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    W: NDArrayFloat,
    theta: float,
    invMfactors: Tuple[NDArrayFloat, NDArrayFloat],
    use_factor: bool,
):
    n = x.size
    m = W.shape[1]
    eps_f_sec = 1e-30

    x_cp = x.copy()
    t = np.empty(n)
    d = np.empty(n)

    # Breakpoints
    for i in range(n):
        gi = grad[i]
        if gi < 0.0:
            t[i] = (x[i] - ub[i]) / gi
        elif gi > 0.0:
            t[i] = (x[i] - lb[i]) / gi
        else:
            t[i] = np.inf

        d[i] = -gi if t[i] != 0.0 else 0.0

    # positive breakpoints
    pos_idx = np.empty(n, dtype=np.int64)
    k = 0
    for i in range(n):
        if t[i] > 0.0:
            pos_idx[k] = i
            k += 1

    if k == 0:
        return x_cp, np.zeros(m)

    pos_idx = pos_idx[:k]
    pos_idx = pos_idx[np.argsort(t[pos_idx])]

    # initialization
    p = W.T @ d
    c = np.zeros(m)

    f_prime = -np.dot(d, d)
    f_second = -theta * f_prime
    f2_org = f_second

    if use_factor:
        tmp = bmv_numba(*invMfactors, p)
        f_second -= np.dot(p, tmp)

    delta_t_min = -f_prime / f_second

    t_old = 0.0
    ibp = pos_idx[0]
    t_cur = t[ibp]
    delta_t = t_cur
    i = 0

    while i < k:
        if delta_t_min < delta_t:
            break

        zb = ub[ibp] - x[ibp] if d[ibp] > 0 else lb[ibp] - x[ibp]
        x_cp[ibp] = x[ibp] + zb

        c += delta_t * p

        Wb = W[ibp]
        gb = grad[ibp]

        f_prime += delta_t * f_second + gb * (gb + theta * zb)
        f_second -= gb * gb * theta

        if use_factor:
            invMWb = bmv_numba(*invMfactors, Wb)
            f_prime -= gb * np.dot(invMWb, c)
            f_second -= gb * (2.0 * np.dot(invMWb, p) + gb * np.dot(invMWb, Wb))

        if f_second < eps_f_sec * f2_org:
            f_second = eps_f_sec * f2_org

        p += gb * Wb
        d[ibp] = 0.0

        delta_t_min = -f_prime / f_second
        t_old = t_cur

        i += 1
        if i < k:
            ibp = pos_idx[i]
            t_cur = t[ibp]
        else:
            t_cur = np.inf

        delta_t = t_cur - t_old

    delta_t_min = max(0.0, delta_t_min)
    t_old += delta_t_min

    for i in range(n):
        if t[i] >= t_cur:
            x_cp[i] = x[i] + t_old * d[i]

    c += delta_t_min * p
    return x_cp, c


def get_cauchy_point(
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    mats: LBFGSB_MATRICES,
    iprint: int,
    logger: Optional[logging.Logger] = None,
    is_use_numba_jit: bool = False,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    r"""
    Computes the generalized Cauchy point (GCP).

    This is the Generalized Cauchy point procedure in section 4 of [1].

    It is defined as the first local minimizer of the quadratic

    .. math::
        \[\langle grad,s\rangle + \frac{1}{2} \langle s,
        (\theta I + WMW^\intercal)s\rangle\]

    along the projected gradient direction .. math::`P_[l,u](x-\theta grad).`

    Parameters
    ----------
    x : NDArrayFloat
        Starting point for the GCP computation.
    grad : NDArrayFloat
        Gradient of fun with respect to x.
    lb : NDArrayFloat
        Lower bound vector.
    ub : NDArrayFloat
        Upper bound vector.
    mats: LBFGSB_MATRICES
        Wrapper for L-BFGS-B matrices.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99``   print details of every iteration except n-vectors;
    logger: Optional[Logger], optional
        :class:`logging.Logger` instance. If None, nothing is displayed, no matter the
        value of `iprint`, by default None.
    is_use_numba_jit: bool
        Whether to use `numba` just-in-time compilation to speed-up the computation
        intensive part of the algorithm. The default is False.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat]
        The array of Cauchy points and c = W @ (Zc - Zk).

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing, 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (2011),
      ACM Transactions on Mathematical Software, 38, 1.
    """
    # Note: the variable names follow the FORTRAN original implementation
    if iprint >= 99 and logger is not None:
        logger.info("---------------- CAUCHY entered-------------------")

    if is_use_numba_jit:
        x_cp, c = _get_cauchy_point_numba(
            x, grad, lb, ub, mats.W, mats.theta, mats.invMfactors, mats.use_factor
        )
    else:
        x_cp, c = _get_cauchy_point_numpy(
            x,
            grad,
            lb,
            ub,
            mats.W,
            mats.theta,
            mats.invMfactors,
            mats.use_factor,
            iprint,
            logger,
        )

    if logger is not None:
        if iprint > 100:
            logger.info(f"Cauchy X =  {x_cp}")
        if iprint >= 99:
            logger.info("---------------- exit CAUCHY----------------------")

    return x_cp, c
