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

import copy
import logging
from typing import Optional, Tuple

import numpy as np

from lbfgsb.bfgsmats import LBFGSB_MATRICES, bmv
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
    logger.info(
        f"Piece    , {nseg},  --f1, f2 at start point , {f_prime} , " f"{f_second}"
    )
    if delta_t is not None:
        logger.info(f"Distance to the next break point =  {delta_t}")
    logger.info(f"Distance to the stationary point =  {delta_t_min}")


def get_cauchy_point(
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    mats: LBFGSB_MATRICES,
    iter: int,
    iprint: int,
    logger: Optional[logging.Logger] = None,
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
        TODO.
    iter: int
        Current iteration.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99``   print details of every iteration except n-vectors;
    logger: Optional[Logger], optional
        :class:`logging.Logger` instance. If None, nothing is displayed, no matter the
        value of `iprint`, by default None.

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

    eps_f_sec = 1e-30
    x_cp: NDArrayFloat = x.copy()

    # To define the breakpoints in each coordinate direction, we compute
    t: NDArrayFloat = np.zeros_like(grad)
    mask = grad != 0
    t[mask] = np.where(
        grad[mask] < 0, (x - ub)[mask] / grad[mask], (x - lb)[mask] / grad[mask]
    )
    t[grad == 0] = np.inf

    # used to store the Cauchy direction `P(x-tg)-x`.
    d = np.where(t == 0, 0.0, -grad)

    # In the end, F is the list of ordered breakpoint indices
    # sort {t;,i = 1,. ..,n} in increasing order to obtain the ordered
    # set {tj :tj <= tj+1 ,j = 1, ...,n}.
    # Keep only the indices where t > 0
    sorted_t_idx: NDArrayInt = np.argsort(t)[t > 0]

    # Initialization
    p = mats.W.T @ d  # 2mn operations

    # Initialize c = W'(xcp - x) = 0.
    c: NDArrayFloat = np.zeros(p.size)

    # Initialize f1
    f_prime: float = -d.dot(d)  # n operations

    # Initialize derivative f2.
    f_second: float = -mats.theta * f_prime
    f2_org: float = copy.deepcopy(f_second)

    # Update f2 with - d^{T} @ W @ M @ W^{T} @ d = - p^{T} @ M @ p
    # old way: f2 = f2 - p.dot(M.dot(p))  # O(m^{2}) operations
    # new_way: not at first iteration -> invMfactors and M are worse zero.
    # And cho_solve produces nan so we use bmv
    if mats.use_factor:
        f_second = f_second - p.dot(bmv(mats.invMfactors, p))  # O(m^{2}) operations

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

    while _i < len(sorted_t_idx):
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
        W_b = mats.W[ibp, :]
        g_b = grad[ibp]

        # Update the derivative information
        # 1) Old way
        # f1 += delta_t * f2 + g_b * (g_b + theta * zb - W_b.dot(M.dot(c)))
        # f2 -= g_b * (g_b * theta + W_b.dot(M.dot(2 * p + g_b * W_b)))
        # 2) New way with the cholesky factorization
        f_prime += delta_t * f_second + g_b * (g_b + mats.theta * zb)
        f_second -= g_b * g_b * mats.theta

        # First iteration -> invMfactors and M are worse zero.
        # And cho_solve produces nan
        if mats.use_factor:
            f_prime -= g_b * W_b.dot(bmv(mats.invMfactors, c))
            f_second -= g_b * W_b.dot(bmv(mats.invMfactors, (2 * p + g_b * W_b)))

        # this is a trick of the original FORTRAN code that prevents very low
        # values of f2
        f_second = max(f_second, eps_f_sec * f2_org)

        # Fix one variable and reset the corresponding component of d to zero.
        p += g_b * W_b
        d[ibp] = 0
        delta_t_min = -f_prime / f_second
        t_old = copy.copy(t_cur)

        _i += 1
        try:
            ibp = sorted_t_idx[_i]
            t_cur = t[ibp]
        except IndexError:
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

    x_cp[t >= t_cur] = (x + t_old * d)[t >= t_cur]

    c += delta_t_min * p

    if logger is not None:
        if iprint > 100:
            logger.info(f"Cauchy X =  {x_cp}")
        if iprint >= 99:
            logger.info("---------------- exit CAUCHY----------------------")

    return x_cp, c
