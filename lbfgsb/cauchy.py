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

from lbfgsb.bfgsmats import bmv
from lbfgsb.types import NDArrayFloat


def get_cauchy_point(
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    W: NDArrayFloat,
    invMfactors: Tuple[NDArrayFloat, NDArrayFloat],
    theta: float,
    col: int,
    max_cor: int,
    iter: int,
    iprint: int,
    logger: Optional[logging.Logger] = None,
):
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
    W : NDArrayFloat
        Part of limited memory BFGS Hessian approximation
    theta : float
        Part of limited memory BFGS Hessian approximation.
    col: int
        The actual number of variable metric corrections stored so far.
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
    Dict
        Dict containing a computed value of:
        - 'xc' the GCP
        - 'c' = W^(T)(xc-x), used for the subspace minimization
        - 'F' set of free variables

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
    x_cp = x.copy()

    # To define the breakpoints in each coordinate direction, we compute
    t = np.where(grad < 0, (x - ub) / grad, (x - lb) / grad)
    t[grad == 0] = np.inf

    # used to store the Cauchy direction `P(x-tg)-x`.
    d = np.where(t == 0, 0.0, -grad)

    # In the end, F is the list of ordered breakpoint indices
    # sort {t;,i = 1,. ..,n} in increasing order to obtain the ordered
    # set {tj :tj <= tj+1 ,j = 1, ...,n}.
    # Keep only the indices where t > 0
    F = np.argsort(t)[t > 0]

    # TODO: The integer t denotes the number of free variables at the Cauchy point zc;
    # in other words there are n - t variables at bound at zC

    # Initialization
    # There is a problem with the size of W -> it should be fixed but it is not here....
    # See what is best with python ???

    p = W.T @ d  # 2mn operations

    # Initialize c = W'(xcp - x) = 0.
    c = np.zeros(p.size)

    # Initialize f1
    f1: float = -d.dot(d)  # n operations

    # Initialize derivative f2.
    f2: float = -theta * f1
    f2_org: float = copy.deepcopy(f2)

    # Update f2 with - d^{T} @ W @ M @ W^{T} @ d = - p^{T} @ M @ p
    # old way: f2 = f2 - p.dot(M.dot(p))  # O(m^{2}) operations
    # new_way: not at first iteration -> invMfactors and M are worse zero.
    # And cho_solve produces nan
    if iter != 0:
        f2 = f2 - p.dot(bmv(invMfactors, p))  # O(m^{2}) operations

    # dtm in the fortran code
    dtm: float = -f1 / f2

    # Number of breakpoints
    nbreak = len(F)
    # Handler the case where there are no breakpoints
    if nbreak == 0:
        # is a zero vector, return with the initial xcp as GCP.
        return {
            "xc": x_cp,
            "c": c,
            "F": F,
        }

    # iter in the fortran code
    F_i = 0
    # break point index (b in section 4 [1])
    ibp = F[F_i]  # TODO: remove b from F ???
    # value of the smallest breakpoint, t in section 4 [1]
    t_min = t[ibp]
    # previous breakpoint value
    t_old = 0.0

    dt = t_min - 0.0

    # Number of the breakpoint segment -> Nseg in Fortran
    nseg: int = 1  # TODO: check that

    if iprint >= 99 and logger is not None:
        logger.info(f"There are {nbreak} breakpoints ")

    while dtm >= dt and F_i < len(F):
        if dt != 0 and iprint >= 100 and logger is not None:
            logger.info(
                f"Piece    , {nseg},  --f1, f2 at start point , {f1} , " f"{f2}"
            )
            logger.info(f"Distance to the next break point =  {dt}")
            logger.info(f"Distance to the stationary point =  {dtm}")

        # Fix one variable and reset the corresponding component of d to zero.
        if d[ibp] > 0:
            x_cp[ibp] = ub[ibp]
        elif d[ibp] < 0:
            x_cp[ibp] = lb[ibp]
        x_bcp = x_cp[ibp]
        zb = x_bcp - x[ibp]

        if iprint >= 100 and logger is not None:
            # ibp +1 to match the Fortran code (because index starts at 1)
            logger.info(f"Variable  {ibp + 1} is fixed.")
        F_i += 1

        c += dt * p
        W_b = W[ibp, :]
        g_b = grad[ibp]

        # Update the derivative information
        # 1) Old way
        # f1 += dt * f2 + g_b * (g_b + theta * zb - W_b.dot(M.dot(c)))
        # f2 -= g_b * (g_b * theta + W_b.dot(M.dot(2 * p + g_b * W_b)))
        # 2) New way with the cholesky factorization
        f1 += dt * f2 + g_b * (g_b + theta * zb)
        f2 -= g_b * g_b * theta

        # First iteration -> invMfactors and M are worse zero.
        # And cho_solve produces nan
        if iter != 0:
            f1 += g_b * W_b.dot(bmv(invMfactors, c))
            f2 += g_b * W_b.dot(bmv(invMfactors, (2 * p + g_b * W_b)))

        f2 = max(f2, eps_f_sec * f2_org)
        dtm = -f1 / f2

        # Fix one variable and reset the corresponding component of d to zero.
        p += g_b * W_b
        d[ibp] = 0
        t_old = t_min

        if F_i < len(F):
            ibp = F[F_i]
            t_min = t[ibp]
            dt = t_min - t_old
        else:
            t_min = np.inf

        nseg += 1

    if iprint >= 99 and logger is not None:
        logger.info("GCP found in this segment")

        # print(f"Piece    {nseg}  --f1, f2 at start point , {f1} , {f2}")
        # print(f"Distance to the stationary point = {dt}")

    dtm = 0 if dtm < 0 else dtm
    t_old += dtm

    x_cp[t >= t_min] = (x + t_old * d)[t >= t_min]

    F = [i for i in F if t[i] != t_min]

    c += dtm * p

    if logger is not None:
        if iprint > 100:
            logging.info(f"Cauchy X =  {x_cp}")
        if iprint >= 99:
            logging.info("---------------- exit CAUCHY----------------------")

    return {
        "xc": x_cp,
        "c": c,
        "F": F,
    }
