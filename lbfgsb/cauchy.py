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
from typing import Tuple

import numpy as np

from lbfgsb.bfgsmats import bmv
from lbfgsb.types import NDArrayFloat


def get_cauchy_point(
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    W: NDArrayFloat,
    M: NDArrayFloat,
    invMfactors: Tuple[NDArrayFloat, NDArrayFloat],
    theta: float,
    col: int,
    max_cor: int,
    iprint: int,
    iter: int,
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
    M : NDArrayFloat
        Part of limited memory BFGS Hessian approximation
    theta : float
        Part of limited memory BFGS Hessian approximation.
    col: int
        The actual number of variable metric corrections stored so far.
    iprint: int
        Printing level.
    iter: int
        Current iteration.

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
    if iprint >= 99:
        print("---------------- CAUCHY entered-------------------")

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
    c = np.zeros(p.size)
    # f1 in the original code
    f_prime: float = -d.dot(d)  # n operations
    # f2 in the original code
    f_second: float = -theta * f_prime
    # f2_org in the fortran code
    f_sec0: float = copy.deepcopy(f_second)
    # Update f2 with - d^{T} @ W @ M @ W^{T} @ d = - p^{T} @ M @ p
    # old way: f_second = f_second - p.dot(M.dot(p))  # O(m^{2}) operations
    # new_way: not at first iteration -> invMfactors and M are worse zero.
    # And cho_solve produces nan
    if iter != 0:
        f_second = f_second - p.dot(bmv(invMfactors, p))  # O(m^{2}) operations

    # dtm in the fortran code
    Dt_min: float = -f_prime / f_second

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
    t_old = 0

    Dt = t_min - 0

    # Number of the breakpoint segment -> Nseg in Fortran
    nseg: int = 1  # TODO: check that

    if iprint >= 99:
        print(f"There are {nbreak} breakpoints ")

    if nbreak != 0:
        pass

        while Dt_min >= Dt and F_i < len(F):
            if Dt != 0 and iprint >= 100:
                print(
                    f"Piece    , {nseg},  --f1, f2 at start point , {f_prime} , "
                    f"{f_second}"
                )
                print(f"Distance to the next break point =  {Dt}")
                print(f"Distance to the stationary point =  {Dt_min}")

            # Fix one variable and reset the corresponding component of d to zero.
            if d[ibp] > 0:
                x_cp[ibp] = ub[ibp]
            elif d[ibp] < 0:
                x_cp[ibp] = lb[ibp]
            x_bcp = x_cp[ibp]
            zb = x_bcp - x[ibp]

            if iprint >= 100:
                # ibp +1 to match the Fortran code (because index starts at 1)
                print(f"Variable  {ibp + 1} is fixed.")
            F_i += 1

            c += Dt * p
            W_b = W[ibp, :]
            g_b = grad[ibp]

            # Update the derivative information
            # 1) Old way
            # f_prime += Dt * f_second + g_b * (g_b + theta * zb - W_b.dot(M.dot(c)))
            # f_second -= g_b * (g_b * theta + W_b.dot(M.dot(2 * p + g_b * W_b)))
            # 2) New way with the cholesky factorization
            f_prime += Dt * f_second + g_b * (g_b + theta * zb)
            f_second -= g_b * (g_b * theta)
            # First iteration -> invMfactors and M are worse zero.
            # And cho_solve produces nan
            if iter != 0:
                f_prime -= g_b * W_b.dot(bmv(invMfactors, c))
                f_second -= g_b * W_b.dot(bmv(invMfactors, (2 * p + g_b * W_b)))

            f_second = min(f_second, eps_f_sec * f_sec0)

            Dt_min = -f_prime / f_second

            # Fix one variable and reset the corresponding component of d to zero.
            p += g_b * W_b
            d[ibp] = 0
            t_old = t_min

            if F_i < len(F):
                ibp = F[F_i]
                t_min = t[ibp]
                Dt = t_min - t_old
            else:
                t_min = np.inf

            nseg += 1

    if iprint >= 99:
        print("GCP found in this segment")

        # print(f"Piece    {nseg}  --f1, f2 at start point , {f_prime} , {f_second}")
        # print(f"Distance to the stationary point = {Dt}")

    Dt_min = 0 if Dt_min < 0 else Dt_min
    t_old += Dt_min

    x_cp[t >= t_min] = (x + t_old * d)[t >= t_min]

    F = [i for i in F if t[i] != t_min]

    c += Dt_min * p

    if iprint > 100:
        print(f"Cauchy X =  {x_cp}")
    if iprint >= 99:
        print("---------------- exit CAUCHY----------------------")

    return {
        "xc": x_cp,
        "c": c,
        "F": F,
    }
