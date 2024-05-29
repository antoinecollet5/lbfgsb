r"""
Implement the line search algorithm by Moré and Thuente (1994),
currently used for the L-BFGS-B algorithm.

The target of this line search algorithm is to find a step size \f$\alpha\f$ that
satisfies the strong Wolfe condition
\f$f(x+\alpha d) \le f(x) + \alpha\mu g(x)^T d\f$ and \f$|g(x+\alpha d)^T d| \le
\eta|g(x)^T d|\f$.

Functions
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    max_allowed_steplength
    line_search

Reference:
[1] Moré, J. J., & Thuente, D. J. (1994). Line search algorithms with guaranteed
sufficient decrease.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import scipy as sp

from lbfgsb.types import NDArrayFloat


def max_allowed_steplength(
    x: NDArrayFloat,
    d: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    max_steplength: float,
    iter: int,
) -> float:
    r"""
    Computes the biggest 0<=k<=max_steplength such that:
        l<= x+kd <= u

    Parameters
    ----------
    x : NDArrayFloat
        Starting point.
    d : NDArrayFloat
        Direction.
    lb : NDArrayFloat
        the lower bound of x.
    ub : NDArrayFloat
        The upper bound of x
    max_steplength : float
        Maximum steplength allowed.

    Returns
    -------
    float
        maximum steplength allowed

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
    # Determine the maximum step length.
    if iter == 0:
        return 1.0  # we are not sure this is a good idea
    with np.errstate(divide="ignore"):
        _mask = d != 0
        _tmp = np.where(
            d[_mask] > 0, (ub - x)[_mask] / d[_mask], (lb - x)[_mask] / d[_mask]
        )
        if _tmp[np.isfinite(_tmp)].size == 0:
            return max_steplength
        return min(max_steplength, np.nanmin(_tmp[np.isfinite(_tmp)]))


def line_search(
    x0: NDArrayFloat,
    f0: float,
    g0: NDArrayFloat,
    d: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    above_iter: int,
    max_steplength_user: float,
    fun_and_grad: Callable[[NDArrayFloat], Tuple[float, NDArrayFloat]],
    ftol: float = 1e-3,
    gtol: float = 0.9,
    xtol: float = 1e-1,
    max_iter: int = 30,
    iprint: int = 10,
    isave: NDArrayFloat = np.zeros((2,), np.intc),
    dsave: NDArrayFloat = np.zeros((13,), np.float64),
) -> Optional[float]:
    r"""
    Find a step that satisfies both decrease condition and a curvature condition.

        f(x0+stp*d) <= f(x0) + alpha*stp*\langle f'(x0),d\rangle,

    and the curvature condition

        abs(f'(x0+stp*d)) <= beta*abs(\langle f'(x0),d\rangle).

    If alpha is less than beta and if, for example, the functionis bounded below, then
    there is always a step which satisfies both conditions.

    Note
    ----
    This subroutine calls subroutine dcsrch from the Minpack2 library
    to perform the line search.  Subroutine dscrch is safeguarded so
    that all trial points lie within the feasible region.

    Parameters
    ----------
    x0 : NDArrayFloat
        Starting point.
    f0 : float
        Objective function value for x0.
    g0 : NDArrayFloat
        Gradient of the objective function for x0.
    lb : NDArrayFloat
        Lower bound vector.
    ub : NDArrayFloat
        Upper bound vector.
    d : NDArrayFloat
        Search direction.
    above_iter : int
        current iteration in optimization process.
    max_steplength : float
        Maximum steplength allowed.
    fun_and_grad : Callable[[NDArrayFloat], Tuple[float, NDArrayFloat]]
        Function returning both the objective function and its gradient with respect to
        a given vector x.
    ftol_linesearch: float, optional
        Specify a nonnegative tolerance for the sufficient decrease condition in
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_
        (used for the line search). This is :math:`c_1` in
        the Armijo condition (or Goldstein, Goldstein-Armijo condition) where
        :math:`\alpha_{k}` is the estimated step.

        .. math::

            f(\mathbf{x}_{k}+\alpha_{k}\mathbf{p}_{k})\leq
            f(\mathbf{x}_{k})+c_{1}\alpha_{k}\mathbf{p}_{k}^{\mathrm{T}}
            \nabla f(\mathbf{x}_{k})

        Note that :math:`0 < c_1 < 1`. Usually :math:`c_1` is small, see the Wolfe
        conditions in :cite:t:`nocedalNumericalOptimization1999`.
        In the fortran implementation
        algo 778, it is hardcoded to 1e-3. The default is 1e-4.
    gtol_linesearch: float, optional
        Specify a nonnegative tolerance for the curvature condition in
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_
        (used for the line search). This is :math:`c_2` in
        the Armijo condition (or Goldstein, Goldstein-Armijo condition) where
        :math:`\alpha_{k}` is the estimated step.

        .. math::

            \left|\mathbf{p}_{k}^{\mathrm {T}}\nabla f(\mathbf{x}_{k}+\alpha_{k}
            \mathbf{p}_{k})\right|\leq c_{2}\left|\mathbf {p}_{k}^{\mathrm{T}}\nabla
            f(\mathbf{x}_{k})\right|

        Note that :math:`0 < c_1 < c_2 < 1`. Usually, :math:`c_2` is
        much larger than :math:`c_2`.
        see :cite:t:`nocedalNumericalOptimization1999`. In the fortran implementation
        algo 778, it is hardcoded to 0.9. The default is 0.9.
    xtol_linesearch: float, optional
        Specify a nonnegative relative tolerance for an acceptable step in the line
        search procedure (see
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_).
        In the fortran implementation algo 778, it is hardcoded to 0.1.
        The default is 1e-5.
    max_iter : int, optional
        Maximum number of linesearch iterations, by default 30.

    Returns
    -------
    Optional[float]
        The step length.

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

    # steplength_0 = 1 if max_steplength > 1 else 0.5 * max_steplength
    max_steplength = max_allowed_steplength(
        x0, d, lb, ub, max_steplength_user, above_iter
    )

    f_m1 = f0
    dphi = g0.dot(d)
    dphi_m1 = dphi
    iter = 0

    if above_iter == 0:
        steplength_0 = min(1.0 / np.sqrt(d.dot(d)), max_steplength)
    else:
        steplength_0 = 1.0

    # print(f"max_steplength = {max_steplength}")
    # print(f"steplength_0 = {steplength_0}")

    task = b"START"

    while iter < max_iter:
        steplength, f0, dphi, task = sp.optimize.minpack2.dcsrch(
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
        if task[:2] == b"FG":
            steplength_0 = steplength
            f_m1, dphi_m1 = fun_and_grad(x0 + steplength * d)
            dphi_m1 = dphi_m1.dot(d)
        else:
            break
        iter += 1
    else:
        # max_iter reached, the line search did not converge
        steplength = None

    if task[:5] == b"ERROR" or task[:4] == b"WARN":
        if task[:21] != b"WARNING: STP = STPMAX":
            print(task)
            steplength = None  # failed

    if iprint >= 99:
        print(f"LINE SEARCH  {iter} times; norm of step = {steplength}")

    return steplength
