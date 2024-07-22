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

import logging
import warnings
from typing import Optional

import numpy as np
import scipy as sp
from packaging.version import Version
from scipy import __version__ as spversion

from lbfgsb.scalar_function import ScalarFunction
from lbfgsb.types import NDArrayFloat


def max_allowed_steplength(
    x: NDArrayFloat,
    d: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    max_steplength: float,
    n_iter: int,
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
    n_iter: int
        Current number of outer itreations.

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
    if n_iter == 0:
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
    is_boxed: bool,
    sf: ScalarFunction,
    ftol: float = 1e-3,
    gtol: float = 0.9,
    xtol: float = 1e-1,
    max_iter: int = 30,
    iprint: int = 10,
    logger: Optional[logging.Logger] = None,
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
    When using scipy-1.11 and below, this subroutine calls subroutine dcsrch from the
    Minpack2 library to perform the line search.  Subroutine dscrch is safeguarded so
    that all trial points lie within the feasible region. Otherwise, it uses the
    python reimplementation introduced in scipy-1.12.

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
    is_boxed: bool
        Whether all values have both lower and upper bounds.
    sf: ScalarFunction
        Wrapper for the objective function and its gradient.
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

    dphi0 = g0.dot(d)

    if above_iter == 0 and not is_boxed:
        steplength_0 = min(1.0 / np.sqrt(d.dot(d)), max_steplength)
    else:
        steplength_0 = 1.0

    # Support for python 3.7 and 3.8: the minpack2 wrapper has been removed from
    # scipy from version 1.12 and replaced with a python implementation.
    # Unfortunately, python 3.7 and 3.8 do not support scipy-1.12
    # So we need to use the old minpack2 Fortran implementation
    is_use_minpack2: bool = Version(spversion) < Version("1.12")

    if is_use_minpack2:  # scipy older than 1.12, uses the Fortran implementation
        task = b"START"
        f_m1 = f0
        dphi_m1 = dphi0
        _iter = 0
        while _iter < max_iter:
            with warnings.catch_warnings():
                # optimize.minpack2 might be deprecated but we handle this deprecation
                # for python above 3.8 so no need to raise a warning.
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                steplength, f0, dphi0, task = sp.optimize.minpack2.dcsrch(
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
                f_m1, dphi_m1 = sf.fun_and_grad(x0 + steplength * d)
                dphi_m1 = dphi_m1.dot(d)
            else:
                break
            _iter += 1
        else:
            # max_iter reached, the line search did not converge
            steplength = None

    else:  # newer version, with a pure python implementation

        def phi(alpha: float) -> float:
            """Return the objective function for a steplength of `alpha`"""
            return sf.fun(x0 + alpha * d)

        def dphi(alpha: float) -> NDArrayFloat:
            """Return the gradient of `phi` with respect to alpha."""
            return sf.grad(x0 + alpha * d).dot(d)

        dcsrch = sp.optimize._dcsrch.DCSRCH(
            phi, dphi, ftol, gtol, xtol, 0.0, max_steplength
        )
        steplength, f0, _, task = dcsrch(
            steplength_0, phi0=f0, derphi0=dphi0, maxiter=max_iter
        )

    if task[:5] == b"ERROR" or task[:4] == b"WARN":
        if task[:21] != b"WARNING: STP = STPMAX":
            warnings.warn(task.decode("utf-8"))
            steplength = None  # failed

    if iprint >= 99 and logger is not None and steplength is not None:
        logger.info(
            f"LINE SEARCH  {iter} times; norm of step = "
            f"{steplength * np.linalg.norm(d)}"
        )

    return steplength
