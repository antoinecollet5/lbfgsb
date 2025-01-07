import logging
from typing import Callable, Optional, Tuple

import numpy as np
import pytest
from lbfgsb.base import get_bounds, is_any_inf
from lbfgsb.linesearch import line_search, max_allowed_steplength
from lbfgsb.scalar_function import ScalarFunction
from lbfgsb.types import NDArrayFloat


@pytest.mark.parametrize(
    "x,d,lb,ub,max_steplength,n_iter,expected",
    (
        (np.array([]), np.array([]), np.array([]), np.array([]), 0.0, 0, 1.0),
        (
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([-np.inf, -np.inf]),
            np.array([np.inf, np.inf]),
            100.0,
            10,
            100.0,
        ),
        (
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([0.5, 0.5]),
            np.array([1.5, 1.5]),
            100.0,
            10,
            0.5,
        ),
        (
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([-np.inf, np.inf]),
            np.array([1.6, np.inf]),
            100.0,
            10,
            0.6,
        ),
        (
            np.array([1.0, 1.0]),
            np.array([1.0, -1.0]),
            np.array([-np.inf, 0.3]),
            np.array([np.inf, np.inf]),
            100.0,
            10,
            0.7,
        ),
    ),
)
def test_max_allowed_steplength(
    x: NDArrayFloat,
    d: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    max_steplength: float,
    n_iter: int,
    expected: float,
) -> None:
    np.testing.assert_allclose(
        max_allowed_steplength(x, d, lb, ub, max_steplength, n_iter), expected
    )


def standalone_linesearch(
    x0: NDArrayFloat,
    fun: Callable,
    grad: Callable,
    d: NDArrayFloat,
    bounds: Optional[NDArrayFloat] = None,
    max_steplength_user: float = 1e-8,
    ftol: float = 1e-3,
    gtol: float = 0.9,
    xtol: float = 1e-1,
    max_iter: int = 30,
    iprint: int = 10,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[float], int, int, float, float, NDArrayFloat]:
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
    fun : Callable
        Objective function.
    grad : Callable
        Gradient of the objective function.
    bounds : sequence or `Bounds`, optional
        Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and
        trust-constr methods. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    d : NDArrayFloat
        Search direction.
    max_steplength : float
        Maximum steplength allowed.
    ftol: float, optional
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
    gtol: float, optional
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
    xtol: float, optional
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
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<myfprime(x_new), pk>``,
        or None if the line search algorithm did not converge.
    """
    lb, ub = get_bounds(x0, bounds)

    sf = ScalarFunction(
        fun=fun,
        x0=x0,
        args=(),
        grad=grad,
        finite_diff_bounds=(lb, ub),
        finite_diff_rel_step=None,
    )
    f0 = sf.fun(x0)
    g0 = sf.grad(x0)

    alpha = line_search(
        x0=x0,
        f0=f0,
        g0=g0,
        d=d,
        lb=lb,
        ub=ub,
        is_boxed=not is_any_inf([lb, ub]),
        sf=sf,
        above_iter=0,
        max_steplength_user=max_steplength_user,
        ftol=ftol,
        gtol=gtol,
        xtol=xtol,
        max_iter=max_iter,
        iprint=iprint,
        logger=logger,
    )
    if alpha is None:
        return (None, sf.nfev, sf.ngev, f0, f0, d)
    x_new = x0 + alpha * d
    return (alpha, sf.nfev, sf.ngev, sf.fun(x_new), f0, grad(x_new))


def obj_func(x) -> float:
    return (x[0]) ** 2 + (x[1]) ** 2


def obj_grad(x) -> NDArrayFloat:
    return np.array([2 * x[0], 2 * x[1]])


def test_standalone_linesearch() -> None:
    start_point = np.array([1.8, 1.7])
    search_gradient = np.array([-1.0, -1.0])
    bounds = np.array([[-100.0, -100.0], [100.0, 100.0]]).T  # this is optional

    print(
        standalone_linesearch(
            x0=start_point,
            fun=obj_func,
            grad=obj_grad,
            d=search_gradient,
            bounds=bounds,
        )
    )
