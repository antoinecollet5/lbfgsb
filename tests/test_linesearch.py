# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

import logging
from typing import Callable, Optional, Tuple

import numpy as np
import pytest
from lbfgsb.backend import get_backend
from lbfgsb.base import get_bounds, is_any_inf
from lbfgsb.dcsrch import DcsrchState, dcsrch
from lbfgsb.linesearch import line_search, max_allowed_steplength
from lbfgsb.scalar_function import ScalarFunction
from lbfgsb.types import NDArrayFloat

# ---------------------------------------------------------------------------
# max_allowed_steplength
# ---------------------------------------------------------------------------


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
    nx = get_backend(x) if x.size > 0 else get_backend(np.array([0.0]))
    np.testing.assert_allclose(
        max_allowed_steplength(nx, x, d, lb, ub, max_steplength, n_iter), expected
    )


# ---------------------------------------------------------------------------
# Pure-Python dcsrch smoke tests
# ---------------------------------------------------------------------------


def test_dcsrch_init() -> None:
    """First call should return FG (request function/gradient evaluation)."""
    state = DcsrchState()
    stp, f, g, task = dcsrch(0.5, 1.0, -1.0, 1e-3, 0.9, 0.1, 0.0, 10.0, state)
    assert task == "FG"
    assert 0.0 < stp <= 10.0


def test_dcsrch_error_bad_gradient() -> None:
    """Non-negative initial gradient must return ERROR."""
    state = DcsrchState()
    _, _, _, task = dcsrch(0.5, 1.0, 0.5, 1e-3, 0.9, 0.1, 0.0, 10.0, state)
    assert task == "ERROR"


def test_dcsrch_convergence_quadratic() -> None:
    """Iterate dcsrch on a simple quadratic until convergence."""

    # f(x) = (x-3)^2,  g(x) = 2*(x-3),  start at x=0, direction d=1
    # phi(stp) = (stp - 3)^2,  dphi(stp) = 2*(stp-3)
    def phi(s):
        return (s - 3.0) ** 2

    def dphi(s):
        return 2.0 * (s - 3.0)

    state = DcsrchState()
    stp = 1.0
    f, g = phi(0.0), dphi(0.0)

    for _ in range(50):
        stp, f, g, task = dcsrch(stp, f, g, 1e-4, 0.9, 1e-6, 0.0, 10.0, state)
        if task != "FG":
            break
        f = phi(stp)
        g = dphi(stp)

    assert task in ("CONV", "WARN"), f"Unexpected task: {task}"
    # Strong Wolfe only guarantees sufficient decrease + curvature, not the
    # exact minimiser. Verify the accepted step actually decreases phi.
    assert phi(stp) < phi(0.0), f"phi({stp})={phi(stp)} >= phi(0)={phi(0.0)}"
    assert stp > 0.0


# ---------------------------------------------------------------------------
# line_search integration helper
# ---------------------------------------------------------------------------


def standalone_linesearch(
    x0: NDArrayFloat,
    fun: Callable,
    grad: Callable,
    d: NDArrayFloat,
    bounds: Optional[NDArrayFloat] = None,
    max_steplength_user: float = 1e8,
    ftol: float = 1e-3,
    gtol: float = 0.9,
    xtol: float = 1e-1,
    max_iter: int = 30,
    iprint: int = -1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[float], int, int, float, float, NDArrayFloat]:
    lb, ub = get_bounds(x0, bounds)
    nx = get_backend(x0)

    sf = ScalarFunction(
        fun=fun,
        x0=x0,
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
        nx=nx,
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
    return float(x[0] ** 2 + x[1] ** 2)


def obj_grad(x) -> NDArrayFloat:
    return np.array([2.0 * x[0], 2.0 * x[1]])


def test_standalone_linesearch() -> None:
    start_point = np.array([1.8, 1.7])
    search_gradient = np.array([-1.0, -1.0])
    bounds = np.array([[-100.0, -100.0], [100.0, 100.0]]).T

    alpha, nfev, ngev, f_new, f_old, g_new = standalone_linesearch(
        x0=start_point,
        fun=obj_func,
        grad=obj_grad,
        d=search_gradient,
        bounds=bounds,
    )
    assert alpha is not None, "Line search should succeed"
    assert f_new < f_old, "Line search should decrease function value"


def test_standalone_linesearch_unbounded() -> None:
    """Line search without bounds."""
    start_point = np.array([5.0, 5.0])
    d = np.array([-1.0, -1.0])

    alpha, _, _, f_new, f_old, _ = standalone_linesearch(
        x0=start_point,
        fun=obj_func,
        grad=obj_grad,
        d=d,
    )
    assert alpha is not None
    assert f_new < f_old
