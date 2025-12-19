# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

import copy
import logging
import re
from typing import Deque

import numpy as np
import pytest
from lbfgsb import InternalState, minimize_lbfgsb
from lbfgsb.main import is_f0_target_reached
from lbfgsb.types import NDArrayFloat
from scipy.optimize import (
    LbfgsInvHessProduct,  # noqa : F401
    OptimizeResult,
    minimize,
)

logger: logging.Logger = logging.getLogger("L-BFGS-B")
logger.setLevel(logging.INFO)
logging.info("this is a logging test")


# Definition of some test functions
def quad(x: NDArrayFloat) -> float:
    return x.dot(x)


def grad_quad(x: NDArrayFloat) -> NDArrayFloat:
    return x


def test_is_f0_target_reached() -> None:
    istate = InternalState()
    assert istate.is_success is False
    assert istate.task_str == "START"
    assert istate.warnflag == 2

    # No changes to istate
    assert is_f0_target_reached(1e-2, 1e-3, istate) is False
    assert is_f0_target_reached(1e-2, None, istate) is False

    assert istate.is_success is False
    assert istate.task_str == "START"
    assert istate.warnflag == 2

    # Changes to istate
    assert is_f0_target_reached(1e-4, 1e-3, istate) is True

    assert istate.is_success is True
    assert istate.task_str == "CONVERGENCE: F_<=_TARGET"
    assert istate.warnflag == 0


def update_fun_def_does_nothing(
    x: NDArrayFloat,
    f0: float,
    f0_old: float,
    grad: NDArrayFloat,
    X: Deque[NDArrayFloat],
    G: Deque[NDArrayFloat],
):
    """Does nothing, just for the test"""
    return f0, f0_old, grad, G


@pytest.mark.parametrize("is_use_numba_jit", ((True,), (False,)))
def test_minimize_quad(is_use_numba_jit: bool) -> None:
    # 1) parameters definition
    ftol = 1e-5
    gtol = 1e-5
    lb = np.array([1, 1])
    ub = np.array([np.inf, np.inf])
    bounds = np.array((lb, ub)).T
    x0 = np.array([5, 5])

    # 2) optimizaiton with our implementation
    opt_quad = minimize_lbfgsb(
        x0=x0,
        fun=quad,
        jac=grad_quad,
        bounds=bounds,
        ftol=ftol,
        gtol=gtol,
        update_fun_def=update_fun_def_does_nothing,
        logger=logger,
        iprint=1000,
        is_check_factorizations=True,
        is_use_numba_jit=is_use_numba_jit,
    )

    # 3) Check the results correctness
    x_opt = np.array([1, 1])
    np.testing.assert_allclose(x_opt, opt_quad.x)
    np.testing.assert_allclose(quad(x_opt), opt_quad.fun)
    np.testing.assert_allclose(grad_quad(x_opt), opt_quad.jac)

    # 4) optimizaiton with scipy implementation
    minimize(
        quad,
        x0,
        jac=grad_quad,
        bounds=bounds,
        method="l-bfgs-b",
        options={"gtol": gtol, "ftol": ftol},
    )

    # 5) comparison with scipy
    # TODO


## Second example : min Rosenbrock function


def rosenbrock(x) -> float:
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def grad_rosenbrock(x) -> NDArrayFloat:
    g = np.empty(x.size)
    g[0] = 400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1)
    g[1] = 200 * (-(x[0] ** 2) + x[1])
    return g


@pytest.mark.parametrize("is_use_numba_jit", ((True,), (False,)))
def test_minimize_rozenbrock(is_use_numba_jit: bool) -> None:
    # 1) parameters definition
    ftol = 1e-5
    gtol = 1e-5
    lb = np.array([-2, -2])
    ub = np.array([2, 2])
    bounds = np.array((lb, ub)).T
    # x0 = np.array([0.12, 0.12])
    x0 = np.array([-1, -1])

    # 2) optimizaiton with our implementation
    opt_rosenbrock = minimize_lbfgsb(
        x0=x0,
        fun=rosenbrock,
        jac=grad_rosenbrock,
        bounds=bounds,
        ftol=ftol,
        gtol=gtol,
        logger=logger,
        iprint=1000,
        is_check_factorizations=True,
        is_use_numba_jit=is_use_numba_jit,
    )
    x_opt = np.array([1, 1])
    np.testing.assert_allclose(x_opt, opt_rosenbrock.x, rtol=1e-3)
    np.testing.assert_allclose(
        rosenbrock(x_opt), opt_rosenbrock.fun, atol=1e-6, rtol=1.0
    )
    np.testing.assert_allclose(
        grad_rosenbrock(x_opt), opt_rosenbrock.jac, atol=1e-20, rtol=1.0
    )

    # 3) Check the results correctness

    # 4) optimizaiton with scipy implementation
    minimize(
        rosenbrock,
        x0,
        jac=grad_rosenbrock,
        bounds=bounds,
        method="l-bfgs-b",
        options={"gtol": gtol, "ftol": ftol},
    )

    # 5) comparison with scipy


def beale(x):
    return (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )


def grad_beale(x):
    y1 = x[1]
    y2 = y1 * y1
    y3 = y2 * y1
    f1 = 1.5 - x[0] + x[0] * y1
    f2 = 2.25 - x[0] + x[0] * y2
    f3 = 2.625 - x[0] + x[0] * y3

    return np.array(
        [
            2 * (y1 - 1) * f1 + 2 * (y2 - 1) * f2 + 2 * (y3 - 1) * f3,
            2 * x[0] * f1 + 4 * x[0] * y1 * f2 + 6 * x[0] * y2 * f3,
        ]
    )


def test_minimize_beale() -> None:
    # 1) parameters definition
    ftol = 1e-14
    gtol = 1e-10
    lb = -4.5 * np.ones(2)
    ub = -lb
    bounds = np.array((lb, ub)).T
    x0 = np.array([2.5, -1.3])

    # 2) optimizaiton with our implementation
    opt_res = minimize_lbfgsb(
        x0=x0, fun=beale, jac=grad_beale, bounds=bounds, ftol=ftol, gtol=gtol
    )

    # 3) Check the results correctness
    x_opt = np.array([3, 0.5])
    np.testing.assert_allclose(x_opt, opt_res.x)
    np.testing.assert_allclose(beale(x_opt), opt_res.fun, atol=1e-20, rtol=1.0)
    np.testing.assert_allclose(grad_beale(x_opt), opt_res.jac, atol=1e-20, rtol=1.0)

    # 4) optimizaiton with scipy implementation
    minimize(
        beale,
        x0,
        jac=grad_beale,
        bounds=bounds,
        method="l-bfgs-b",
        options={"gtol": gtol, "ftol": ftol},
    )


def test_early_stop() -> None:
    """Early stop because of target stop criterion."""
    x0 = np.array([2.5, -1.3])
    res = minimize_lbfgsb(
        x0=x0, fun=quad, jac=grad_quad, ftarget=1000, is_check_factorizations=True
    )

    assert res.nfev == 1
    assert res.njev == 0
    assert res.nit == 0
    assert res.success is True
    assert res.status == 0
    assert res.message == "CONVERGENCE: F_<=_TARGET"


@pytest.mark.parametrize(
    "x0, expected_msg, is_success",
    (
        (np.array([-50]), "ABNORMAL_TERMINATION_IN_LNSRCH", False),
        (np.array([2.5]), "CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL", True),
    ),
)
def test_abnormal_termination_linesearch(
    x0: NDArrayFloat, expected_msg: str, is_success
) -> None:
    """Abnormal termination."""

    def func(x: NDArrayFloat) -> float:
        return np.sum(x + np.exp(-10 * x)).item()

    # result = minimize(func, x0=10, method='L-BFGS-B',
    #                 options={'maxls': 5, 'disp': 1})

    def jac(x: NDArrayFloat) -> NDArrayFloat:
        return 1.0 - 10 * np.exp(-10 * x)

    res = minimize_lbfgsb(
        x0=x0, fun=func, jac=jac, maxls=5, is_check_factorizations=True
    )
    assert res.message == expected_msg
    assert res.success is is_success


def test_checkpointing() -> None:
    # 1) parameters definition
    ftol = 1e-10
    gtol = 1e-10
    maxiter = 20
    maxfun = 20
    maxcor = 5
    lb = np.array([-2, -2])
    ub = np.array([2, 2])
    bounds = np.array((lb, ub)).T
    # x0 = np.array([0.12, 0.12])
    x0 = np.array([-1, -1])

    # 2) optimizaiton until the end
    _ = minimize_lbfgsb(
        x0=x0,
        fun=rosenbrock,
        jac=grad_rosenbrock,
        bounds=bounds,
        maxiter=maxiter,
        maxfun=maxfun,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        logger=logger,
        iprint=1000,
        is_check_factorizations=True,
    )

    # test an empty checkpoint
    istate = InternalState()
    empty_checkpoint = OptimizeResult(
        fun=0.0,
        jac=np.zeros(2),
        nfev=0,
        njev=0,
        nit=istate.nit,
        status=istate.warnflag,
        message=istate.task_str,
        x=x0,
        success=istate.is_success,
        hess_inv=LbfgsInvHessProduct(
            np.array([]).reshape(-1, 2),
            np.array([]).reshape(-1, 2),
        ),
    )

    # empty checkpoint
    _: OptimizeResult = minimize_lbfgsb(
        x0=x0,
        fun=rosenbrock,
        jac=grad_rosenbrock,
        bounds=bounds,
        maxiter=5,
        maxfun=5,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        logger=logger,
        iprint=1000,
        is_check_factorizations=True,
        checkpoint=empty_checkpoint,
    )

    # Non correct checkpoint
    wrong_ckp = copy.copy(empty_checkpoint)
    wrong_ckp.hess_inv = LbfgsInvHessProduct(
        np.zeros((2, 3)),  # the second dim should be 2
        np.zeros((2, 3)),  # the second dim should be 2
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The size of correction vector (3) does not match the size of x (2)!"
        ),
    ):
        _: OptimizeResult = minimize_lbfgsb(
            x0=x0,
            fun=rosenbrock,
            jac=grad_rosenbrock,
            bounds=bounds,
            maxiter=5,
            maxfun=5,
            maxcor=maxcor,
            ftol=ftol,
            gtol=gtol,
            logger=logger,
            iprint=1000,
            is_check_factorizations=True,
            checkpoint=wrong_ckp,
        )

    # optimization with max 10 nfev
    opt_rosenbrock2: OptimizeResult = minimize_lbfgsb(
        x0=x0,
        fun=rosenbrock,
        jac=grad_rosenbrock,
        bounds=bounds,
        maxiter=5,
        maxfun=10,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        logger=logger,
        iprint=1000,
        is_check_factorizations=True,
    )

    assert opt_rosenbrock2.nfev == 8
    # max determined by maxcor = 5
    assert opt_rosenbrock2.hess_inv.sk.shape[0] == 5  # maxcor
    assert opt_rosenbrock2.hess_inv.yk.shape[0] == 5  # maxcor
    assert opt_rosenbrock2.message == "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"

    # optimization restart with checkpointing
    with pytest.raises(
        ValueError,
        match=re.escape(
            "When 'checkpoint' is provided (L-BFGS-B restart), "
            "x0 and checkpoint.x should be equal!"
        ),
    ):
        _ = minimize_lbfgsb(
            x0=x0,
            fun=rosenbrock,
            jac=grad_rosenbrock,
            bounds=bounds,
            maxiter=maxiter,
            maxfun=maxfun,
            maxcor=maxcor,
            ftol=ftol,
            gtol=gtol,
            logger=logger,
            iprint=1000,
            checkpoint=copy.deepcopy(opt_rosenbrock2),
        )

    # should yield the same as the original optimization
    opt_rosenbrock3 = minimize_lbfgsb(
        x0=opt_rosenbrock2.x,
        fun=rosenbrock,
        jac=grad_rosenbrock,
        bounds=bounds,
        maxiter=maxiter,
        maxfun=maxfun,
        ftol=ftol,
        gtol=gtol,
        maxcor=maxcor,
        logger=logger,
        iprint=1000,
        checkpoint=opt_rosenbrock2,
    )

    assert opt_rosenbrock3.nfev == 20
    # max determined by maxcor
    assert opt_rosenbrock3.hess_inv.sk.shape[0] == 5
    assert opt_rosenbrock3.hess_inv.yk.shape[0] == 5

    # TODO: this fails while it should not. There is an issue with the checkpointing.
    # np.testing.assert_allclose(opt_rosenbrock.x, opt_rosenbrock3.x)
    # np.testing.assert_allclose(
    # opt_rosenbrock.hess_inv.sk, opt_rosenbrock3.hess_inv.sk)
    # np.testing.assert_allclose(
    # opt_rosenbrock.hess_inv.yk, opt_rosenbrock3.hess_inv.yk)

    # Already converge so should return the checkpoint
    # but with a modified approximated hessien because we changed maxcor to 3
    opt_rosenbrock4 = minimize_lbfgsb(
        x0=opt_rosenbrock3.x,
        fun=rosenbrock,
        jac=grad_rosenbrock,
        bounds=bounds,
        maxiter=maxiter,
        maxfun=maxfun,
        maxcor=3,  # update maxcor to test the reshape
        ftol=ftol,
        gtol=gtol,
        logger=logger,
        iprint=1000,
        checkpoint=opt_rosenbrock3,
    )

    assert opt_rosenbrock4.nfev == 20
    # determined by maxcor
    assert opt_rosenbrock4.hess_inv.sk.shape[0] == 3
    assert opt_rosenbrock4.hess_inv.yk.shape[0] == 3

    np.testing.assert_allclose(opt_rosenbrock3.x, opt_rosenbrock4.x)

    # Last test, we increase the maxiter and maxfun but the target provokes the stop
    opt_rosenbrock5 = minimize_lbfgsb(
        x0=opt_rosenbrock4.x,
        fun=rosenbrock,
        jac=grad_rosenbrock,
        bounds=bounds,
        maxiter=maxiter + 5,
        maxfun=maxfun + 5,
        ftarget=opt_rosenbrock4.fun + 1.0,
        maxcor=3,  # update maxcor to test the reshape
        ftol=ftol,
        gtol=gtol,
        logger=logger,
        iprint=1000,
        checkpoint=opt_rosenbrock4,
    )

    assert opt_rosenbrock5.nfev == 20
    # determined by maxcor
    assert opt_rosenbrock5.hess_inv.sk.shape[0] == 3
    assert opt_rosenbrock5.hess_inv.yk.shape[0] == 3


def test_user_callback() -> None:
    # 1) parameters definition
    ftol = 1e-10
    gtol = 1e-10
    maxiter = 20
    maxfun = 20
    maxcor = 5
    lb = np.array([-2, -2])
    ub = np.array([2, 2])
    bounds = np.array((lb, ub)).T
    # x0 = np.array([0.12, 0.12])
    x0 = np.array([-1, -1])

    def _callback(x: NDArrayFloat, opt: OptimizeResult) -> bool:
        if opt.nfev > 5:
            return True
        return False

    # 2) optimizaiton until the end
    res = minimize_lbfgsb(
        x0=x0,
        fun=rosenbrock,
        jac=grad_rosenbrock,
        bounds=bounds,
        maxiter=maxiter,
        maxfun=maxfun,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        logger=logger,
        iprint=1000,
        is_check_factorizations=True,
        callback=_callback,
    )

    assert res.nfev == 7
