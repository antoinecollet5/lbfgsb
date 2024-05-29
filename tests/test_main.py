import numpy as np
from lbfgsb import minimize_lbfgsb
from lbfgsb.types import NDArrayFloat
from scipy.optimize import minimize


# Definition of some test functions
def quad(x: NDArrayFloat) -> float:
    return x.dot(x)


def grad_quad(x: NDArrayFloat) -> NDArrayFloat:
    return x


def test_minimize_quad() -> None:
    # 1) parameters definition
    ftol = 1e-5
    gtol = 1e-5
    lb = np.array([1, 1])
    ub = np.array([np.inf, np.inf])
    bounds = np.array((lb, ub)).T
    x0 = np.array([5, 5])

    # 2) optimizaiton with our implementation
    opt_quad = minimize_lbfgsb(
        x0=x0, fun=quad, jac=grad_quad, bounds=bounds, ftol=ftol, gtol=gtol
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
    g[1] = 200 * (-x[0] ** 2 + x[1])
    return g


def test_minimize_rozenbrock() -> None:
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
        iprint=1,
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
    # 4) optimizaiton with scipy implementation
    x0 = np.array([2.5, -1.3])
    res = minimize_lbfgsb(x0=x0, fun=quad, jac=grad_quad, ftarget=1000)

    assert res.nfev == 1
    assert res.njev == 0
    assert res.nit == 0
    assert res.success is True
    assert res.status == 0
    assert res.message == "CONVERGENCE: F_<=_TARGET"
