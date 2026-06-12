from typing import Tuple

import numpy as np
from lbfgsb import minimize_lbfgsb
from lbfgsb.types import NDArrayFloat  # for type hints, numpy array of floats


def rosenbrock(x: NDArrayFloat) -> float:
    """
    The Rosenbrock function.

    Parameters
    ----------
    x : array_like
    Array of of points at which the Rosenbrock function is to be computed.
    It can be of shape (m,) or (m, n), m being the number of variables per vector
    of parameters and n the number of different vectors.

    Returns
    -------
    float
        The gradient of the Rosenbrock function with size (n,).

    """
    x = np.asarray(x)
    sum1 = ((x[1:] - x[:-1] ** 2.0) ** 2.0).sum(axis=0)
    sum2 = np.square(1.0 - x[:-1]).sum(axis=0)
    return 100.0 * sum1 + sum2


def rosenbrock_grad(x: NDArrayFloat) -> NDArrayFloat:
    """
    The gradient of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
    Array of of points at which the Rosenbrock function is to be computed.
    It can be of shape (m,) or (m, n), m being the number of variables per vector
    of parameters and n the number of different vectors.

    Returns
    -------
    NDArrayFloat
        The gradient(s) of the Rosenbrock function with the same shapes as the input x.
    """
    x = np.asarray(x)
    g = np.zeros(x.shape)
    # derivation of sum1
    g[1:] += 100.0 * (2.0 * x[1:] - 2.0 * x[:-1] ** 2.0)
    g[:-1] += 100.0 * (-4.0 * x[1:] * x[:-1] + 4.0 * x[:-1] ** 3.0)
    # derivation of sum2
    g[:-1] += 2.0 * (x[:-1] - 1.0)
    return g


class FunGradWrapper:
    """Wrapper to transform fun_grad in ``fun`` and ``grad``."""

    def __init__(self, x0: NDArrayFloat) -> None:
        """Initialize the instance."""
        self._x: NDArrayFloat = x0
        self._fun, self._grad = self.fun_grad(x0)

    def _update_x_fun_grad_if_needed(self, x) -> None:
        if np.equal(self._x, x).all():
            return
        self._x = x
        # replace `self.fun_grad` by your own function
        self._fun, self._grad = self.fun_grad(x)

    def fun(self, x: NDArrayFloat) -> float:
        self._update_x_fun_grad_if_needed(x)
        return self._fun

    def grad(self, x: NDArrayFloat) -> NDArrayFloat:
        self._update_x_fun_grad_if_needed(x)
        return self._grad

    @staticmethod
    def fun_grad(x: NDArrayFloat) -> Tuple[float, NDArrayFloat]:
        return rosenbrock(x), rosenbrock_grad(x)


lb = np.array([-2, -2])  # lower bounds
ub = np.array([2, 2])  # upper bounds
bounds = np.array((lb, ub)).T  # The number of variables to optimize is len(bounds)
x0 = np.array([-0.8, -1])  # The initial guess

wrapper = FunGradWrapper(x0)

res = minimize_lbfgsb(
    x0=x0, fun=wrapper.fun, jac=wrapper.grad, bounds=bounds, ftol=1e-5, gtol=1e-5
)
print(res)


def objective(x):
    return x[0] ** 2 + x[1] ** 2


def gradient(x):
    return np.array([2 * x[0], 2 * x[1]])


x0 = np.array([10.0, 10.0])

res = minimize_lbfgsb(
    x0=x0,
    fun=objective,
    jac=gradient,
)

print(res.x)
print(res.fun)


def objective(x):
    return x[0] ** 2 + x[1] ** 2


x0 = np.array([10.0, 10.0])
bounds = [(0.0, None), (0.0, None)]

res = minimize_lbfgsb(
    x0=x0,
    fun=objective,
    jac="2-point",
    bounds=np.array(bounds),
)

np.testing.assert_allclose(res.x, [0.0, 0.0])
assert res.fun == 0.0


def objective_and_gradient(x):
    f = x[0] ** 2 + x[1] ** 2
    g = np.array([2 * x[0], 2 * x[1]])
    return f, g


x0 = np.array([10.0, 10.0])

res = minimize_lbfgsb(
    x0=x0,
    fun=objective_and_gradient,
    jac=True,
)

np.testing.assert_allclose(res.x, [0.0, 0.0])
assert res.fun == 0.0
