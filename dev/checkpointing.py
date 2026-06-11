import numpy as np
from lbfgsb import minimize_lbfgsb


def rosenbrock(x):
    x = np.asarray(x)
    sum1 = ((x[1:] - x[:-1] ** 2.0) ** 2.0).sum(axis=0)
    sum2 = np.square(1.0 - x[:-1]).sum(axis=0)
    return 100.0 * sum1 + sum2


def rosenbrock_grad(x):
    x = np.asarray(x)
    g = np.zeros(x.shape)
    # derivation of sum1
    g[1:] += 100.0 * (2.0 * x[1:] - 2.0 * x[:-1] ** 2.0)
    g[:-1] += 100.0 * (-4.0 * x[1:] * x[:-1] + 4.0 * x[:-1] ** 3.0)
    # derivation of sum2
    g[:-1] += 2.0 * (x[:-1] - 1.0)
    return g


lb = np.array([-2, -2])  # lower bounds
ub = np.array([2, 2])  # upper bounds
bounds = np.array((lb, ub)).T  # The number of variables to optimize is len(bounds)
x0 = np.array([-0.8, -1])  # The initial guess
maxfun: int = 1  # maximum number of cost function calls


res_checkpoint = minimize_lbfgsb(
    x0=x0,
    fun=rosenbrock,
    jac=rosenbrock_grad,
    bounds=bounds,
    ftol=1e-5,
    gtol=1e-5,
    maxfun=maxfun,
    is_use_numba_jit=False,
)
print(res_checkpoint, "\n")


for i in range(10):
    maxfun += 1  # we allow one more function call
    res_checkpoint = minimize_lbfgsb(
        x0=res_checkpoint.x,
        fun=rosenbrock,
        jac=rosenbrock_grad,
        bounds=bounds,
        ftol=1e-5,
        gtol=1e-5,
        maxfun=maxfun,
        checkpoint=res_checkpoint,
        is_use_numba_jit=False,
    )

    print(res_checkpoint, "\n")
