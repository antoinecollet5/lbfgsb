"""
Benchmark with the Rosenbrock function
"""

from time import perf_counter

import matplotlib.pyplot as plt
from lbfgsb import minimize_lbfgsb
from lbfgsb.benchmarks import rosenbrock, rosenbrock_grad
from lbfgsb.mathops import np, set_backend_to_cupy, set_backend_to_defaults
from scipy.optimize import minimize

# Set up problem sizes and bounds
problem_sizes = [2**i for i in range(1, 12)]
FTOL = 1e-5
GTOL = 1e-5
MAXCOR = 5
MAXITER = 15000

times_lbfgsb = []
times_scipy = []

NTRIALS = 5

set_backend_to_cupy()

for PROBLEM_SIZE in problem_sizes:
    # Set up problem bounds and initial guess
    lb = np.full(PROBLEM_SIZE, -2.0)
    ub = np.full(PROBLEM_SIZE, 2.0)
    bounds = np.array((lb, ub)).T
    x0 = np.random.random(PROBLEM_SIZE)

    times_lbfgsb_trial = []
    times_scipy_trial = []
    errors_lbfgsb = []
    errors_scipy = []

    for _ in range(NTRIALS):
        t1 = perf_counter()
        opt_rosenbrock = minimize_lbfgsb(
            x0=x0,
            fun=rosenbrock,
            jac=rosenbrock_grad,
            bounds=bounds,
            maxcor=MAXCOR,
            ftol=FTOL,
            gtol=GTOL,
            iprint=0,
            maxiter=MAXITER,
        )
        t2 = perf_counter()
        times_lbfgsb_trial.append(t2 - t1)

        t1 = perf_counter()
        opt_rosenbrock_scipy = minimize(
            rosenbrock,
            x0=x0,
            jac=rosenbrock_grad,
            bounds=bounds,
            method="L-BFGS-B",
            options={
                "maxiter": MAXITER,
                "ftol": FTOL,
                "gtol": GTOL,
                "iprint": 0,
                "maxcor": MAXCOR,
            },
        )
        t2 = perf_counter()
        times_scipy_trial.append(t2 - t1)

    times_lbfgsb.append(np.mean(times_lbfgsb_trial))
    times_scipy.append(np.mean(times_scipy_trial))
    errors_lbfgsb.append(np.std(times_lbfgsb_trial))
    errors_scipy.append(np.std(times_scipy_trial))

plt.figure()
plt.errorbar(
    problem_sizes,
    times_lbfgsb,
    yerr=errors_lbfgsb,
    fmt="-o",
    label="lbfgsb",
    capsize=5,
    ecolor="black",
)
plt.errorbar(
    problem_sizes,
    times_scipy,
    yerr=errors_scipy,
    fmt="-s",
    label="scipy",
    capsize=5,
    ecolor="black",
)
plt.xlabel("Problem size")
plt.ylabel("Time (s) - mean ± std")
plt.legend()
plt.title("Benchmark: L-BFGS-B (cupy) vs scipy")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.show()
