"""
Benchmark with multiple benchmark functions.

This benchmark evaluates the performance of L-BFGS-B with CuPy backend
compared to NumPy backend and SciPy's L-BFGS-B implementation.

The benchmark includes error bar plotting to show standard deviation across
multiple trials, and supports configurable problem sizes and number of trials.

AI Disclosure
    This benchmark used qwen3.5, an open-weights AI model, to generate the plotting
    component of this code, linting, and clean up the docstrings.
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as tnp
from lbfgsb import minimize_lbfgsb
from lbfgsb.mathops import np, set_backend_to_cupy, set_backend_to_defaults
from scipy.optimize import minimize

# =======================================================================
# CONFIGURATION
# =======================================================================

# Number of trials for each benchmark run (use 5 for stable results)
NTRIALS = 1

# Problem sizes (dimensions) to test
# For CuPy: use powers of 2 for optimal GPU memory alignment
# For NumPy: use various sizes to capture performance characteristics
problem_sizes = [2**i for i in range(16, 21)]

# Optimization tolerances
FTOL = 1e-5  # Function tolerance
GTOL = 1e-5  # Gradient tolerance
MAXCOR = 10  # Maximum correlation
MAXITER = 20_000  # Maximum iterations

# Number of function evaluations limit
MAXFUN = 20_000

# =======================================================================
# HELPER FUNCTIONS
# =======================================================================


def get_bounds(problem_size):
    """Added by qwen3.5
    Generate uniform bounds for the benchmark problem using CuPy.

    Parameters
    ----------
    problem_size : int
        Number of dimensions.

    Returns
    -------
    tuple
        (lower_bounds, upper_bounds) CuPy arrays.
    """
    lb = np.full(problem_size, -5.0)
    ub = np.full(problem_size, 5.0)
    return lb, ub


def get_function_label(label):
    """Added by qwen3.5
    Get a formatted label for the plot.

    Parameters
    ----------
    label : str
        Function name.

    Returns
    -------
    str
        Formatted label string.
    """
    return f"{label.capitalize()} function"  # noqa: E702


# =======================================================================
# MAIN BENCHMARK EXECUTION
# =======================================================================


def run_function_benchmark(
    problem_size,
    fun,
    jac,
    bounds,
    actual_backend,
):
    """
    Run benchmark for a single function.

    Parameters
    ----------
    problem_size : int
        Number of dimensions.
    fun : callable
        Objective function.
    jac : callable
        Gradient function.
    bounds : numpy.ndarray
        Bounds array.
    actual_backend : str
        'numpy', 'cupy', 'scipy'.

    Returns
    -------
    tuple
        (lbfgsb_times, scipy_times, lbfgsb_errors, scipy_errors)
    """
    times_lbfgsb_trial = []
    times_scipy_trial = []

    assert actual_backend in ["numpy", "cupy", "scipy"], (
        f"Invalid backend: {actual_backend}"
    )

    # Whether to grab the raw CuPy/CUDA arrays or convert to NumPy
    if actual_backend == "cupy":
        test_fun = fun
        test_jac = jac
        x0 = np.random.random(problem_size) * 10 - 5
        x0 = np.clip(x0, -5.0, 5.0).reshape(-1)
    else:
        test_fun = lambda x, f=fun: f(x).get() if hasattr(f(x), "get") else f(x)
        test_jac = lambda x, g=jac: g(x).get() if hasattr(g(x), "get") else g(x)
        x0 = tnp.random.random(problem_size) * 10 - 5
        x0 = tnp.clip(x0, -5.0, 5.0).reshape(-1)

    for trial in range(NTRIALS):
        # Reset initial guess within bounds [-5, 5]

        # Run L-BFGS-B
        print(f"    Trial {trial + 1}/{NTRIALS}... for backend {actual_backend}")
        t1 = perf_counter()
        if actual_backend != "scipy":
            opt_result = minimize_lbfgsb(
                x0=x0,
                fun=test_fun,
                jac=test_jac,
                bounds=bounds,
                maxcor=MAXCOR,
                ftol=FTOL,
                gtol=GTOL,
                iprint=0,
                maxiter=MAXITER,
                maxfun=MAXFUN,
            )

        else:
            opt_result = minimize(
                x0=x0,
                fun=test_fun,
                jac=test_jac,
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
        times_lbfgsb_trial.append(t2 - t1)
        print(f"    L-BFGS-B time: {times_lbfgsb_trial[-1]:.4f}s")
        print(f"    Success: {opt_result.success}")

        # Run SciPy's L-BFGS-B for comparison
        print(f"    Running SciPy L-BFGS-B optimization...")

    # Calculate mean and std
    lbfgsb_times = tnp.mean(times_lbfgsb_trial)
    lbfgsb_errors = tnp.std(times_lbfgsb_trial)

    return lbfgsb_times, lbfgsb_errors


def benchmark_lbfgsb(fun, jac):
    """
    Run the benchmark for a specific backend.

    Parameters
    ----------
    fun : callable
        Objective function to minimize.
    jac : callable
        Gradient of the objective function.

    Returns
    -------
    dict
        Dictionary with times and errors for the benchmark.
    """

    # Track backend being used
    actual_backend = np.__name__

    # Prepare figure for plotting
    fig, axes = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(8, 6),
        constrained_layout=True,
    )
    print("\n" + "=" * 70)
    print(f"{actual_backend.upper()} Backend Benchmark")
    print("=" * 70)

    # Results storage
    for actual_backend in ["cupy", "scipy", "numpy"]:
        # Technically, this changes the objective function as well.
        # Maybe we should make a cupy-only version of the benchmark.
        if actual_backend == "cupy":
            set_backend_to_cupy()
        else:
            set_backend_to_defaults()

        times_lbfgsb = []
        errors_lbfgsb = []

        for PROBLEM_SIZE in problem_sizes:
            print(f"\nProblem size: {PROBLEM_SIZE} dimensions")

            # Get bounds for the current problem
            lb, ub = get_bounds(PROBLEM_SIZE)

            # Convert to NumPy array explicitly for SciPy comparison
            bounds = tnp.asarray(
                (
                    lb.get() if hasattr(lb, "get") else lb,
                    ub.get() if hasattr(ub, "get") else ub,
                )
            ).T

            lbfgsb_time, lbfgsb_error = run_function_benchmark(
                PROBLEM_SIZE,
                fun,
                jac,
                bounds,
                actual_backend,
            )
            times_lbfgsb.append(lbfgsb_time)
            errors_lbfgsb.append(lbfgsb_error)

        # Plot for this function
        axes.errorbar(
            problem_sizes,
            times_lbfgsb,
            yerr=errors_lbfgsb,
            fmt="-o",
            label=f"L-BFGS-B ({actual_backend})",
            capsize=5,
            ecolor="black",
            linewidth=2,
        )

    # Plotting preferences
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("Problem size (dimensions)", fontsize=12)
    axes.set_ylabel("Time (s) - mean ± std (log scale)", fontsize=12)
    axes.legend(loc="best", fontsize=10)
    axes.grid(True, which="both", linestyle="--", alpha=0.7)

    plt.suptitle(
        "Rosenbrock L-BFGS-B Benchmark",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # Save plot
    plt.savefig(
        f"benchmark_comparison_rosenbrock.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"\nResults saved to 'benchmark_comparison_rosenbrock.png'")

    return


if __name__ == "__main__":
    # Check if CuPy is available
    try:
        import cupy as cp

        HAS_CUPY = True

        def rosenbrock(x):
            x = cp.asarray(x)
            sum1 = ((x[1:] - x[:-1] ** 2.0) ** 2.0).sum(axis=0)
            sum2 = cp.square(1.0 - x[:-1]).sum(axis=0)
            return 100.0 * sum1 + sum2

        def rosenbrock_grad(x):
            x = cp.asarray(x)
            g = cp.zeros(x.shape)
            # derivation of sum1
            g[1:] += 100.0 * (2.0 * x[1:] - 2.0 * x[:-1] ** 2.0)
            g[:-1] += 100.0 * (-4.0 * x[1:] * x[:-1] + 4.0 * x[:-1] ** 3.0)
            # derivation of sum2
            g[:-1] += 2.0 * (x[:-1] - 1.0)
            return g

        print("CuPy is available")
    except ImportError:
        from lbfgsb.benchmarks import rosenbrock, rosenbrock_grad

        HAS_CUPY = False
        print("CuPy is not available")

    set_backend_to_cupy()

    # Run benchmarks for each backend
    results = {}

    # Run NumPy benchmark
    print("\n" + "=" * 70)
    print("Running NumPy Benchmark")
    print("=" * 70)
    results = benchmark_lbfgsb(fun=rosenbrock, jac=rosenbrock_grad)
