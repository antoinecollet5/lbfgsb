"""
Benchmark with multiple benchmark functions.

This benchmark evaluates the performance of L-BFGS-B with CuPy backend
compared to NumPy backend and SciPy's L-BFGS-B implementation.

The benchmark includes error bar plotting to show standard deviation across
multiple trials, and supports configurable problem sizes and number of trials.

References
----------
* Moré, J. J., & Thuente, D. J. (1994). Line search algorithms with guaranteed
  sufficient decrease.
* Byrd, R. H., Lu, P., & Nocedal, J. (1995). A Limited Memory Algorithm for Bound
  Constrained Optimization.

AI Disclosure:
    This benchmark was developed by qwen3.5 while analyzing the L-BFGS-B codebase
    for CuPy backend compatibility improvements.
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy
import numpy as tnp
from lbfgsb import minimize_lbfgsb
from lbfgsb.benchmarks import (
    rosenbrock,
    rosenbrock_grad,
)
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
problem_sizes = [2**i for i in range(8, 10)]  # 2 to 4096

# Function definitions for benchmarking
BENCHMARK_FUNCTIONS = {
    "rosenbrock": (rosenbrock, rosenbrock_grad),
}

# Optimization tolerances
FTOL = 1e-5  # Function tolerance
GTOL = 1e-5  # Gradient tolerance
MAXCOR = 10  # Maximum correlation
MAXITER = 5000  # Maximum iterations

# Number of function evaluations limit
MAXFUN = 20000

# =======================================================================
# HELPER FUNCTIONS
# =======================================================================


def get_bounds_numpy(problem_size):
    """
    Generate uniform bounds for the benchmark problem using NumPy.

    Parameters
    ----------
    problem_size : int
        Number of dimensions.

    Returns
    -------
    tuple
        (lower_bounds, upper_bounds) NumPy arrays.
    """
    lb = tnp.full(problem_size, -5.0)
    ub = tnp.full(problem_size, 5.0)
    return lb, ub


def get_bounds_cupy(problem_size):
    """
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


def get_bounds(problem_size):
    """
    Generate uniform bounds for the benchmark problem.

    Parameters
    ----------
    problem_size : int
        Number of dimensions.

    Returns
    -------
    tuple
        (lower_bounds, upper_bounds) arrays.
    """
    # Use CuPy bounds when using CuPy backend
    if hasattr(np, "_srcmodule") and np._srcmodule.__name__ == "cupy":
        return get_bounds_cupy(problem_size)
    else:
        return get_bounds_numpy(problem_size)


def get_function_label(label):
    """
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


def benchmark_lbfgsb(backend_name=None, force_numpy=False):
    """
    Run the benchmark for a specific backend.

    Parameters
    ----------
    backend_name : str, optional
        'numpy' or 'cupy'. If None, uses current backend.
    force_numpy : bool, optional
        Force NumPy backend regardless of backend_name.

    Returns
    -------
    dict
        Dictionary with times and errors for the benchmark.
    """
    # Set backend if specified
    if backend_name == "cupy":
        set_backend_to_cupy()
    elif backend_name == "numpy" or force_numpy:
        set_backend_to_defaults()

    # Track backend being used
    actual_backend = np.__name__

    print("\n" + "=" * 70)
    print(f"{actual_backend.upper()} Backend Benchmark")
    print("=" * 70)

    # Results storage
    times_lbfgsb = []
    times_scipy = []
    errors_lbfgsb = []
    errors_scipy = []

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

        times_lbfgsb_trial = []
        times_scipy_trial = []

        for benchmark_func_name, (fun, jac) in BENCHMARK_FUNCTIONS.items():
            print(f"\n  {get_function_label(benchmark_func_name).capitalize()}")

            # Reset initial guess within bounds [-5, 5]
            if actual_backend == "cupy":
                x0 = (np.random.random(PROBLEM_SIZE) * 10 - 5).reshape(-1)
                x0 = tnp.clip(x0, -5.0, 5.0).reshape(-1)
            else:
                x0 = tnp.random.random(PROBLEM_SIZE) * 10 - 5
                x0 = tnp.clip(x0, -5.0, 5.0).reshape(-1)

            # Run L-BFGS-B
            print(f"    Running L-BFGS-B optimization...")
            t1 = perf_counter()
            try:
                opt_result = minimize_lbfgsb(
                    x0=x0,
                    fun=fun,
                    jac=jac,
                    bounds=bounds,
                    maxcor=MAXCOR,
                    ftol=FTOL,
                    gtol=GTOL,
                    iprint=0,
                    maxiter=MAXITER,
                    maxfun=MAXFUN,
                )
            except Exception as e:
                print(f"    Optimization failed: {e}")
                continue
            t2 = perf_counter()
            times_lbfgsb_trial.append(t2 - t1)
            print(f"    L-BFGS-B time: {times_lbfgsb_trial[-1]:.4f}s")
            print(f"    Success: {opt_result.success}")

            # Run SciPy's L-BFGS-B for comparison
            print(f"    Running SciPy L-BFGS-B optimization...")
            scipy_fun = lambda x, f=fun: f(x).get() if hasattr(f(x), "get") else f(x)
            scipy_jac = lambda x, g=jac: g(x).get() if hasattr(g(x), "get") else g(x)

            t1 = perf_counter()
            try:
                opt_scipy = minimize(
                    scipy_fun,
                    x0=tnp.random.random(PROBLEM_SIZE).reshape(-1),
                    jac=scipy_jac,
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
            except Exception as e:
                print(f"    SciPy optimization failed: {e}")
                continue
            t2 = perf_counter()
            times_scipy_trial.append(t2 - t1)
            print(f"    SciPy time: {times_scipy_trial[-1]:.4f}s")
            print(f"    Success: {opt_scipy.success}")

        times_lbfgsb.append(numpy.mean(times_lbfgsb_trial))
        times_scipy.append(numpy.mean(times_scipy_trial))

    print("\n" + "=" * 70)
    print(f"{actual_backend.upper()} Benchmark Complete")
    print("=" * 70)
    if times_lbfgsb[-1] > 0:
        print(f"Mean L-BFGS-B time: {times_lbfgsb[-1]:.4f}s")
        print(f"Mean SciPy time: {times_scipy[-1]:.4f}s")
        print(f"Speedup ratio: {times_scipy[-1] / times_lbfgsb[-1]:.2f}x")

    # ========================================================================
    # Plotting Results
    # ========================================================================

    return {
        "times": times_lbfgsb,
        "scipy_times": times_scipy,
        "errors_lbfgsb": errors_lbfgsb,
        "errors_scipy": errors_scipy,
    }


if __name__ == "__main__":
    # Check if CuPy is available
    try:
        import cupy

        HAS_CUPY = True
        print("CuPy is available")
    except ImportError:
        HAS_CUPY = False
        print("CuPy is not available")

    # Run benchmarks for each backend
    results = {}

    # Run NumPy benchmark
    print("\n" + "=" * 70)
    print("Running NumPy Benchmark")
    print("=" * 70)
    results["numpy"] = benchmark_lbfgsb(backend_name="numpy")

    # Run CuPy benchmark if available
    if HAS_CUPY:
        print("\n" + "=" * 70)
        print("Running CuPy Benchmark")
        print("=" * 70)
        results["cupy"] = benchmark_lbfgsb(backend_name="cupy")
    else:
        print("\nSkipping CuPy benchmark - CuPy not installed")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)
    print("\nSummary:")
    for backend, data in results.items():
        if "times" in data and len(data["times"]) > 0:
            avg_time = numpy.mean([t for t in data["times"] if t is not None])
            print(f"{backend.upper()}: avg time = {avg_time:.4f}s")
