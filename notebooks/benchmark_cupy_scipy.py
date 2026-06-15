"""
Benchmark with multiple benchmark functions.

This benchmark evaluates the performance of L-BFGS-B across backends
(NumPy, NumPy+Numba, CuPy, SciPy) compared across multiple problem sizes.

Backends are selected automatically from the array type passed to
``minimize_lbfgsb`` — no global ``set_backend_to_*`` call is needed.
Pass a CuPy array as ``x0`` and the solver uses the GPU; pass a NumPy
array and it uses the CPU.

AI Disclosure
    Originally generated with qwen3.5 assistance; updated to the new
    POT-style backend API (``get_backend`` / auto-detection).
"""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from lbfgsb import minimize_lbfgsb
from lbfgsb.backend import get_backend
from scipy.optimize import minimize

# =======================================================================
# CONFIGURATION
# =======================================================================

NTRIALS = 3  # trials per (backend, problem_size) cell
FTOL = 1e-5
GTOL = 1e-5
MAXCOR = 10
MAXITER = 20_000
MAXFUN = 20_000

problem_sizes = [2**i for i in range(16, 21)]

# Backends to benchmark; skip "cupy" automatically when not installed.
ALL_BACKENDS = ["numpy+numba", "cupy", "scipy"]

# =======================================================================
# PROBLEM DEFINITIONS
# All functions accept either NumPy or CuPy arrays transparently via
# get_backend() — no explicit conversion needed.
# =======================================================================


def rosenbrock(x):
    nx = get_backend(x)
    _ = nx  # used only for type inference; arithmetic uses __add__ etc.
    return float(
        100.0 * ((x[1:] - x[:-1] ** 2.0) ** 2.0).sum() + ((1.0 - x[:-1]) ** 2.0).sum()
    )


def rosenbrock_grad(x):
    nx = get_backend(x)
    g = nx.zeros_like(x)
    g = nx.set_item(g, slice(1, None), g[1:] + 100.0 * 2.0 * (x[1:] - x[:-1] ** 2.0))
    g = nx.set_item(
        g,
        slice(None, -1),
        g[:-1]
        + 100.0 * (-4.0 * x[1:] * x[:-1] + 4.0 * x[:-1] ** 3.0)
        + 2.0 * (x[:-1] - 1.0),
    )
    return g


# =======================================================================
# BENCHMARK HELPERS
# =======================================================================


def _make_x0(problem_size: int, backend: str):
    """Return an initial guess in the correct array type for *backend*."""
    x0_np = np.clip(np.random.uniform(-5.0, 5.0, problem_size), -5.0, 5.0)

    if backend == "cupy":
        import cupy as cp

        return cp.asarray(x0_np)
    # numpy, numpy+numba, scipy — all use plain numpy
    return x0_np


def _make_bounds(problem_size: int, backend: str):
    """Return a (n, 2) bounds array in the format expected by minimize_lbfgsb."""
    lb = np.full(problem_size, -5.0)
    ub = np.full(problem_size, 5.0)
    # bounds must always be numpy for scipy; minimize_lbfgsb accepts numpy too
    # (the solver clips x0 at entry; bounds themselves are never put on GPU)
    return np.column_stack([lb, ub])


def _run_once(
    backend: str,
    problem_size: int,
    fun: Callable,
    jac: Callable,
) -> tuple[float, bool]:
    """Run one trial; return (wall_time_seconds, success)."""
    x0 = _make_x0(problem_size, backend)
    bounds = _make_bounds(problem_size, backend)

    t0 = perf_counter()

    if backend == "scipy":
        result = minimize(
            x0=x0,
            fun=lambda x: float(fun(x)),
            jac=lambda x: np.asarray(jac(x)),
            bounds=bounds,
            method="L-BFGS-B",
            options={
                "maxiter": MAXITER,
                "ftol": FTOL,
                "gtol": GTOL,
                "iprint": -1,
                "maxcor": MAXCOR,
            },
        )
    else:
        result = minimize_lbfgsb(
            x0=x0,
            fun=fun,
            jac=jac,
            bounds=bounds,
            maxcor=MAXCOR,
            ftol=FTOL,
            gtol=GTOL,
            iprint=-1,
            maxiter=MAXITER,
            maxfun=MAXFUN,
            is_use_numba_jit=(backend == "numpy+numba"),
        )

    wall = perf_counter() - t0
    return wall, bool(result.success)


def benchmark(
    fun: Callable,
    jac: Callable,
    backends: Optional[list[str]] = None,
) -> dict[str, dict[str, list[float]]]:
    """
    Run the full benchmark matrix.

    Returns
    -------
    dict
        ``results[backend]["times"]``  — list of mean times per problem size
        ``results[backend]["errors"]`` — list of std times per problem size
    """
    if backends is None:
        backends = ALL_BACKENDS

    # Filter out unavailable backends
    available: list[str] = []
    for b in backends:
        if b == "cupy":
            try:
                import cupy  # noqa: F401

                available.append(b)
            except ImportError:
                print("  [skip] cupy not installed")
        elif b == "numpy+numba":
            try:
                import numba  # noqa: F401

                available.append(b)
            except ImportError:
                print("  [skip] numba not installed")
        else:
            available.append(b)

    results: dict[str, dict[str, list[float]]] = {}

    for backend in available:
        print(f"\n{'=' * 60}")
        print(f"Backend: {backend.upper()}")
        print(f"{'=' * 60}")
        times_mean: list[float] = []
        times_std: list[float] = []

        for size in problem_sizes:
            trial_times: list[float] = []
            print(f"  size={size:>7d}  ", end="", flush=True)

            for t in range(NTRIALS):
                wall, ok = _run_once(backend, size, fun, jac)
                trial_times.append(wall)
                status = "✓" if ok else "✗"
                print(f"[{status} {wall:.3f}s]", end=" ", flush=True)

            mu = float(np.mean(trial_times))
            std = float(np.std(trial_times))
            times_mean.append(mu)
            times_std.append(std)
            print(f"→ {mu:.3f}±{std:.3f}s")

        results[backend] = {"times": times_mean, "errors": times_std}

    return results


# =======================================================================
# PLOTTING
# =======================================================================

_STYLES: dict[str, dict] = {
    "numpy": {"fmt": "-o", "color": "steelblue", "label": "lbfgsb (numpy)"},
    "numpy+numba": {
        "fmt": "-s",
        "color": "darkorange",
        "label": "lbfgsb (numpy+numba)",
    },
    "cupy": {"fmt": "-^", "color": "green", "label": "lbfgsb (cupy)"},
    "scipy": {"fmt": "--D", "color": "firebrick", "label": "scipy L-BFGS-B"},
}


def plot_results(
    results: dict[str, dict[str, list[float]]],
    title: str = "L-BFGS-B Backend Benchmark (Rosenbrock)",
    outfile: str = "benchmark_comparison_rosenbrock.png",
) -> None:
    """Plot mean ± std wall-clock time vs problem size for every backend."""
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    for backend, data in results.items():
        style = _STYLES.get(backend, {"fmt": "-o", "color": "grey", "label": backend})
        ax.errorbar(
            problem_sizes[: len(data["times"])],
            data["times"],
            yerr=data["errors"],
            fmt=style["fmt"],
            color=style["color"],
            label=style["label"],
            capsize=4,
            linewidth=2,
            markersize=6,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Problem size (n)", fontsize=12)
    ax.set_ylabel("Wall time (s)  —  mean ± std", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Annotate NTRIALS
    ax.text(
        0.02,
        0.02,
        f"trials per cell: {NTRIALS}  |  ftol={FTOL}  gtol={GTOL}",
        transform=ax.transAxes,
        fontsize=8,
        color="grey",
    )

    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {outfile}")
    plt.show()


# =======================================================================
# ENTRY POINT
# =======================================================================

if __name__ == "__main__":
    print("L-BFGS-B backend benchmark")
    print(f"problem sizes : {problem_sizes}")
    print(f"trials/cell   : {NTRIALS}")

    results = benchmark(fun=rosenbrock, jac=rosenbrock_grad)
    plot_results(results)
