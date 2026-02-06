# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

"""
This code is a python port of the famous implementation of Limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS), algorithm 778 written in Fortran [2,3]
(last update in 2011).
Note that this is not a wrapper such as minimize in scipy but a complete
reimplementation (pure python).
The original code can be found here: https://dl.acm.org/doi/10.1145/279232.279236

The aim of this reimplementation was threefold. First, familiarize ourselves with
the code, its logic and inner optimizations. Second, gain access to certain
parameters that are hard-coded in the Fortran code and cannot be modified (typically
wolfe conditions parameters for the line search). Third,
implement additional functionalities that require significant modification of
the code core.

Main interface
^^^^^^^^^^^^^^

Interface for scalar function minimization with L-BFGS-B.

.. autosummary::
   :toctree: _autosummary

    minimize_lbfgsb

Utilitairies
^^^^^^^^^^^^

Additional utilitairy functions to work with inputs or outputs.

.. autosummary::
   :toctree: _autosummary

    extract_hess_inv_diag
    get_grad_projection_inf_norm


Benchmark functions
^^^^^^^^^^^^^^^^^^^

Provide the following function and their gradients to benchmark our implementation.

.. autosummary::
   :toctree: _autosummary

    ackley
    ackley_grad
    beale
    beale_grad
    griewank
    griewank_grad
    quartic
    quartic_grad
    rastrigin
    rastrigin_grad
    rosenbrock
    rosenbrock_grad
    sphere
    sphere_grad
    styblinski_tang
    styblinski_tang_grad

Inner functions
^^^^^^^^^^^^^^^

Inner functions of L-BFGS-B.

.. autosummary::
   :toctree: _autosummary

    base
    bfgsmats
    cauchy
    linesearch
    scalar_function
    subspacemin


References
----------
[1] R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
    Constrained Optimization, (1995), SIAM Journal on Scientific and
    Statistical Computing, 16, 5, pp. 1190-1208.
[2] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
    FORTRAN routines for large scale bound constrained optimization (1997),
    ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
[3] J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
    FORTRAN routines for large scale bound constrained optimization (2011),
    ACM Transactions on Mathematical Software, 38, 1.

"""

# Make scooby a soft dependency:
try:
    from scooby import Report as ScoobyReport
except ImportError:

    class ScoobyReport:
        def __init__(self, *args, **kwargs):
            message = (
                "\n  *ERROR*: `Report` requires `scooby`."
                "\n           Install it via `pip install scooby` or"
                "\n           `conda install -c conda-forge scooby`.\n"
                "\n           Note that python 3.10 or above is required!\n"
            )
            raise ImportError(message)


from lbfgsb import base, bfgsmats, cauchy, linesearch, scalar_function, subspacemin
from lbfgsb.__about__ import __author__, __email__, __version__
from lbfgsb.benchmarks import (
    ackley,
    ackley_grad,
    beale,
    beale_grad,
    griewank,
    griewank_grad,
    quartic,
    quartic_grad,
    rastrigin,
    rastrigin_grad,
    rosenbrock,
    rosenbrock_grad,
    sphere,
    sphere_grad,
    styblinski_tang,
    styblinski_tang_grad,
)
from lbfgsb.main import InternalState, minimize_lbfgsb
from lbfgsb.utils import extract_hess_inv_diag, get_grad_projection_inf_norm


class Report(ScoobyReport):
    def __init__(self, additional=None, ncol=3, text_width=80, sort=False):
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = [
            "lbfgsb",
            "numpy",
            "scipy",
            "typing_extensions",
            "scooby",
        ]

        # Optional packages.
        optional = ["numba", "matplotlib"]

        ScoobyReport.__init__(
            self,
            additional=additional,
            core=core,
            optional=optional,
            ncol=ncol,
            text_width=text_width,
            sort=sort,
        )


__all__ = [
    "minimize_lbfgsb",
    "extract_hess_inv_diag",
    "__author__",
    "__email__",
    "__version__",
    "ackley",
    "ackley_grad",
    "beale",
    "beale_grad",
    "griewank",
    "griewank_grad",
    "quartic",
    "quartic_grad",
    "rastrigin",
    "rastrigin_grad",
    "rosenbrock",
    "rosenbrock_grad",
    "sphere",
    "sphere_grad",
    "styblinski_tang",
    "styblinski_tang_grad",
    "base",
    "bfgsmats",
    "cauchy",
    "linesearch",
    "scalar_function",
    "subspacemin",
    "get_grad_projection_inf_norm",
    "InternalState",
    "Report",
]
