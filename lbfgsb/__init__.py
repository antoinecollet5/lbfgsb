"""
@author: Antoine COLLET.

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

Additional features
--------------------
Explain about objective function update on the fly.
TODO: point to the doc of the main routine.

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

Provide the following benchmark functions.

    ackley
    griewank
    quartic
    rastrigin
    rosenbrock
    sphere
    styblinski_tang

"""

from lbfgsb.__about__ import __author__, __email__, __version__
from lbfgsb.benchmarks import (
    ackley,
    ackley_grad,
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
    styblinski_tang_grad
)
from lbfgsb.main import minimize_lbfgsb
from lbfgsb.utils import extract_hess_inv_diag

# TODO: add references to all internal functions

__all__ = [
    "minimize_lbfgsb",
    "extract_hess_inv_diag",
    "__author__",
    "__email__",
    "__version__",
    "ackley",
    "griewank",
    "quartic",
    "rastrigin",
    "rosenbrock",
    "sphere",
    "styblinski_tang",
]
