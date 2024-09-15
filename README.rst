======
LBFGSB
======

|License| |Stars| |Python| |PyPI| |Downloads| |Build Status| |Documentation Status| |Coverage| |Codacy| |Precommit: enabled| |Ruff| |Mypy| |DOI|

A python impementation of the famous L-BFGS-B quasi-Newton solver [1].

This code is a python port of the famous implementation of Limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS), algorithm 778 written in Fortran [2,3]
(last update in 2011).
Note that this is not a wrapper like `minimize`` in scipy but a complete
reimplementation (pure python).
The original Fortran code can be found here: https://dl.acm.org/doi/10.1145/279232.279236

Motivations
-----------

Although there are many implementations or ports (wrappings) of the lbfgsb code,
as evidenced by the list compiled by `Jonathan Schilling <https://github.com/jonathanschilling/L-BFGS-B>`_,
in the vast majority, these are merely interfaces (wrapper) to access highly optimized
Fortran or C code. In Python, for example, this is the case for `scipy <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_.
These codes mainly date back to the 90s and are difficult to understand, and therefore difficult to maintain or modify.
Incidentally, the only other python implementation we know of to date,
by `Avieira <https://github.com/avieira/python_lbfgsb>`_, is not very optimized and under GPL3 license,
which makes it tricky to use.

In this context, the objectives of this code are as follows:

- learn the underlying mechanisms of lbfgsb code;
- provide understandable, modern code using the high-level language python, while using typing, explicit function names and standardized formatting thanks to `Ruff <https://docs.astral.sh/ruff/>`_;
- provide detailed and explicit documentation;
- offer totally free code, including for commercial use, thanks to the MIT license;
- garantee efficient code, with the number of calls to the objective function and gradient at least as low as in the reference implementation, and without drastically increasing memory consumption or computation time, thanks to the use of numpy and vectorization;
- add relevant stopping criteria;
- add the possibility to restart the solver from a checkpoint;
- add the possibility of modifying on-the-fly the gradient sequences stored in memory, an essential mechanism for the automatic and adaptive weighting of a possible regularization term, See (TODO). This is one of the initial motivation;
- use a logging system rather than prints, for better integration within complex apps.

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

* Free software: MIT license
* Documentation: https://lbfgsb.readthedocs.io.

Quick start
-----------

Given an optimization problem defined by an objective function and a feasible space:

.. code-block:: python

   import numpy as np
   from lbfgsb.types import NDArrayFloat  # for type hints, numpy array of floats

    def rosenbrock(x: NDArrayFloat) -> float:
        """
        The Rosenbrock function.

        Parameters
        ----------
        x : array_like
            1-D array of points at which the Rosenbrock function is to be computed.

        Returns
        -------
        float
            The value of the Rosenbrock function.

        """
        x = np.asarray(x)
        sum1 = ((x[1:] - x[:-1] ** 2.0) ** 2.0).sum()
        sum2 = np.square(1.0 - x[:-1]).sum()
        return 100.0 * sum1 + sum2


    def rosenbrock_grad(x: NDArrayFloat) -> NDArrayFloat:
        """
        The gradient of the Rosenbrock function.

        Parameters
        ----------
        x : array_like
            1-D array of points at which the Rosenbrock function is to be derivated.

        Returns
        -------
        NDArrayFloat
            The gradient of the Rosenbrock function.
        """
        x = np.asarray(x)
        g = np.zeros(x.size)
        # derivation of sum1
        g[1:] += 100.0 * (2.0 * x[1:] - 2.0 * x[:-1] ** 2.0)
        g[:-1] += 100.0 * (-4.0 * x[1:] * x[:-1] + 4.0 * x[:-1] ** 3.0)
        # derivation of sum2
        g[:-1] += 2.0 * (x[:-1] - 1.0)
        return g

   lb = np.array([-2, -2])  # lower bounds
   ub = np.array([2, 2])  # upper bounds
   bounds = np.array((l, u)).T  # The number of variables to optimize is len(bounds)
   x0 = np.array([-0.8, -1])  # The initial guess

The optimal solution can be found following:

.. code-block:: python

   from lbfgsb import minimize_lbfgsb

   x = minimize_lbfgsb(
     x0=x0, fun=rosenbrock, jac=rosenbrock_grad, bounds=bounds, ftol=1e-5, gtol=1e-5
   )

``minimize_lbfgsb`` returns an `OptimalResult` instance (from scipy) that contains the results of the optimization:

.. code-block::

    message: CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL
    success: True
     status: 0
        fun: 3.9912062309350614e-08
          x: [ 1.000e+00  1.000e+00]
        nit: 18
        jac: [-6.576e-02  3.220e-02]
       nfev: 23
       njev: 23
   hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>

See all use cases in the tutorials section of the `documentation <https://lbfgsb.readthedocs.io/en/latest/usage.html>`_.


.. |License| image:: https://img.shields.io/badge/License-MIT license-blue.svg
    :target: https://github.com/antoinecollet5/lbfgsb/-/blob/master/LICENSE

.. |Stars| image:: https://img.shields.io/github/stars/antoinecollet5/lbfgsb.svg?style=social&label=Star&maxAge=2592000
    :target: https://github.com/antoinecollet5/lbfgsb/stargazers
    :alt: Stars

.. |Python| image:: https://img.shields.io/pypi/pyversions/lbfgsb.svg
    :target: https://pypi.org/pypi/lbfgsb
    :alt: Python

.. |PyPI| image:: https://img.shields.io/pypi/v/lbfgsb.svg
    :target: https://pypi.org/pypi/lbfgsb
    :alt: PyPI

.. |Downloads| image:: https://static.pepy.tech/badge/lbfgsb
    :target: https://pepy.tech/project/lbfgsb
    :alt: Downoads

.. |Build Status| image:: https://github.com/antoinecollet5/lbfgsb/actions/workflows/main.yml/badge.svg
    :target: https://github.com/antoinecollet5/lbfgsb/actions/workflows/main.yml
    :alt: Build Status

.. |Documentation Status| image:: https://readthedocs.org/projects/lbfgsb/badge/?version=latest
    :target: https://lbfgsb.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Coverage| image:: https://codecov.io/gh/antoinecollet5/lbfgsb/branch/master/graph/badge.svg?token=ISE874MMOF
    :target: https://codecov.io/gh/antoinecollet5/lbfgsb
    :alt: Coverage

.. |Codacy| image:: https://app.codacy.com/project/badge/Grade/c41f65d98b824de394162520b0d8a17a
    :target: https://app.codacy.com/gh/antoinecollet5/lbfgsb/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
    :alt: codacy

.. |Precommit: enabled| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/pre-commit/pre-commit

.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
    :target: https://github.com/psf/black
    :alt: Black

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. |Mypy| image:: https://www.mypy-lang.org/static/mypy_badge.svg
    :target: https://mypy-lang.org/
    :alt: Checked with mypy

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11384588.svg
   :target: https://doi.org/10.5281/zenodo.11384588
