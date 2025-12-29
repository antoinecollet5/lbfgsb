======
LBFGSB
======

|License| |Stars| |Python| |PyPI| |Downloads| |Build Status| |Documentation Status| |Coverage| |Codacy| |Precommit: enabled| |Ruff| |ty| |DOI|

🐍 A python impementation of the famous L-BFGS-B quasi-Newton solver [1].

This code is a python port of the famous implementation of Limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS), algorithm 778 written in Fortran [2,3]
(last update in 2011).
Note that this is not a wrapper like `minimize`` in `Scipy <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_ but a complete
reimplementation (pure python).
The original Fortran code can be found here: https://dl.acm.org/doi/10.1145/279232.279236

**The complete and up to date documentation can be found here**: https://lbfgsb.readthedocs.io.

===============
🎯 Motivations
===============

Although there are many implementations or ports (wrappings) of the lbfgsb code,
as evidenced by the list compiled by `Jonathan Schilling <https://github.com/jonathanschilling/L-BFGS-B>`_,
in the vast majority, these are merely interfaces (wrapper) to access highly optimized
Fortran or C code. In Python, for example, this is the case for `Scipy <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_.
These codes mainly date back to the 90s and are difficult to understand, and therefore difficult to maintain or modify.
Incidentally, the only other python implementation we know of to date,
by `Avieira <https://github.com/avieira/python_lbfgsb>`_, is not very optimized and under GPL3 license,
which makes it tricky to use.

In this context, the objectives of this code are as follows:

- Learn the underlying mechanisms of lbfgsb code;
- Provide understandable, modern code using the high-level language python, while using typing, explicit function names and standardized formatting thanks to `Ruff <https://docs.astral.sh/ruff/>`_ and `ty <https://docs.astral.sh/ty/>`_;
- Provide detailed and explicit documentation;
- Offer totally free code, including for commercial use, thanks to the **BSD 3-Clause License**;
- Garantee efficient code, with the number of calls to the objective function and gradient at least as low as in the reference implementation, and without drastically increasing memory consumption or computation time, thanks to the use of numpy and vectorization;
- Add relevant stopping criteria;
- Add the possibility to restart the solver from a checkpoint;
- Add the possibility of modifying on-the-fly the gradient sequences stored in memory, an essential mechanism for the automatic and adaptive weighting of a possible regularization term, See (TODO). This is one of the initial motivation;
- Use a logging system rather than `prints`, for better integration within complex apps.

===============
🚀 Quick start
===============

To install `lbfgsb`, the easiest way is through `pip`:

.. code-block::

    pip install lbfgsb

Or alternatively using `conda`

.. code-block::

    conda install lbfgsb

You might also clone the repository and install from source

.. code-block::

    pip install -e .

Once the installation is done, given an optimization problem defined by an objective function and a feasible space:

.. code-block:: python

   import numpy as np
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

   lb = np.array([-2, -2])  # lower bounds
   ub = np.array([2, 2])  # upper bounds
   bounds = np.array((l, u)).T  # The number of variables to optimize is len(bounds)
   x0 = np.array([-0.8, -1])  # The initial guess

The optimal solution can be found following:

.. code-block:: python

   from lbfgsb import minimize_lbfgsb

   res = minimize_lbfgsb(
     x0=x0, fun=rosenbrock, jac=rosenbrock_grad, bounds=bounds, ftol=1e-5, gtol=1e-5
   )

``minimize_lbfgsb`` returns an `OptimalResult` instance (from `Scipy <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_) that contains the results of the optimization:

.. code-block::

    message: CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL
    success: True
    status: 0
    fun: 5.834035724922196e-07
    x: [ 9.994e-01  9.989e-01]
    nit: 16
    jac: [-2.171e-02  1.030e-02]
    nfev: 19
    njev: 19
    hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>

Note that unlike `minimize` from scipy, `minimize_lbfgsb` does not accept `args` nor `kwargs`. Hence you will have to define
wrappers if needed.

.. code-block:: python

    # This cannot be passed to minimize_lbfgsb
    def my_cost_function(x: NDArrayFloat, arg1: int, arg2: float, *, kwargs1: int=0, kwargs2: str="blabla") -> float:
        """
        Return a float and takes args and kwargs.
        """
        ... # just do something and return a float

    # This can be passed to minimize_lbfgsb
    def my_wrapper(x: NDArrayFloat) -> float:
        """
        Return a float and takes args and kwargs.
        """
        retur my_cost_function(x, 10, 239.9, kwargs1=1, kwargs2="blabla2")


See all use cases in the tutorials section of the `documentation <https://lbfgsb.readthedocs.io/en/latest/usage.html>`_.

===============
⚡ Performance
===============

Although memory usage and runtime remain reasonable thanks to NumPy and extensive
vectorization, a pure Python implementation is inherently slower and more memory-intensive
than the SciPy reference implementation. The latter is written in low-level languages
(`originally Fortran 77 and, since SciPy v1.15.0, ported to C <https://docs.scipy.org/doc/scipy/release/1.15.0-notes.html#scipy-optimize-improvements>`_)) and therefore benefits
from decades of compiler and library-level optimizations.

In practice, this performance gap is negligible for small- to medium-scale problems,
or when the overall runtime is dominated by evaluations of the objective function and
its gradient rather than by the optimization routine itself.

To mitigate the overhead of Python in performance-critical sections, we provide a
Numba JIT-compiled implementation of the most expensive components of the algorithm.
This approach significantly reduces Python overhead and brings performance close to
that of `Scipy <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_
for large-scale problems (2 fold speed up for 1M parameters), while still preserving the additional flexibility and features offered
by our implementation. The only overhead introduced is a one-time LLVM compilation
during the first call; subsequent calls benefit from Numba’s caching mechanism.

Numba acceleration is enabled by default and can be disabled via the boolean
parameter `is_use_numba_jit`. If this option is set to True but Numba is not available,
a warning is issued and the code automatically falls back to the non-JIT implementation.

.. code-block:: python

   res = minimize_lbfgsb(
     x0=x0,
     fun=rosenbrock,
     jac=rosenbrock_grad,
     bounds=bounds,
     ftol=1e-5,
     gtol=1e-5,
     is_use_numba_jit=False
   )

Note that numba comes as an optional dependency and should be installed in one of the following ways:

.. code-block::

    pip install lbfgsb[numba]

.. code-block::

    pip install lbfgsb numba

Or alternatively using conda

.. code-block::

    conda install lbfgsb[numba]

.. code-block::

    conda install lbfgsb numba

If numba is not found in your environement, a RunTime warning will be raised.

===================
🛠️ Unique features
===================

Here are some of the unique features that this implementation provides (to the best of our knowledge in 2025).

✨ Checkpointing
-----------------

In quasi-Newtons (L-BFGS-B is one of them), the inverse of the Hessian is approximated from the
series of the (`m` last) past gradients. If the optimization is stopped, the history is lost and restarting
the optimization would results in a slower convergence (more total objective function and gradient calls) because
the inverse Hessian would be reinitiated as the identity.

Let's take the previous example with the rosenbrock objective function. But this time, the process is stopped after three calls of the function (`maxfun=3`)

.. code-block:: python

    res_3_fun = minimize_lbfgsb(
        x0=x0,
        fun=rosenbrock,
        jac=rosenbrock_grad,
        bounds=bounds,
        ftol=1e-5,
        gtol=1e-5,
        maxfun=3
    )
    print(res_3_fun)

It yields

.. code-block::

    message: STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT
    success: True
    status: 1
    fun: 0.8619211711864526
    x: [ 6.370e-01  4.913e-01]
    nit: 1
    jac: [-2.250e+01  1.709e+01]
    nfev: 3
    njev: 3
    hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>

Resuming the minimization from the previous parameters state (`x0=res_3_fun.x`):

.. code-block:: python

    res_final = minimize_lbfgsb(
        x0=res_3_fun.x,
        fun=rosenbrock,
        jac=rosenbrock_grad,
        bounds=bounds,
        ftol=1e-5,
        gtol=1e-5
    )
    print(res_final)

gives

.. code-block::

    message: CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL
    success: True
    status: 0
    fun: 1.349454245280619e-10
    x: [ 1.000e+00  1.000e+00]
    nit: 14
    jac: [ 1.030e-04 -6.267e-05]
    nfev: 21
    njev: 21
    hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>

One can see that the total number of calls to the objective function and to its gradient is higher (`3+21 = 24` vs `19` originally).
In addition, one needs to compute it manually because it looses track of the state when restarting `L-BFGS-B`.
Also one sees that the final result is different.

With the checkpointing, it is possible to restore the last state in a straighforward manner and obtain strictly the same results:

.. code-block:: python

    res_checkpoint = minimize_lbfgsb(
        x0=res_3_fun.x,
        fun=rosenbrock,
        jac=rosenbrock_grad,
        bounds=bounds,
        ftol=1e-5,
        gtol=1e-5,
        checkpoint=res_3_fun
    )
    print(res_checkpoint)


yields

.. code-block::

    message: CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL
    success: True
    status: 0
    fun: 5.834035724922196e-07
    x: [ 9.994e-01  9.989e-01]
    nit: 16
    jac: [-2.171e-02  1.030e-02]
    nfev: 19
    njev: 19
    hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>

Note that this time, we keep track of `nfev` and `njev`. In addition, the results is strictly the same as minimizing the function in
a single run. This can be pretty useful if computing the objective function and its gradient is expensive but one
is not so sure about what stopping criteria to use. TODO: add something about use case for scaling.

✨ Callback
------------

Our implementation of L-BFGS-B allows to use several standard stop criteria:

• The absolute value of the objective function
• The change in objective function value between two iterations.
• And the norm of the objective function gradient.
• The maximum number of iterations.
• The maximum number of objective function calls.

The callback mechanism allows to enhance these possibilities and define custom stopping criteria.
For example, one can redefine the criterion based on the number of objective function evaluations
(`maxfun`). If the algorithm should stop, the callback must return `True`.

.. code-block:: python

    import numpy as np
    from scipy.optimize import OptimizeResult

    def my_custom_callback(xk: np.typing.NDArray[np.float64], state: OptimizeResult) -> bool:
        # if the objective function has been called 10 times or more => stop
        if state.nfev >= 10:
            return True
        return False


    res_callback = minimize_lbfgsb(
        x0=x0,
        fun=rosenbrock,
        jac=rosenbrock_grad,
        bounds=bounds,
        ftol=1e-5,
        gtol=1e-5,
        callback=my_custom_callback
    )
    print(res_callback)

yields

.. code-block::

    message: STOP: USER CALLBACK
    success: True
    status: 2
    fun: 0.08537354414890966
    x: [ 7.354e-01  5.284e-01]
    nit: 8
    jac: [ 3.115e+00 -2.478e+00]
    nfev: 10
    njev: 10
    hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>

✨ Cost function update
------------------------

Function to update the gradient sequence. This allows changing the objective function definition on the fly.
In the first place this functionality is dedicated to regularized problems for which the
regularization weight is computed while optimizing the cost function. In order to get a
Hessian matching the new definition of `fun`, the gradient sequence must be updated.

.. code-block::

    ``update_fun_def(x, f0, f0_old, grad, x_deque, grad_deque)
    -> f0, f0_old, grad, updated grad_deque``

🏗️ Complete example with supporting paper coming Q1 2026.

===========
🔑 License
===========

This project is released under the **BSD 3-Clause License**.

Copyright (c) 2025, Antoine COLLET. All rights reserved.

For more details, see the `LICENSE <https://github.com/antoinecollet5/lbfgsb/blob/master/LICENSE>`_ file included in this repository.

==============
⚠️ Disclaimer
==============

This software is provided "as is", without warranty of any kind, express or implied,
including but not limited to the warranties of merchantability, fitness for a particular purpose,
or non-infringement. In no event shall the authors or copyright holders be liable for
any claim, damages, or other liability, whether in an action of contract, tort,
or otherwise, arising from, out of, or in connection with the software or the use
or other dealings in the software.

By using this software, you agree to accept full responsibility for any consequences,
and you waive any claims against the authors or contributors.

==========
📧 Contact
==========

For questions, suggestions, or contributions, you can reach out via:

- Email: antoinecollet5@gmail.com
- GitHub: https://github.com/antoinecollet5/lbfgsb

We welcome contributions!

=============
📚 References
=============

[1] R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
Constrained Optimization, (1995), SIAM Journal on Scientific and
Statistical Computing, 16, 5, pp. 1190-1208.

[2] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
FORTRAN routines for large scale bound constrained optimization (1997),
ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.

[3] J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
FORTRAN routines for large scale bound constrained optimization (2011),
ACM Transactions on Mathematical Software, 38, 1.

* Free software: SPDX-License-Identifier: BSD-3-Clause


.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
    :target: https://github.com/antoinecollet5/lbfgsb/blob/master/LICENSE

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

.. |ty| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json
    :target: https://github.com/astral-sh/ty
    :alt: Checked with ty

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11384588.svg
   :target: https://doi.org/10.5281/zenodo.11384588
