==============
Changelog
==============

1.1.0 (2026-06-05)
------------------

Added
~~~~~

* Added SciPy-compatible support for ``jac=True`` in ``minimize_lbfgsb`` and
  ``prepare_scalar_function``. When ``jac=True``, the objective function may now
  return both the objective value and gradient as ``fun(x) -> (f, g)``.
* Added memoization support for objective functions returning ``(f, g)`` to avoid
  duplicate evaluations when both the function value and gradient are requested
  at the same point.

Changed
~~~~~~~

* Improved typing annotations for objective functions, gradients, finite-difference
  options, and scalar-function wrappers.
* Improved compatibility with static type checkers, including ``ty``.
* Improved Python 3.7 compatibility for typing-related constructs.
* Updated line-search handling to be more defensive against non-finite trial
  points, non-finite directional derivatives, and invalid step lengths.
* Improved numerical robustness of line-search utilities, including safer dot
  products, safer squared-norm computations, and safer maximum step-length
  detection.
* Replaced mutable default work arrays in the line-search routine with
  ``None`` defaults initialized inside the function.

Fixed
~~~~~

* Fixed scalar objective normalization in ``ScalarFunction`` so that objective
  values are consistently converted to plain Python ``float`` values.
* Fixed support for complex-step finite differences by preserving complex
  objective values during numerical differentiation.
* Fixed typing issues caused by broad NumPy scalar/object unions in objective
  function evaluation.
* Fixed potential use of uninitialized line-search step variables.

1.0.1 (2026-02-10)
------------------

* FIX: `typing_extensions` and `packaging` dependencies for all python versions.

1.0.0 (2025-12-31)
------------------

This is the first stable release with a non negligeable number of new features and
performance improvement:

* Checkpointing (start and stop mechanisms)
* On-the-fly modification of the objective function
* Added benchmarks with scipy
* Provide an optional numba jit implementation for the expensive parts of the code.
* Enhanced documentation and tests.

0.1.1 (2024-06-29)
------------------

* First release on PyPI.
