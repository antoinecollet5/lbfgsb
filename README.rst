======
LBFGSB
======

|License| |Stars| |Python| |PyPI| |Downloads| |Build Status| |Documentation Status| |Coverage| |Codacy| |Precommit: enabled| |Code style: black| |Ruff| |Mypy| |DOI|

A python impementation of the famous L-BFGS-B quasi-Newton solver [1].

This code is a python port of the famous implementation of Limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS), algorithm 778 written in Fortran [2,3]
(last update in 2011).
Note that this is not a wrapper like `minimize`` in scipy but a complete
reimplementation (pure python).
The original Fortran code can be found here: https://dl.acm.org/doi/10.1145/279232.279236

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

The aim of this reimplementation was threefold. First, familiarize ourselves with
the code, its logic and inner optimizations. Second, gain access to certain
parameters that are hard-coded in the Fortran code and cannot be modified (typically
wolfe conditions parameters for the line search). Third,
implement additional functionalities that require significant modification of
the code core.

* Free software: MIT license
* Documentation: https://lbfgsb.readthedocs.io.

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

.. |Codacy| image:: https://app.codacy.com/project/badge/Grade/a3ad37554c5845e6a27e096e77dcca2f
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
