# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

"""
Base functions used by the L-BFGS-B routine.

Functions
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    get_bounds
    is_any_inf
    clip2bounds
    count_var_at_bounds
    projgr
    display_start
    display_iter
    display_results

"""

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from lbfgsb.types import NDArrayFloat


def get_bounds(
    x0: NDArrayFloat, bounds: Optional[ArrayLike]
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Return the lower and upper bounds arrays.

    Parameters
    ----------
    x0 : NDArrayFloat
        Vector of unknowns to optimize.
    bounds : Optional[ArrayLike]
        Array like with shape (n, 2), n being the number of parameters to optimize.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat]
        1-D arrays with lower and upper bounds respectively.

    Raises
    ------
    ValueError
        If x0 is an empty vector or if the length of x0 does not match the length
        of the bounds.
    ValueError
        If there are some values in x0 that violate the bounds.

    """
    n = x0.shape[0]
    if n == 0:
        raise ValueError("x0 cannot be an empty vector!")
    if bounds is None:
        return np.repeat(-np.inf, n), np.repeat(np.inf, n)

    # make sure than None are converted to nan
    _bounds = np.asarray(bounds, dtype=np.float64)
    if np.shape(_bounds) != (n, 2):
        raise ValueError(
            f"Bounds have shape ({np.shape(_bounds)}), while shape "
            f"({n}, 2) is expected!"
        )

    lb, ub = _bounds.T
    # replace nan by inf
    lb[np.isnan(lb)] = -np.inf
    ub[np.isnan(ub)] = np.inf

    # check bounds
    if (lb > ub).any():
        raise ValueError("One of the lower bounds is greater than an upper bound.")

    if (x0 < lb).any() or (x0 > ub).any():
        raise ValueError(
            f"There are {np.count_nonzero(x0 < lb)} values violating the lower bounds"
            f" and {np.count_nonzero(x0 > ub)} values violating the upper bounds!"
        )

    # initial vector must lie within the bounds. Otherwise ScalarFunction and
    # approx_derivative will cause problems
    return lb, ub


def is_any_inf(arrs: Sequence[NDArrayFloat]) -> bool:
    """
    Return whether any of the values in the given arrays is inf.

    Parameters
    ----------
    arrs : Sequence[NDArrayFloat]
        Sequence of arrays.

    Returns
    -------
    bool

    """
    return any([np.isinf(arr).any() for arr in arrs])


def clip2bounds(x0: NDArrayFloat, lb: NDArrayFloat, ub: NDArrayFloat) -> NDArrayFloat:
    """
    Impose the bounds to x0.

    Parameters
    ----------
    x0 : NDArrayFloat
        Adjusted variables. May be a 1D vector of size :math:`N_{n}`,
        or a 2D array of shape (:math:`N_{n}`, :math:`N_{e}`)
        with :math:`N_{n}` the number of adjusted variables and
        :math:`N_{e}` the number of columns (members in the ensemble).
    lb : NDArrayFloat
        Lower bounds (1D vector).
    ub : NDArrayFloat
        Upper bounds (1D vector).

    Returns
    -------
    NDArrayFloat
        Bounded adjusted variables.
    """
    if x0.dtype != np.float64:
        return np.clip(x0.T.astype(np.float64, copy=True), lb, ub).T
    return np.clip(x0.T, lb, ub).T


def count_var_at_bounds(x: NDArrayFloat, lb: NDArrayFloat, ub: NDArrayFloat) -> int:
    """
    Count the number of variables exactly at the bounds.

    Parameters
    ----------
    x : NDArrayFloat
        Adjusted variables. May be a 1D vector of size :math:`N_{n}`,
        or a 2D array of shape (:math:`N_{n}`, :math:`N_{e}`)
        with :math:`N_{n}` the number of adjusted variables and
        :math:`N_{e}` the number of columns (members in the ensemble).
    lb : NDArrayFloat
        Lower bounds.
    ub : NDArrayFloat
        Upper bounds.

    Returns
    -------
    int
        Number of variables exactly at the bounds.
    """
    return np.count_nonzero(np.logical_or(x >= ub, x <= lb)).item()


def display_start(
    epsmch,
    n: int,
    m: int,
    nvar_at_b: int,
    iprint: int,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Display information at solver start.

    Parameters
    ----------
    epsmch : _type_
        Machine precision.
    n : int
        Number of variables.
    m : int
        Number of updates.
    nvar_at_b : int
        Number of variables at bounds.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99``   print details of every iteration except n-vectors;
    logger: Optional[Logger], optional
        :class:`logging.Logger` instance. If None, nothing is displayed, no matter the
        value of `iprint`, by default None.

    """
    if iprint < 0 or logger is None:
        return
    logger.info("RUNNING THE L-BFGS-B CODE")
    logger.info("           * * *")
    logger.info(f"Machine precision = {epsmch}")
    logger.info(f"N = \t{n}\tM = \t{m}")
    logger.info(f"At X0, {nvar_at_b} variables are exactly at the bounds")


def projgr(
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
) -> float:
    """
    Computes the infinity norm of the projected gradient.

    Parameters
    ----------
    x : NDArrayFloat
        Parameters vector with size n.
    g : NDArrayFloat
        Gradient of the cost function with respect to x.
    lb : NDArrayFloat
        Lower bounds vector with size n.
    ub : NDArrayFloat
        Upper bounds vector with size n.

    Returns
    -------
    NDArrayFloat
        Infinity norm of the projected gradient
    """
    return np.max(np.abs(np.clip(x - grad, lb, ub) - x))


def display_iter(
    niter: int,
    sbgnrm: float,
    f: float,
    iprint: int,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Display the objective function and the projected gradient for the iteration.

    Parameters
    ----------
    niter: int
        Current iteration number (0 to n).
    sbgnrm: float
        Infinity norm of the (-) projected gradient.
    iter: int
        Current iteration.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99``   print details of every iteration except n-vectors;
    logger: Optional[Logger], optional
        :class:`logging.Logger` instance. If None, nothing is displayed, no matter the
        value of `iprint`, by default None.

    """
    if iprint >= 1 and logger is not None:
        logger.info(f"At iterate {niter} , f= {f:.3e} , |proj g|= {sbgnrm:.3e}")


def display_results(
    n_iterations: int,
    max_iter,
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    f0: float,
    gtol: float,
    is_final_display: bool,
    iprint: int,
    logger: Optional[logging.Logger] = None,
) -> bool:
    r"""
    Display the optimization results on the fly.

    Parameters
    ----------
    n_iterations : int
        Current number of iterations.
    max_iter : _type_
        Maximum number of iterations allowed.
    x : NDArrayFloat
        Adjusted parameters vectors. Array of real elements of size (n,),
        where ``n`` is the number of independent variables.
    grad : NDArrayFloat
        Gradient of the cost function with respect to `x`.
    lb : NDArrayFloat
        Lower bound vector.
    ub : NDArrayFloat
        Upper bound vector.
    f0 : NDArrayFloat
        Last objective function value.
    gtol : float
        Relative tolerance on gradient.
    is_final_display: bool
        Is it the final display, after convergence or stop.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99``   print details of every iteration except n-vectors;
    logger: Optional[Logger], optional
        :class:`logging.Logger` instance. If None, nothing is displayed, no matter the
        value of `iprint`, by default None.

    """
    if iprint is None or logger is None:
        return False
    if iprint < 0:
        return False
    if iprint == 0 and not is_final_display:
        return False
    elif iprint == 0:
        pass
    elif iprint < 99 and n_iterations % iprint != 0:
        return False
    logger.info(
        f"Iteration #{n_iterations:d} "
        f"(max: {max_iter:d}): "
        f"||x||={np.linalg.norm(x, np.inf):.3e}, "
        f"f(x)={f0:.3e}, "
        f"||jac(x)||={np.linalg.norm(grad, np.inf):.3e}, "
        f"cdt_arret={projgr(x, grad, lb, ub):.3e} "
        f"(eps={gtol:.3e})"
    )
    return True
