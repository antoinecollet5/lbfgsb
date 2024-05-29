"""
Base functions used by the L-BFGS-B routine.

Functions
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    get_bounds
    clip2bounds
    count_var_at_bounds
    projgr
    projgr_ens
    display_start
    display_iter
    display_iter_ensemble
    display_results

"""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize._constraints import old_bound_to_new

from lbfgsb.types import NDArrayFloat


def get_bounds(
    x0: NDArrayFloat, bounds: Optional[NDArrayFloat]
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Return the lower and upper bounds arrays.

    Parameters
    ----------
    x0 : NDArrayFloat
        TODO: x0 can be or an ensemble or a vector. add shapes.
    bounds : Optional[NDArrayFloat]
        _description_

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat]
        1-D arrays with lower and upper bounds respectively.

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """
    n = x0.shape[0]
    if bounds is None:
        bounds = np.repeat(np.array([(-np.inf, np.inf)]), n, axis=0)
    if len(bounds) != n:
        raise ValueError("length of x0 != length of bounds")

    lb, ub = old_bound_to_new(bounds)

    # check bounds
    if (lb > ub).any():
        raise ValueError(
            "LBFGSB - one of the lower bounds is greater than an upper bound."
        )

    # initial vector must lie within the bounds. Otherwise ScalarFunction and
    # approx_derivative will cause problems
    return lb, ub


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
    return (x.T[x.T == ub]).size + (x.T[x.T == lb]).size


def display_start(epsmch, n: int, m: int, nvar_at_b: int, iprint: int) -> None:
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
    """
    if iprint < 0:
        return
    print("RUNNING THE L-BFGS-B CODE")
    print("           * * *")
    print(f"Machine precision = {epsmch}")
    print(f"N = \t{n}\tM = \t{m}")
    print(f"At X0, {nvar_at_b} variables are exactly at the bounds")


def projgr(
    x: NDArrayFloat, grad: NDArrayFloat, lb: NDArrayFloat, ub: NDArrayFloat
) -> float:
    """
    Computes the infinity norm of the projected gradient.

    Parameters
    ----------
    x : NDArrayFloat
        _description_
    g : NDArrayFloat
        _description_
    lb : NDArrayFloat
        _description_
    ub : NDArrayFloat
        _description_

    Returns
    -------
    NDArrayFloat
        Infinity norm of the projected gradient
    """
    return np.max(np.abs(np.clip(x - grad, lb, ub) - x))


def projgr_ens(
    x: NDArrayFloat, grad: NDArrayFloat, lb: NDArrayFloat, ub: NDArrayFloat
) -> NDArrayFloat:
    """
    Computes the minimal infinity norm of the projected gradient of the ensemble

    Parameters
    ----------
    x : NDArrayFloat
        _description_
    g : NDArrayFloat
        _description_
    lb : NDArrayFloat
        _description_
    ub : NDArrayFloat
        _description_

    Returns
    -------
    NDArrayFloat
        Infinity norm of the projected gradient
    """
    return np.array([projgr(_x, _g, lb, ub) for (_x, _g) in zip(x.T, grad.T)])


def display_iter(iter: int, sbgnrm: float, f: float, iprint: int) -> None:
    """
    Compute the infinity norm of the (-) projected gradient.

    Parameters
    ----------
    iter: int
        Current iteration number (0 to n).
    sbgnrm: float
        Infinity norm of the (-) projected gradient.
    iter: int
        Current iteration.
    iprint: int
        Level of display.
    """
    if iprint > 1:
        print(f"At iterate {iter} , f= {f} , |proj g|= {sbgnrm}")


def display_iter_ensemble(iter: int, sbgnrm: float, f: float, iprint: int) -> None:
    """
    Compute the infinity norm of the (-) projected gradient.

    Parameters
    ----------
    iter: int
        Current iteration number (0 to n).
    sbgnrm: float
        Infinity norm of the (-) projected gradient.
    iter: int
        Current iteration.
    iprint: int
        Level of display.
    """
    if iprint > 1:
        print(f"At iterate {iter} , min(f)= {f} , min(|proj g|)= {sbgnrm}")


def display_results(
    iprint: int,
    n_iterations: int,
    max_iter,
    x: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    f0: float,
    gtol: float,
    is_final_display: bool,
) -> None:
    r"""
    Disaply the optimization results on the fly.

    Parameters
    ----------
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99``   print details of every iteration except n-vectors;
    n_iterations : int
        _description_
    max_iter : _type_
        _description_
    x : NDArrayFloat
        _description_
    grad : NDArrayFloat
        _description_
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
    """
    if iprint is None:
        return
    if iprint < 0:
        return
    if iprint == 0 and not is_final_display:
        return
    if iprint < 99 and n_iterations % iprint != 0:
        return
    print(
        "Iteration #%d (max: %d): ||x||=%.3e, f(x)=%.3e, ||jac(x)||=%.3e, "
        "cdt_arret=%.3e (eps=%.3e)"
        % (
            n_iterations,
            max_iter,
            np.linalg.norm(x, np.inf),
            f0,
            np.linalg.norm(grad, np.inf),
            projgr(x, grad, lb, ub),
            gtol,
        )
    )
