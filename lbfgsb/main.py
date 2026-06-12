# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

"""
This code is a python port of the famous implementation of Limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS), algorithm 778 written in Fortran [2,3]
(last update in 2011).
Note that this is not a wrapper such as minimize in scipy but a complete
reimplementation (nevertheless relying heavily on numpy and scipy to
maintain correct performances).
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
TODO:
https://towardsdatascience.com/numerical-optimization-based-on-the-l-bfgs-method-f6582135b0ca

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

import copy
import logging
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple, Union

import numpy as tnp
from numpy.typing import ArrayLike
from scipy.optimize import (
    LbfgsInvHessProduct,  # noqa : F401
    OptimizeResult,
)

from lbfgsb._numba_helpers import NUMBA_AVAILABLE
from lbfgsb.base import (
    clip2bounds,
    count_var_at_bounds,
    display_iter,
    display_results,
    display_start,
    get_bounds,
    is_any_inf,
    projgr,
)
from lbfgsb.bfgsmats import (
    LBFGSB_MATRICES,
    make_X_and_G_respect_strong_wolfe,
    update_lbfgs_matrices,
)
from lbfgsb.cauchy import get_cauchy_point
from lbfgsb.linesearch import line_search
from lbfgsb.mathops import np, sp
from lbfgsb.scalar_function import (
    JacOption,
    Objective,
    ObjectiveAndGradient,
    ScalarFunction,
    prepare_scalar_function,
)
from lbfgsb.subspacemin import get_freev, subspace_minimization
from lbfgsb.types import NDArrayFloat


def _asnumpy(X):
    """Convert CuPy array to NumPy array if needed.

    CuPy arrays cannot be implicitly converted to NumPy arrays via np.array().
    Instead, use .get() to explicitly convert CuPy arrays to NumPy arrays.

    For deques (which store CuPy arrays), convert each element individually.

    AI Disclosure:
        this function was largely written by qwen3.5 while debugging the cupy backend
    """
    from collections import deque as Deque

    # If X is a deque, convert each element to ensure it's a NumPy array
    if isinstance(X, Deque):
        # Convert deque of CuPy arrays to NumPy array
        # First, convert each CuPy array element to NumPy
        X_list = [x.get() if hasattr(x, "get") else x for x in X]
        # Convert list of NumPy arrays to NumPy array
        return tnp.asarray(X_list)

    # For arrays, check if it's a CuPy array and convert
    # CuPy arrays have .get() method to convert to NumPy
    if hasattr(X, "get"):
        return X.get()
    # If not a CuPy array or NumPy array, assume it's already NumPy
    return X


@dataclass
class InternalState:
    """Class to keep track of internal state."""

    # keep track of some values (best, init)
    nit: int = 0
    status: str = "IDLE."
    task_str: str = "START"
    is_success: bool = False
    warnflag: int = 2


def minimize_lbfgsb(
    *,
    x0: NDArrayFloat,
    fun: Union[Objective, ObjectiveAndGradient],
    jac: JacOption,
    update_fun_def: Optional[
        Callable[
            [
                NDArrayFloat,
                float,
                float,
                NDArrayFloat,
                Deque[NDArrayFloat],
                Deque[NDArrayFloat],
            ],
            Tuple[float, float, NDArrayFloat, Deque[NDArrayFloat]],
        ]
    ] = None,
    bounds: Optional[ArrayLike] = None,
    checkpoint: Optional[OptimizeResult] = None,
    maxcor: int = 10,
    ftarget: Optional[Union[float, Callable[[], float]]] = None,
    ftol: float = 1e-5,
    gtol: Union[float, Callable[[], float]] = 1e-5,
    maxiter: int = 50,
    eps: float = 1e-8,
    maxfun: int = 15000,
    callback: Optional[Callable[[NDArrayFloat, OptimizeResult], bool]] = None,
    maxls: int = 20,
    finite_diff_rel_step: Optional[float] = None,
    max_steplength: float = 1e8,
    ftol_linesearch: float = 1e-3,
    gtol_linesearch: float = 0.9,
    xtol_linesearch: float = 1e-1,
    eps_SY: float = 2.2e-16,
    iprint: int = -1,
    logger: Optional[logging.Logger] = None,
    is_check_factorizations: bool = False,
    is_use_numba_jit: bool = True,
) -> OptimizeResult:
    r"""
    Minimize a scalar function of one or more variables using the L-BFGS-B algorithm.

    This routine solves bound-constrained optimization problems using a pure Python
    implementation of the limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm
    with bound constraints, commonly known as L-BFGS-B.

    The problem solved is

    .. math::

        \min_x f(x)

    subject to

    .. math::

        l_i \leq x_i \leq u_i,\quad i = 1, \ldots, n.

    The algorithm stores only a limited number of correction pairs to approximate the
    inverse Hessian, making it suitable for large-scale problems.

    Parameters
    ----------
    x0 : ndarray, shape (n,)
        Initial guess. The array contains the starting values of the ``n``
        independent variables. If bounds are provided, ``x0`` is clipped to the
        feasible box before the first evaluation.

    fun : callable
        Objective function to minimize.

        If ``jac`` is ``None``, ``False``, a finite-difference scheme, or a callable,
        ``fun`` must return only the scalar objective value:

        .. code-block:: python

            fun(x) -> float

        If ``jac`` is ``True``, ``fun`` must return both the objective value and the
        gradient:

        .. code-block:: python

            fun(x) -> tuple[float, ndarray]

        In all cases, ``x`` is a one-dimensional array with shape ``(n,)``. If
        additional arguments are required by the objective function, use a closure,
        ``functools.partial``, or another wrapper.

    jac : {callable, bool, '2-point', '3-point', 'cs'}, optional
        Method used to compute the gradient vector.

        If ``jac`` is a callable, it must return the gradient vector:

        .. code-block:: python

            jac(x) -> ndarray, shape (n,)

        If ``jac`` is ``True``, ``fun`` is assumed to return a pair ``(f, g)``, where
        ``f`` is the scalar objective value and ``g`` is the gradient vector. This is
        compatible with the convention used by :func:`scipy.optimize.minimize`.

        If ``jac`` is one of ``'2-point'``, ``'3-point'``, or ``'cs'``, the gradient
        is approximated numerically using the corresponding finite-difference scheme.
        These finite-difference schemes obey the specified bounds.

        If ``jac`` is ``None`` or ``False``, the gradient is approximated using
        forward finite differences with absolute step size ``eps``.

        Default is ``None``.

    update_fun_def : callable, optional
        Experimental callback used to update the objective-function definition during
        optimization and to modify the stored gradient sequence accordingly.

        This is primarily intended for problems where the objective function changes
        during optimization, for example regularized problems where the regularization
        weight is adapted on the fly. Since the L-BFGS approximation depends on past
        gradient differences, the stored gradient sequence must be updated to remain
        consistent with the new objective definition.

        The callable must have signature:

        .. code-block:: python

            update_fun_def(
                x,
                f0,
                f0_old,
                grad,
                x_deque,
                grad_deque,
            ) -> tuple[float, float, ndarray, deque[ndarray]]

        where ``x`` is the current point, ``f0`` is the current objective value,
        ``f0_old`` is the previous objective value, ``grad`` is the current gradient,
        ``x_deque`` is the stored sequence of past iterates, and ``grad_deque`` is the
        stored sequence of past gradients.

    bounds : sequence or scipy.optimize.Bounds, optional
        Bounds on the variables.

        Bounds may be specified in one of two forms:

        1. An instance of :class:`scipy.optimize.Bounds`.
        2. A sequence of ``(min, max)`` pairs for each variable.

        Use ``None`` for one side of a bound to indicate no bound in that direction.

    checkpoint : scipy.optimize.OptimizeResult, optional
        Previous optimization result used to restart the solver.

        When provided, the solver restarts from the checkpoint without discarding the
        stored L-BFGS correction pairs. The previous objective value, gradient, and
        inverse-Hessian approximation are reused. The objective-function definition
        must be unchanged between the original run and the restarted run.

        This is useful when an optimization was stopped early, when stopping criteria
        need to be modified, or when parameters such as ``maxcor`` or ``ftol`` need to
        be changed between runs.

        .. versionadded:: 1.0

    maxcor : int, optional
        Maximum number of variable-metric corrections used to approximate the inverse
        Hessian. The L-BFGS-B algorithm does not store the full Hessian matrix; it
        stores at most ``maxcor`` correction pairs. Default is ``10``.

    ftarget : float or callable, optional
        Target objective-function value.

        The optimization stops when

        .. math::

            f^{k+1} \leq f_{\mathrm{target}}.

        If ``ftarget`` is callable, it is called once after the first objective and
        gradient evaluation. This makes it possible to define a target value based on
        the initial objective value, for example when scaling is applied.

        If ``None``, this stopping criterion is disabled. Default is ``None``.

    ftol : float, optional
        Relative objective-function decrease tolerance.

        The optimization stops when

        .. math::

            \frac{f^k - f^{k+1}}
                {\max(|f^k|, |f^{k+1}|, 1)}
            \leq \mathrm{ftol}.

        In the original Fortran implementation, this corresponds to
        ``factr * epsmch``. Typical values are:

        - ``5e-3`` for low accuracy,
        - ``5e-8`` for moderate accuracy,
        - ``5e-14`` for high accuracy.

        If ``ftol`` is ``0``, the test stops only when the objective function remains
        unchanged after one iteration. Default is ``1e-5``.

    gtol : float or callable, optional
        Projected-gradient tolerance.

        The optimization stops when

        .. math::

            \max_i |\operatorname{proj} g_i| \leq \mathrm{gtol},

        where ``proj g`` is the projected gradient. If ``gtol`` is callable, it is
        called once after the first objective and gradient evaluation. This allows the
        stopping tolerance to depend on the initial objective value. Default is
        ``1e-5``.

    eps : float, optional
        Absolute step size used for forward finite-difference approximation of the
        gradient when ``jac`` is ``None`` or ``False``. Default is ``1e-8``.

    maxfun : int, optional
        Maximum number of objective-function evaluations.

        The limit may be exceeded when gradients are estimated by numerical
        differentiation. Interruptions due to ``maxfun`` are postponed until the end
        of the current minimization iteration. Default is ``15000``.

    maxiter : int, optional
        Maximum number of iterations. Default is ``50``.

    callback : callable, optional
        Callback called after each iteration.

        The callable must have signature:

        .. code-block:: python

            callback(xk, state) -> bool

        where ``xk`` is the current parameter vector and ``state`` is an
        :class:`scipy.optimize.OptimizeResult` containing the current solver state.

        If the callback returns ``True``, the optimization stops.

    maxls : int, optional
        Maximum number of line-search steps per iteration. Default is ``20``.

    finite_diff_rel_step : float, optional
        Relative step size used for numerical gradient approximation when ``jac`` is
        ``'2-point'``, ``'3-point'``, or ``'cs'``.

        The absolute step size is computed as

        .. math::

            h = \mathrm{rel\_step} \operatorname{sign}(x) \max(1, |x|),

        and is adjusted if necessary to respect the bounds. For ``jac='3-point'``,
        the sign of ``h`` is ignored. If ``None``, the step size is selected
        automatically. Default is ``None``.

    max_steplength : float, optional
        Maximum allowed step length during the line search. Default is ``1e8``.

    ftol_linesearch : float, optional
        Tolerance for the sufficient-decrease condition in the line search.

        This is the Wolfe condition parameter :math:`c_1` in

        .. math::

            f(x_k + \alpha_k p_k)
            \leq
            f(x_k) + c_1 \alpha_k p_k^\mathrm{T} \nabla f(x_k).

        Usually, :math:`c_1` is small and satisfies
        :math:`0 < c_1 < 1`. In the original L-BFGS-B Fortran implementation, this
        parameter is hard-coded to ``1e-3``. Default is ``1e-3``.

    gtol_linesearch : float, optional
        Tolerance for the curvature condition in the line search.

        This is the Wolfe condition parameter :math:`c_2` in

        .. math::

            \left|
            p_k^\mathrm{T} \nabla f(x_k + \alpha_k p_k)
            \right|
            \leq
            c_2
            \left|
            p_k^\mathrm{T} \nabla f(x_k)
            \right|.

        The parameters should satisfy :math:`0 < c_1 < c_2 < 1`. Usually,
        :math:`c_2` is much larger than :math:`c_1`. In the original L-BFGS-B
        Fortran implementation, this parameter is hard-coded to ``0.9``. Default is
        ``0.9``.

    xtol_linesearch : float, optional
        Relative tolerance for acceptable steps in the line-search procedure. In the
        original L-BFGS-B Fortran implementation, this parameter is hard-coded to
        ``0.1``. Default is ``0.1``.

    eps_SY : float, optional
        Numerical threshold used when updating the L-BFGS correction matrices.
        Default is ``2.2e-16``.

    iprint : int, optional
        Verbosity level.

        - ``iprint < 0``: no output.
        - ``iprint == 0``: print only the final summary.
        - ``0 < iprint < 99``: print objective value and projected-gradient norm
        every ``iprint`` iterations.
        - ``iprint >= 99``: print detailed information at every iteration, except
        full vectors.

        Default is ``-1``.

    logger : logging.Logger, optional
        Logger used for textual output. If ``None``, no output is emitted regardless
        of ``iprint``. Default is ``None``.

    is_check_factorizations : bool, optional
        Whether to run additional consistency checks on matrix factorizations.
        Intended for development and debugging only. Default is ``False``.

    is_use_numba_jit : bool, optional
        Whether to use ``numba`` just-in-time compilation for performance-critical
        parts of the algorithm.

        If ``True`` but ``numba`` is unavailable, a warning is raised and the solver
        falls back to the pure Python implementation. The expected speed-up for large
        problems is typically between two and five times. Default is ``True``.

        .. versionadded:: 1.0

    Returns
    -------
    scipy.optimize.OptimizeResult
        Optimization result represented as an :class:`scipy.optimize.OptimizeResult`.

        Important fields include:

        ``x`` : ndarray
            Final solution.
        ``fun`` : float
            Objective value at the final solution.
        ``jac`` : ndarray
            Gradient at the final solution.
        ``nfev`` : int
            Number of objective-function evaluations.
        ``njev`` : int
            Number of gradient evaluations.
        ``nit`` : int
            Number of iterations.
        ``status`` : int
            Solver status code.
        ``message`` : str
            Termination message.
        ``success`` : bool
            Whether the solver terminated successfully.
        ``hess_inv`` : scipy.optimize.LbfgsInvHessProduct
            Limited-memory inverse-Hessian approximation.

    Raises
    ------
    ValueError
        Raised when the provided checkpoint is inconsistent with ``x0`` or when
        restored correction vectors have incompatible dimensions.

    Notes
    -----
    This implementation is a Python reimplementation of the L-BFGS-B algorithm,
    rather than a wrapper around the original Fortran code. It exposes several
    parameters that are fixed in the historical implementation, including line-search
    Wolfe-condition parameters.

    The convention ``jac=True`` follows :func:`scipy.optimize.minimize`: in that
    case, ``fun`` must return both the objective value and the gradient. For example:

    .. code-block:: python

        def fun(x):
            f = x[0] ** 2 + x[1] ** 2
            g = np.array([2 * x[0], 2 * x[1]])
            return f, g

        res = minimize_lbfgsb(x0=x0, fun=fun, jac=True)

    If ``jac`` is ``None`` or ``False``, finite differences are used.

    Examples
    --------
    Minimize a quadratic function with a separate gradient function:

    .. code-block:: python

        import numpy as np
        from lbfgsb import minimize_lbfgsb

        def objective(x):
            return x[0] ** 2 + x[1] ** 2

        def gradient(x):
            return np.array([2 * x[0], 2 * x[1]])

        x0 = np.array([10.0, 10.0])

        res = minimize_lbfgsb(
            x0=x0,
            fun=objective,
            jac=gradient,
        )

        print(res.x)
        print(res.fun)

    Use the SciPy-compatible ``jac=True`` convention:

    .. code-block:: python

        import numpy as np
        from lbfgsb import minimize_lbfgsb

        def objective_and_gradient(x):
            f = x[0] ** 2 + x[1] ** 2
            g = np.array([2 * x[0], 2 * x[1]])
            return f, g

        x0 = np.array([10.0, 10.0])

        res = minimize_lbfgsb(
            x0=x0,
            fun=objective_and_gradient,
            jac=True,
        )

        print(res.x)
        print(res.fun)

    Use finite differences with bounds:

    .. code-block:: python

        import numpy as np
        from lbfgsb import minimize_lbfgsb

        def objective(x):
            return x[0] ** 2 + x[1] ** 2

        x0 = np.array([10.0, 10.0])
        bounds = [(0.0, None), (0.0, None)]

        res = minimize_lbfgsb(
            x0=x0,
            fun=objective,
            jac="2-point",
            bounds=bounds,
        )

    References
    ----------
    .. [1] R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
        Constrained Optimization. SIAM Journal on Scientific and Statistical
        Computing, 16(5), pp. 1190-1208, 1995.

    .. [2] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778:
        L-BFGS-B, FORTRAN routines for large scale bound constrained
        optimization. ACM Transactions on Mathematical Software, 23(4),
        pp. 550-560, 1997.

    .. [3] J. L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778:
        L-BFGS-B, FORTRAN routines for large scale bound constrained
        optimization. ACM Transactions on Mathematical Software, 38(1), 2011.
    """
    if not NUMBA_AVAILABLE and is_use_numba_jit:
        warnings.warn(
            "`is_use_numba_jit` set to `True` (default) but numba cannot be imported!"
            " Falling back to pure python (slower)."
        )
        is_use_numba_jit = False

    lb, ub = get_bounds(x0, bounds)
    max_steplength_user: float = copy.copy(max_steplength)

    # True if all values have lower and upper bounds
    is_boxed: bool = not is_any_inf([lb, ub])

    # applying the bounds to the initial guess x0
    n = x0.size
    x = clip2bounds(x0, lb, ub)

    # Some display about the problem at hand. The display depends on the value of iprint
    display_start(
        np.finfo(float).eps, n, maxcor, count_var_at_bounds(x, lb, ub), iprint
    )

    X, G = initialize_X_and_G(x, checkpoint, maxcor)

    # Initialization of the matrices
    mats = LBFGSB_MATRICES(n)

    # wrapper storing the calls to f and g and handling finite difference approximation
    sf: ScalarFunction = prepare_scalar_function(
        fun,
        x,
        jac=jac,
        epsilon=eps,
        bounds=(lb, ub),
        finite_diff_rel_step=finite_diff_rel_step,
    )

    # restore the number of iterations and objective function evaluation
    if checkpoint is not None:
        sf.nfev = checkpoint.nfev
        sf.ngev = checkpoint.njev

    # First evaluation of the objective function if no checkpoint provided
    if checkpoint is None:
        f0 = sf.fun(x)
    else:
        f0 = checkpoint.fun

    # potential update of stop criterion
    if ftarget is not None:
        try:
            _ftarget: Optional[float] = ftarget()  # type: ignore
        except TypeError:
            _ftarget = ftarget  # type: ignore
    else:
        _ftarget = None

    try:
        _gtol: float = gtol()  # type: ignore
    except TypeError:
        _gtol = gtol  # type: ignore

    # Create an internal state instance
    istate = InternalState()

    if checkpoint is not None:
        istate.nit = checkpoint.nit

    # early check of stop criterion -> Extreme case in which x0 satisfies the
    # criterion, then no optimization is needed and one does not need to compute
    # anything else.
    if is_f0_target_reached(f0, _ftarget, istate):
        # leave the optimization routine
        if checkpoint is None:
            if len(X) == 0:
                X.append(x)
                G.append(np.zeros_like(x))
            return OptimizeResult(
                fun=f0,
                jac=G[0],
                nfev=sf.nfev,
                njev=sf.ngev,
                nit=istate.nit,
                status=istate.warnflag,
                message=istate.task_str,
                x=x,
                success=istate.is_success,
                hess_inv=LbfgsInvHessProduct(
                    np.diff(np.array(X), axis=0), np.diff(np.array(G), axis=0)
                ),
            )
        else:
            return checkpoint

    # Compute the first gradient if no checkpoint provided
    if checkpoint is None:
        grad = sf.grad(x)
    else:
        grad = checkpoint.jac

    # Note, no need to further update anything because the scaling is handled by the
    # ScalarFunction instance

    # perform an early potential update of the objective function definition and
    # upgrade the gradient and the past sequence of gradients accordingly
    if update_fun_def is not None:
        f0, f0_old, grad, G = update_fun_def(x, f0, copy.copy(f0), grad, X, G)

    if len(X) > 0:
        # only happens if checkpoint is provided (L-BFGS-B restart)
        mats = update_lbfgs_matrices(
            x.copy(),  # copy otherwise x might be changed in X when updated
            grad,
            X,
            G,
            maxcor,
            mats,
            is_force_update=True,
            eps=eps_SY,
            is_check_factorization=is_check_factorizations,
            is_use_numba_jit=is_use_numba_jit,
        )
    else:
        # Store first res to X and G
        X.append(np.copy(x))
        G.append(grad)

    # For now the free variables at the cauchy points is an empty set
    free_vars = np.array([], dtype=np.int_)

    # Check the infinity norm of the projected gradient
    sbgnrm = projgr(x, grad, lb, ub)
    display_iter(istate.nit, sbgnrm, f0, iprint, logger=logger)

    # bool indicating if results of the current iteration have been displayed. It
    # avoid duplicates for the last round.
    has_displayed_results = False

    # Note that interruptions due to maxfun are postponed
    # until the completion of the current minimization iteration.
    while (
        projgr(x, grad, lb, ub) > _gtol
        and istate.nit < maxiter
        and sf.nfev < maxfun
        and not istate.is_success
    ):
        if iprint > 99 and logger is not None:
            logger.info("\n")
            logger.info(f"ITERATION {istate.nit + 1}\n")

        f0_old = f0 + 0.0  # copy

        # find cauchy point
        x_cp, c = get_cauchy_point(
            x, grad, lb, ub, mats, iprint, logger, is_use_numba_jit=is_use_numba_jit
        )

        # Get the free variables for the GCP
        # Note: no numba needed here
        free_vars, active_vars = get_freev(
            x_cp, lb, ub, istate.nit, free_vars, iprint, logger
        )

        # subspace minimization: find the search direction for the minimization problem
        xbar: NDArrayFloat = subspace_minimization(
            x,
            x_cp,
            free_vars,
            active_vars,
            c,
            grad,
            lb,
            ub,
            mats,
            is_check_factorizations=is_check_factorizations,
            is_use_numba_jit=is_use_numba_jit,
        )
        d = xbar - x

        steplength = line_search(
            x,
            f0,
            grad,
            d,
            lb,
            ub,
            istate.nit,
            max_steplength_user,
            is_boxed,
            sf,
            ftol_linesearch,
            gtol_linesearch,
            xtol_linesearch,
            # The maximum number of function evaluation in linesearch must take into
            # account maxfun and the number of call already performed.
            min(maxls, maxfun - sf.nfev),
            iprint,
            logger,
            is_use_numba_jit=is_use_numba_jit,
        )
        if steplength is None:
            if len(X) == 1:
                # Hessian already rebooted: abort.
                istate.task_str = "ABNORMAL_TERMINATION_IN_LNSRCH"
                istate.warnflag = 2
                istate.is_success = False
                break  # leave the while and finish the program.
            else:
                istate.task_str = "RESTART_FROM_LNSRCH"
                # Keep only the last correction
                X = deque([X[-1]])
                G = deque([G[-1]])
                # Reboot BFGS-Hessian
                mats = LBFGSB_MATRICES(n)
        else:
            # x update
            x += steplength * d

            # new evaluation -> normally, the function has been updated in
            # the linesearch step
            f0, grad = sf.fun_and_grad(x)

            if update_fun_def is None:
                if is_f0_target_reached(f0, _ftarget, istate):
                    break  # the while loop
                elif is_f0_min_change_reached(f0, f0_old, ftol, istate):
                    break  # the while loop

            # perform a potential update of the objective function definition and
            # upgrade the gradient and the past sequence of gradients accordingly
            else:
                f0, f0_old, grad, G = update_fun_def(x, f0, f0_old, grad, X, G)

                # Check stop criterion: minimum relative change in the
                # objective function
                if is_f0_min_change_reached(f0, f0_old, ftol, istate):
                    break  # the while loop

                # Check stop criterion: minimum objective function value
                elif is_f0_target_reached(f0, _ftarget, istate):
                    break  # the while loop

                # We must check if the updated G satisfy the strong wolfe condition
                X, G = make_X_and_G_respect_strong_wolfe(
                    X, G, eps_SY, logger=logger, is_use_numba_jit=is_use_numba_jit
                )

            mats = update_lbfgs_matrices(
                x.copy(),  # copy otherwise x might be changed in X when updated
                grad,
                X,
                G,
                maxcor,
                mats,
                is_force_update=False,
                eps=eps_SY,
                is_check_factorization=is_check_factorizations,
                is_use_numba_jit=is_use_numba_jit,
            )

            # callback is a user defined mechanism to stop optimization
            # if callback returns True, then it stops.
            # Note: no need to callback if an other stop criterion has already been
            # reached (istate.is_success)
            if callback is not None and not istate.is_success:
                if callback(
                    np.copy(x),
                    OptimizeResult(
                        fun=f0,
                        jac=grad,
                        nfev=sf.nfev,
                        njev=sf.ngev,
                        nit=istate.nit,
                        status=istate.warnflag,
                        message=istate.task_str,
                        x=x,
                        success=istate.is_success,
                        hess_inv=LbfgsInvHessProduct(
                            np.atleast_2d(np.diff(_asnumpy(X), axis=0)),
                            np.atleast_2d(np.diff(_asnumpy(G), axis=0)),
                        ),
                    ),
                ):
                    istate.task_str = "STOP: USER CALLBACK"
                    istate.is_success = True

        display_iter(istate.nit + 1, projgr(x, grad, lb, ub), f0, iprint, logger)

        # Result display
        has_displayed_results = display_results(
            istate.nit + 1, maxiter, x, grad, lb, ub, f0, _gtol, False, iprint, logger
        )

        istate.nit += 1

    # Final display. If is_success, then it already happened
    if not has_displayed_results:
        display_results(
            istate.nit, maxiter, x, grad, lb, ub, f0, _gtol, True, iprint, logger
        )

    if projgr(x, grad, lb, ub) <= _gtol:
        istate.task_str = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL"
        istate.is_success = True
        istate.warnflag = 1
    elif istate.nit == maxiter:
        istate.task_str = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
        istate.is_success = True
        istate.warnflag = 1
    elif sf.nfev >= maxfun:
        istate.task_str = "STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT"
        istate.is_success = True
        istate.warnflag = 1

    # error: b'ERROR: STPMAX .LT. STPMIN'

    return OptimizeResult(
        fun=f0,
        jac=grad,
        nfev=sf.nfev,
        njev=sf.ngev,
        nit=istate.nit,
        status=istate.warnflag,
        message=istate.task_str,
        x=x,
        success=istate.is_success,
        hess_inv=LbfgsInvHessProduct(
            np.atleast_2d(np.diff(_asnumpy(X), axis=0)),
            np.atleast_2d(np.diff(_asnumpy(G), axis=0)),
        ),
    )


def initialize_X_and_G(
    x: NDArrayFloat, checkpoint: Optional[OptimizeResult], maxcor: int
) -> Tuple[Deque[NDArrayFloat], Deque[NDArrayFloat]]:
    """
    Initialize the sequence of adjusted values and associated gradients.

    This routine is mainly dedicated to restore the sequence from a checkpoint.
    The sequence is stored in the `hess_inv` attribute as "sk" and "yk" which
    are differences, e.g., sk = np.atleast_2d(np.diff(np.array(X), axis=0)).

    Parameters
    ----------
    x : NDArrayFloat
        Adjusted values.
    checkpoint : Optional[OptimizeResult]
        Optional checkpoint (see solver restart).
    maxcor : int
        Maximum number of corrections stored.

    Returns
    -------
    Tuple[Deque[NDArrayFloat], Deque[NDArrayFloat]]
        X and G.
    """
    # Initialize X and G
    # Deque = similar to list but with faster operations to remove and add
    # values to extremities
    X: Deque[NDArrayFloat] = deque()
    G: Deque[NDArrayFloat] = deque()

    # if it is a L-BFGS-B restart (checkpoint is provided), then X and G are restored
    # from x, jac and the sequence of differences sk and yk stored in the inverse
    # Hessian approximation instance (LbfgsInvHessProduct).
    if checkpoint is None:
        return X, G

    # x0 and checkpoint.x should be the same otherwisee there is an issue
    try:
        np.testing.assert_equal(x, checkpoint.x)
    except AssertionError as e:
        raise ValueError(
            "When 'checkpoint' is provided (L-BFGS-B restart), x0 and checkpoint.x"
            " should be equal!"
        ) from e
    n_corrs, n = checkpoint.hess_inv.sk.shape
    if n_corrs == 0:
        return X, G

    if n != x.size:
        raise ValueError(
            f"The size of correction vector ({n}) does"
            f" not match the size of x ({x.size})!"
        )
    # restore the past X and G
    for x, g in zip(
        checkpoint.x - np.cumsum(checkpoint.hess_inv.sk, axis=0),
        checkpoint.jac - np.cumsum(checkpoint.hess_inv.yk, axis=0),
    ):
        if len(X) > maxcor:
            X.popleft()
            G.popleft()
        X.append(x)
        G.append(g)
    # at this point, X and G do not have x nor jac -> it is added a bit later
    return X, G


def is_f0_min_change_reached(
    f0: float, f0_old: float, ftol: float, istate: InternalState
) -> bool:
    """
    Return whether the minimum obj fun change has been reached.

    It updates the internal state.
    """
    if (f0_old - f0) / max(abs(f0_old), abs(f0), 1) < ftol:
        istate.task_str = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL"
        istate.is_success = True
        istate.warnflag = 0
        return True
    return False


def is_f0_target_reached(
    f0: float, ftarget: Optional[float], istate: InternalState
) -> bool:
    """
    Return whether the obj fun target has been reached.

    It updates the internal state.
    """
    if ftarget is None:
        return False
    if f0 > ftarget:
        return False
    istate.task_str = "CONVERGENCE: F_<=_TARGET"
    istate.is_success = True
    istate.warnflag = 0
    return True
