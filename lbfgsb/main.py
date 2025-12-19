"""
@author: Antoine COLLET.

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
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple, Union

import numpy as np
from scipy.optimize import (
    LbfgsInvHessProduct,  # noqa : F401
    OptimizeResult,
)
from typing_extensions import Protocol  # support python 3.7

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
from lbfgsb.scalar_function import ScalarFunction, prepare_scalar_function
from lbfgsb.subspacemin import get_freev, subspace_minimization
from lbfgsb.types import NDArrayFloat


class ObjectiveFunction(Protocol):
    """Protocol for objective function signature."""

    def __call__(self, __x, *args, **kwargs) -> float: ...


class GradientFunction(Protocol):
    """Protocol for gradient signature."""

    def __call__(self, __x, *args, **kwargs) -> NDArrayFloat: ...


@dataclass
class InternalState:
    """Class to keep track of internal state."""

    # keep track of some values (best, init)
    nit = 0
    status = "IDLE."
    task_str = "START"
    is_success = False
    warnflag = 2


def minimize_lbfgsb(
    *,
    x0: NDArrayFloat,
    fun: Optional[ObjectiveFunction] = None,
    args: Tuple = (),
    jac: Optional[Union[GradientFunction, str, bool]] = None,
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
    bounds: Optional[NDArrayFloat] = None,
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
    is_check_factorization: bool = False,
) -> OptimizeResult:
    r"""
    Solves bound constrained optimization problems by using the compact formula
    of the limited memory BFGS updates.

    fun :  Optional[Callable[[NDArrayFloat, Tuple[Any]], float]],
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is a 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function. Mandatory if `fun_and_jax` is not specified. The default
        is None.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where ``n`` is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
    jac : {callable,  '2-point', '3-point', 'cs', bool}, optional
        Method for computing the gradient vector.
        If it is a callable, it should be a function that returns the gradient
        vector:

            ``jac(x, *args) -> array_like, shape (n,)``

        where ``x`` is an array with shape (n,) and ``args`` is a tuple with
        the fixed parameters. If `jac` is a Boolean and is True, `fun` is
        assumed to return a tuple ``(f, g)`` containing the objective
        function and the gradient.
        If None or False, the gradient will be estimated using 2-point finite
        difference estimation with an absolute step size.
        Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
        to select a finite difference scheme for numerical estimation of the
        gradient with a relative step size. These finite difference schemes
        obey any specified `bounds`.
    update_fun_def: Optional[Callable]
        Function to update the gradient sequence. This is an experimental feature to
        allow changing the objective function definition on the fly. In the first place
        this functionality is dedicated to regularized problems for which the
        regularization weight is computed while optimizing the cost function. In order
        to get a hessian matching the new definition of `fun`, the gradient sequence
        must be updated.

            ``update_fun_def(x, f0, f0_old, grad, x_deque, grad_deque)
            -> f0, f0_old, grad, updated grad_deque``

    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    checkpoint: Optional[OptimizeResult]
        OptimizeResult instance. This parameter allow to pass the output of a previous
        `minimize_lbfgsb` run (a 'checkpoint') and restart the solver without losing
        the sequence of adjusted values and associated gradients (hence the
        approximation of the inverse Hessian). The last objective function and
        associated gradient is also used. Of course the objective function definition
        must remain the same between the two optimization rounds.
        It can be useful if the optimization has been stopped too early,
        if some stop criteria or other parameters must be changed (e.g., `maxcor`
        or `ftol`) or if some scaling must be performed before starting L-BFGS-B. It
        avoids recalculating some expensive objective functions and gradients.
        This is a unique feature among L-BFGS-B implementations. The default is None.
    maxcor : int
        The maximum number of variable metric corrections used to
        define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms
        in an approximation to it.)
    ftarget: Optional[Union[float, Callable]] = None
        Target objective function (stop criterion) .
        The iteration stops when ``f^{k+1} <= fmin``.
        If Callable, it is called only once after the first function and gradient
        computation. This option is available so that the stop criteria could be
        adatped based on the first objective function value, i.e.,
        this is particularly useful if a scaling is applied. If None, the stop criterion
        is ignored. The default is None.
    ftol : float
        Objective function minimum change (stop criterion). The iteration stops
        when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
        In the original Fortran algorithm, this corresponds to `factr * epsmch`.
        Typical values for `ftol` on a computer with 15 digits of accuracy in double
        precision are as follows: `ftol` = 5e-3 for low accuracy; `ftol` = 5e-8
        for moderate accuracy; `ftol` = 5e-14 for extremely high accuracy.
        If `ftol` = 0, the test will stop the algorithm only if the objective function
        remains unchanged after one iteration. The default is 1e-5.
    gtol : Union[float, Callable]
        Projected gradient mininmum value (stop criterion).
        The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
        <= gtol`` where ``pg_i`` is the i-th component of the
        projected gradient.
        As for gtol, if Callable, it is called only once after the first function
        and gradient computation. This option is available so that the stop criteria
        could be adatped based on the first objective function value, i.e.,
        this is particularly useful if a scaling is applied. The default is 1e-5.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    maxfun : int
        Maximum number of function evaluations. Note that this function
        may violate the limit because of evaluating gradients by numerical
        differentiation.
        Note that interruptions due to maxfun are postponed
        until the completion of a minimization iteration, consequently it might
        stop after maxfun has been reached.
    maxiter : int
        Maximum number of iterations.
    callback : Optional[Callable[[NDArrayFloat, OptimizeResult], bool]]
        Called after each iteration. It is a callable with
        the signature:

            ``callback(xk, OptimizeResult state) -> bool``

        where ``xk`` is the current parameter vector. and ``state``
        is an `OptimizeResult` object, with the same fields
        as the ones from the return. If callback returns True
        the algorithm execution is terminated.
    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    max_steplength: float
        Maximum steplength allowed. The default is 1e8.
    ftol_linesearch: float, optional
        Specify a nonnegative tolerance for the sufficient decrease condition in
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_
        (used for the line search). This is :math:`c_1` in
        the Armijo condition (or Goldstein, Goldstein-Armijo condition) where
        :math:`\alpha_{k}` is the estimated step.

        .. math::

            f(\mathbf{x}_{k}+\alpha_{k}\mathbf{p}_{k})\leq
            f(\mathbf{x}_{k})+c_{1}\alpha_{k}\mathbf{p}_{k}^{\mathrm{T}}
            \nabla f(\mathbf{x}_{k})

        Note that :math:`0 < c_1 < 1`. Usually :math:`c_1` is small, see the Wolfe
        conditions in :cite:t:`nocedalNumericalOptimization1999`.
        In the fortran implementation
        algo 778, it is hardcoded to 1e-3. The default is 1e-4.
    gtol_linesearch: float, optional
        Specify a nonnegative tolerance for the curvature condition in
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_
        (used for the line search). This is :math:`c_2` in
        the Armijo condition (or Goldstein, Goldstein-Armijo condition) where
        :math:`\alpha_{k}` is the estimated step.

        .. math::

            \left|\mathbf{p}_{k}^{\mathrm {T}}\nabla f(\mathbf{x}_{k}+\alpha_{k}
            \mathbf{p}_{k})\right|\leq c_{2}\left|\mathbf {p}_{k}^{\mathrm{T}}\nabla
            f(\mathbf{x}_{k})\right|

        Note that :math:`0 < c_1 < c_2 < 1`. Usually, :math:`c_2` is
        much larger than :math:`c_2`.
        see :cite:t:`nocedalNumericalOptimization1999`. In the fortran implementation
        algo 778, it is hardcoded to 0.9. The default is 0.9.
    xtol_linesearch: float, optional
        Specify a nonnegative relative tolerance for an acceptable step in the line
        search procedure (see
        `minpack2.dcsrch <https://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/dcsrch.f>`_).
        In the fortran implementation algo 778, it is hardcoded to 0.1.
        The default is 0.1.
        See :func:`line_search` parameters.
    eps_SY: float
        Parameter used for updating the L-BFGS matrices. The default is 2.2e-16.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99``   print details of every iteration except n-vectors;
    logger: Optional[Logger], optional
        :class:`logging.Logger` instance. If None, nothing is displayed, no matter the
        value of `iprint`, by default None.
    is_check_factorization: bool
        For development purposes only, leave to False. The default is False.

    Returns
    -------
    OptimizeResult
        Wrapper for optimization results (from scipy).

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing, 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (2011),
      ACM Transactions on Mathematical Software, 38, 1.
    """
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
        args=args,
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
            is_force_update=False,
            eps=eps_SY,
            is_check_factorization=is_check_factorization,
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

        f0_old = copy.copy(f0)

        # find cauchy point
        x_cp, c = get_cauchy_point(
            x,
            grad,
            lb,
            ub,
            mats,
            iprint,
            logger,
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
            is_check_factorizations=is_check_factorization,
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
                X = Deque([X[-1]])
                G = Deque([G[-1]])
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
                    X,
                    G,
                    eps_SY,
                    logger=logger,
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
                is_check_factorization=is_check_factorization,
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
                            np.atleast_2d(np.diff(np.array(X), axis=0)),
                            np.atleast_2d(np.diff(np.array(G), axis=0)),
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
            np.atleast_2d(np.diff(np.array(X), axis=0)),
            np.atleast_2d(np.diff(np.array(G), axis=0)),
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
