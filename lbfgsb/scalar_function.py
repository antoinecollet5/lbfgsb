import numpy as np
from scipy.optimize._numdiff import approx_derivative

FD_METHODS = ("2-point", "3-point", "cs")


class ScalarFunction:
    """Scalar function and its derivatives.

    This class defines a scalar function F: R^n->R and methods for
    computing or approximating its first and second derivatives.

    Parameters
    ----------
    fun : callable
        evaluates the scalar function. Must be of the form ``fun(x, *args)``,
        where ``x`` is the argument in the form of a 1-D array and ``args`` is
        a tuple of any additional fixed parameters needed to completely specify
        the function. Should return a scalar.
    x0 : array-like
        Provides an initial set of variables for evaluating fun. Array of real
        elements of size (n,), where 'n' is the number of independent
        variables.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify the scalar
        function.
    grad : {callable, '2-point', '3-point', 'cs'}
        Method for computing the gradient vector.
        If it is a callable, it should be a function that returns the gradient
        vector:

            ``grad(x, *args) -> array_like, shape (n,)``

        where ``x`` is an array with shape (n,) and ``args`` is a tuple with
        the fixed parameters.
        Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
        to select a finite difference scheme for numerical estimation of the
        gradient with a relative step size. These finite difference schemes
        obey any specified `bounds`.
    finite_diff_rel_step : None or array_like
        Relative step size to use. The absolute step size is computed as
        ``h = finite_diff_rel_step * sign(x0) * max(1, abs(x0))``, possibly
        adjusted to fit into the bounds. For ``method='3-point'`` the sign
        of `h` is ignored. If None then finite_diff_rel_step is selected
        automatically,
    finite_diff_bounds : tuple of array_like
        Lower and upper bounds on independent variables. Defaults to no bounds,
        (-np.inf, np.inf). Each bound must match the size of `x0` or be a
        scalar, in the latter case the bound will be the same for all
        variables. Use it to limit the range of function evaluation.
    epsilon : None or array_like, optional
        Absolute step size to use, possibly adjusted to fit into the bounds.
        For ``method='3-point'`` the sign of `epsilon` is ignored. By default
        relative steps are used, only if ``epsilon is not None`` are absolute
        steps used.

    Notes
    -----
    This class implements a memoization logic. There are methods `fun`,
    `grad`, and corresponding attributes `f`, `g`. The following
    things should be considered:

        1. Use only public methods `fun` and `grad`.
        2. After one of the methods is called, the corresponding attribute
           will be set. However, a subsequent call with a different argument
           of *any* of the methods may overwrite the attribute.
    """

    def __init__(
        self,
        fun,
        x0,
        args,
        grad,
        finite_diff_rel_step,
        finite_diff_bounds,
        epsilon=None,
    ):
        if not callable(grad) and grad not in FD_METHODS:
            raise ValueError(f"`grad` must be either callable or one of {FD_METHODS}.")

        # the astype call ensures that self.x is a copy of x0
        self.x = np.atleast_1d(x0).astype(float)
        self.n = self.x.size
        self.nfev = 0
        self.ngev = 0
        self.nhev = 0
        self.f_updated = False
        self.g_updated = False
        self.H_updated = False

        self._lowest_x = None
        self._lowest_f = np.inf

        finite_diff_options = {}
        if grad in FD_METHODS:
            finite_diff_options["method"] = grad
            finite_diff_options["rel_step"] = finite_diff_rel_step
            finite_diff_options["abs_step"] = epsilon
            finite_diff_options["bounds"] = finite_diff_bounds

        # Function evaluation
        def fun_wrapped(x):
            self.nfev += 1
            # Send a copy because the user may overwrite it.
            # Overwriting results in undefined behaviour because
            # fun(self.x) will change self.x, with the two no longer linked.
            fx = fun(np.copy(x), *args)
            # Make sure the function returns a true scalar
            if not np.isscalar(fx):
                try:
                    fx = np.asarray(fx).item()
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        "The user-provided objective function "
                        "must return a scalar value."
                    ) from e

            if fx < self._lowest_f:
                self._lowest_x = x
                self._lowest_f = fx

            return fx

        def update_fun():
            self.f = fun_wrapped(self.x)

        self._update_fun_impl = update_fun

        # Gradient evaluation
        if callable(grad):

            def grad_wrapped(x):
                self.ngev += 1
                return np.atleast_1d(grad(np.copy(x), *args))

            def update_grad():
                self.g = grad_wrapped(self.x)

        elif grad in FD_METHODS:

            def update_grad():
                self._update_fun()
                self.ngev += 1
                self.g = approx_derivative(
                    fun_wrapped, self.x, f0=self.f, **finite_diff_options
                )

        self._update_grad_impl = update_grad

    def update_x(self, x) -> None:
        # ensure that self.x is a copy of x. Don't store a reference
        # otherwise the memoization doesn't work properly.
        self.x = np.atleast_1d(x).astype(float)
        self.f_updated = False
        self.g_updated = False
        self.H_updated = False

    def _update_fun(self) -> None:
        if not self.f_updated:
            self._update_fun_impl()
            self.f_updated = True

    def _update_grad(self) -> None:
        if not self.g_updated:
            self._update_grad_impl()
            self.g_updated = True

    def fun(self, x) -> float:
        if not np.array_equal(x, self.x):
            self.update_x(x)
        self._update_fun()
        return self.f

    def grad(self, x):
        if not np.array_equal(x, self.x):
            self.update_x(x)
        self._update_grad()
        return self.g

    def fun_and_grad(self, x):
        if not np.array_equal(x, self.x):
            self.update_x(x)
        self._update_fun()
        self._update_grad()
        return self.f, self.g


def prepare_scalar_function(
    fun,
    x0,
    jac=None,
    args=(),
    bounds=None,
    epsilon=None,
    finite_diff_rel_step=None,
) -> ScalarFunction:
    """
    Creates a ScalarFunction object for use with scalar minimizers
    (BFGS/LBFGSB/SLSQP/TNC/CG/etc).

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    jac : {callable,  '2-point', '3-point', 'cs', None}, optional
        Method for computing the gradient vector. If it is a callable, it
        should be a function that returns the gradient vector:

            ``jac(x, *args) -> array_like, shape (n,)``

        If one of `{'2-point', '3-point', 'cs'}` is selected then the gradient
        is calculated with a relative step for finite differences. If `None`,
        then two-point finite differences with an absolute step is used.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` functions).
    bounds : sequence, optional
        Bounds on variables. 'new-style' bounds are required.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.

    Returns
    -------
    sf : ScalarFunction
    """
    if callable(jac):
        grad = jac
    elif jac in FD_METHODS:
        # epsilon is set to None so that ScalarFunction is made to use
        # rel_step
        epsilon = None
        grad = jac
    else:
        # default (jac is None) is to do 2-point finite differences with
        # absolute step size. ScalarFunction has to be provided an
        # epsilon value that is not None to use absolute steps. This is
        # normally the case from most _minimize* methods.
        grad = "2-point"
        epsilon = epsilon

    if bounds is None:
        bounds = (-np.inf, np.inf)

    # ScalarFunction caches. Reuse of fun(x) during grad
    # calculation reduces overall function evaluations.
    sf = ScalarFunction(
        fun, x0, args, grad, finite_diff_rel_step, bounds, epsilon=epsilon
    )

    return sf
