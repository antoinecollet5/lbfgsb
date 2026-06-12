# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union, cast

from numpy.typing import ArrayLike
from typing_extensions import Literal  # for compatibility with python 3.7
from scipy.optimize._numdiff import approx_derivative

from lbfgsb.mathops import np, optimize, sp
from lbfgsb.types import NDArrayFloat

FDMethod = Literal["2-point", "3-point", "cs"]
FD_METHODS: Tuple[FDMethod, FDMethod, FDMethod] = ("2-point", "3-point", "cs")

Scalar = Union[float, int, np.floating]
ObjectiveValue = Union[Scalar, complex, np.complexfloating]
Objective = Callable[[NDArrayFloat], ObjectiveValue]
Gradient = Callable[[NDArrayFloat], NDArrayFloat]
ObjectiveAndGradient = Callable[[NDArrayFloat], Tuple[Scalar, NDArrayFloat]]

JacOption = Optional[Union[Gradient, bool, FDMethod]]
GradOption = Union[Gradient, FDMethod]

FiniteDiffBounds = Union[Tuple[ArrayLike, ArrayLike], Tuple[float, float]]


def _as_real_float(value: object) -> float:
    """Convert a scalar objective value to a Python float.

    Complex values with a nonzero imaginary part are rejected.
    """
    arr = np.asarray(value)

    if arr.ndim != 0:
        raise ValueError(
            "The user-provided objective function must return a scalar value."
        )

    item = arr.item()

    if isinstance(item, complex):
        if item.imag != 0.0:
            raise ValueError(
                "The user-provided objective function returned a complex value "
                "with a nonzero imaginary part."
            )
        return float(item.real)

    return float(item)


def _as_scalar_for_numdiff(value: object) -> Union[float, complex]:
    """Convert objective output to a scalar while preserving complex-step values."""
    arr = np.asarray(value)

    if arr.ndim != 0:
        raise ValueError(
            "The user-provided objective function must return a scalar value."
        )

    item = arr.item()

    if isinstance(item, complex):
        return complex(item)

    return float(item)


def _as_float_array(value: ArrayLike) -> NDArrayFloat:
    """Convert an array-like gradient to a 1-D float ndarray."""
    return np.atleast_1d(np.asarray(value, dtype=float))


def _is_fd_method(value: object) -> bool:
    """Return whether value is a supported finite-difference method."""
    return isinstance(value, str) and value in FD_METHODS


class MemoizedObjectiveAndGradient:
    """Memoized adapter for callables returning ``(f, g)``.

    This adapter allows a SciPy-like objective function

    ``fun(x) -> (f, g)``

    to be exposed as two separate callables:

    ``fun(x) -> f`` and ``grad(x) -> g``.

    The last evaluation is cached so that requesting ``fun(x)`` followed by
    ``grad(x)`` at the same point does not evaluate the user function twice.
    """

    def __init__(self, fun: ObjectiveAndGradient) -> None:
        self._fun = fun
        self._x: Optional[NDArrayFloat] = None
        self._f: Optional[float] = None
        self._g: Optional[NDArrayFloat] = None

    def _same_x(self, x: NDArrayFloat) -> bool:
        return self._x is not None and np.array_equal(x, self._x)

    def _evaluate(self, x: NDArrayFloat) -> None:
        if self._same_x(x):
            return

        f_raw, g_raw = self._fun(np.copy(x))

        self._x = np.copy(x)
        self._f = _as_real_float(f_raw)
        self._g = _as_float_array(g_raw)

    def fun(self, x: NDArrayFloat) -> float:
        self._evaluate(x)
        if self._f is None:
            raise RuntimeError("Objective cache was not initialized.")
        return self._f

    def grad(self, x: NDArrayFloat) -> NDArrayFloat:
        self._evaluate(x)
        if self._g is None:
            raise RuntimeError("Gradient cache was not initialized.")
        return self._g


class ScalarFunction:
    """Scalar function and its gradient.

    This class defines a scalar function ``F: R^n -> R`` and methods for
    evaluating its value and gradient. The gradient can either be provided by
    the user or approximated using finite differences.

    Parameters
    ----------
    fun : callable
        Objective function. Must have signature:

        .. code-block:: python

            fun(x) -> float

        where ``x`` is a one-dimensional array with shape ``(n,)``.

    x0 : ndarray, shape (n,)
        Initial point.

    grad : callable or {'2-point', '3-point', 'cs'}
        Gradient evaluation method.

        If callable, it must have signature:

        .. code-block:: python

            grad(x) -> ndarray, shape (n,)

        If one of ``'2-point'``, ``'3-point'``, or ``'cs'``, the gradient is
        approximated numerically using the corresponding finite-difference
        scheme.

    finite_diff_rel_step : array_like or None
        Relative step size used for finite-difference gradient approximation.

    finite_diff_bounds : tuple
        Lower and upper bounds used by finite differences.

    epsilon : array_like or None, optional
        Absolute step size used for finite differences. If ``None``, relative
        steps are used.

    Notes
    -----
    This class implements memoization. Use the public methods ``fun``,
    ``grad``, and ``fun_and_grad``. Calling one of these methods with a new
    point invalidates cached values from the previous point.
    """

    def __init__(
        self,
        fun: Objective,
        x0: NDArrayFloat,
        grad: GradOption,
        finite_diff_rel_step: Optional[ArrayLike],
        finite_diff_bounds: FiniteDiffBounds,
        epsilon: Optional[ArrayLike] = None,
    ) -> None:
        if not callable(grad) and not _is_fd_method(grad):
            raise ValueError(f"`grad` must be either callable or one of {FD_METHODS}.")

        # The astype call ensures that self.x is a copy of x0.
        self.x: NDArrayFloat = np.atleast_1d(np.asarray(x0, dtype=float))
        self.n: int = self.x.size

        self.nfev: int = 0
        self.ngev: int = 0
        self.nhev: int = 0

        self.f_updated: bool = False
        self.g_updated: bool = False
        self.H_updated: bool = False

        self.f: float = float("inf")
        self.g: NDArrayFloat = np.zeros_like(self.x)

        self._lowest_x: NDArrayFloat = np.copy(self.x)
        self._lowest_f: float = float("inf")

        finite_diff_options: Dict[str, object] = {}
        if _is_fd_method(grad):
            finite_diff_options["method"] = grad
            finite_diff_options["rel_step"] = finite_diff_rel_step
            finite_diff_options["abs_step"] = epsilon
            finite_diff_options["bounds"] = finite_diff_bounds

        def fun_wrapped(x: NDArrayFloat) -> float:
            self.nfev += 1

            fx_raw = fun(np.copy(x))

            try:
                fx = _as_real_float(fx_raw)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "The user-provided objective function must return a scalar value."
                ) from e

            if fx < self._lowest_f:
                self._lowest_x = np.copy(x)
                self._lowest_f = fx

            return fx

        def fun_wrapped_for_numdiff(x: NDArrayFloat) -> Union[float, complex]:
            self.nfev += 1

            fx_raw = fun(np.copy(x))

            try:
                return _as_scalar_for_numdiff(fx_raw)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "The user-provided objective function must return a scalar value."
                ) from e

        def update_fun() -> None:
            self.f = fun_wrapped(self.x)

        self._update_fun_impl: Callable[[], None] = update_fun

        if callable(grad):
            grad_callable = grad

            def grad_wrapped(x: NDArrayFloat) -> NDArrayFloat:
                self.ngev += 1
                return _as_float_array(grad_callable(np.copy(x)))

            def update_grad() -> None:
                self.g = grad_wrapped(self.x)

        else:

            def update_grad() -> None:
                self._update_fun()
                self.ngev += 1
                g = approx_derivative(
                    fun_wrapped_for_numdiff,
                    self.x,
                    f0=self.f,
                    **finite_diff_options,
                )
                self.g = _as_float_array(g)

        self._update_grad_impl: Callable[[], None] = update_grad

    def update_x(self, x: NDArrayFloat) -> None:
        """Update the current point and invalidate cached values."""
        self.x = np.atleast_1d(np.asarray(x, dtype=float))
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

    def fun(self, x: NDArrayFloat) -> float:
        """Return the objective value at ``x``."""
        if not np.array_equal(x, self.x):
            self.update_x(x)
        self._update_fun()
        return self.f

    def grad(self, x: NDArrayFloat) -> NDArrayFloat:
        """Return the gradient at ``x``."""
        if not np.array_equal(x, self.x):
            self.update_x(x)
        self._update_grad()
        return self.g

    def fun_and_grad(self, x: NDArrayFloat) -> Tuple[float, NDArrayFloat]:
        """Return both the objective value and gradient at ``x``."""
        if not np.array_equal(x, self.x):
            self.update_x(x)
        self._update_fun()
        self._update_grad()
        return self.f, self.g


def prepare_scalar_function(
    fun: Union[Objective, ObjectiveAndGradient],
    x0: NDArrayFloat,
    jac: JacOption = None,
    bounds: Optional[FiniteDiffBounds] = None,
    epsilon: Optional[ArrayLike] = None,
    finite_diff_rel_step: Optional[ArrayLike] = None,
) -> ScalarFunction:
    """Create a :class:`ScalarFunction` for scalar minimizers.

    Parameters
    ----------
    fun : callable
        Objective function.

        If ``jac`` is not ``True``, ``fun`` must return a scalar:

        .. code-block:: python

            fun(x) -> float

        If ``jac`` is ``True``, ``fun`` must return both the objective value and
        the gradient:

        .. code-block:: python

            fun(x) -> tuple[float, ndarray]

    x0 : ndarray, shape (n,)
        Initial point.
        
    jac : callable, bool, {'2-point', '3-point', 'cs'} or None, optional
        Gradient evaluation method.

        If callable, ``jac`` must return the gradient vector.

        If ``jac=True``, ``fun`` is assumed to return ``(f, g)``.

        If ``jac`` is one of ``'2-point'``, ``'3-point'``, or ``'cs'``, the
        gradient is approximated using the selected finite-difference method.

        If ``jac`` is ``None`` or ``False``, two-point finite differences with
        absolute step size ``epsilon`` are used.

    bounds : tuple, optional
        Lower and upper bounds used for finite differences. If ``None``, no
        bounds are used.

    epsilon : array_like or None, optional
        Absolute finite-difference step size.

    finite_diff_rel_step : array_like or None, optional
        Relative finite-difference step size.

    Returns
    -------
    ScalarFunction
        Wrapped scalar function with memoized objective and gradient evaluations.
    """
    scalar_fun: Objective
    grad: GradOption

    if jac is True:
        memoized = MemoizedObjectiveAndGradient(cast(ObjectiveAndGradient, fun))
        scalar_fun = memoized.fun
        grad = memoized.grad

    elif jac is False or jac is None:
        scalar_fun = cast(Objective, fun)
        grad = "2-point"

    elif callable(jac):
        scalar_fun = cast(Objective, fun)
        grad = jac

    elif _is_fd_method(jac):
        scalar_fun = cast(Objective, fun)
        grad = jac
        # Explicit finite-difference methods use relative steps.
        epsilon = None

    else:
        raise ValueError(
            "jac must be callable, bool, None, or one of ['2-point', '3-point', 'cs']."
        )

    if bounds is None:
        bounds = (-np.inf, np.inf)

    return ScalarFunction(
        scalar_fun,
        x0,
        grad,
        finite_diff_rel_step,
        bounds,
        epsilon=epsilon,
    )
