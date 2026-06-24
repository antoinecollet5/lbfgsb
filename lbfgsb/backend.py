# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Backend abstraction for lbfgsb.

Follows the POT (Python Optimal Transport) pattern: a Backend class per
array library, auto-detected from the array types passed by the caller.
Convention: use ``nx`` (not ``np``) for a backend instance inside functions.

Supported backends
------------------
- ``NumpyBackend``      : plain NumPy (default)
- ``NumbaBackend``      : NumPy arrays + Numba JIT inner loops
- ``CupyBackend``       : CuPy GPU arrays
- ``JaxBackend``        : JAX arrays (functional, immutable)

Adding a new backend
--------------------
1. Subclass :class:`Backend`.
2. Set ``__name__`` and ``__type__``.
3. Implement every abstract method.
4. Call :func:`register_backend` with an instance.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generator

import numpy as np
import numpy as np_real  # always real numpy, never shimmed
import scipy.linalg as _scipy_la

if TYPE_CHECKING:
    try:
        import cupy as cp
    except ImportError:
        cp = None

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        jax = None  # type: ignore[assignment]
        jnp = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Backend(ABC):
    """Abstract backend — one concrete subclass per array library."""

    __name__: str = ""
    __type__: type = type(None)

    def __repr__(self) -> str:
        return f"<Backend: {self.__name__}>"

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: Any = np.float64) -> Any: ...

    @abstractmethod
    def zeros_like(self, a: Any) -> Any: ...

    @abstractmethod
    def empty_like(self, a: Any) -> Any: ...

    @abstractmethod
    def full(
        self, shape: tuple[int, ...], fill_value: float, dtype: Any = np.float64
    ) -> Any: ...

    @abstractmethod
    def copy(self, a: Any) -> Any: ...

    # ------------------------------------------------------------------
    # Element-wise / reduction
    # ------------------------------------------------------------------

    @abstractmethod
    def where(self, cond: Any, x: Any, y: Any) -> Any: ...

    @abstractmethod
    def isfinite(self, a: Any) -> Any: ...

    @abstractmethod
    def isinf(self, a: Any) -> Any: ...

    @abstractmethod
    def isnan(self, a: Any) -> Any: ...

    @abstractmethod
    def all(self, a: Any) -> bool: ...

    @abstractmethod
    def any(self, a: Any) -> bool: ...

    @abstractmethod
    def max(self, a: Any) -> float: ...

    @abstractmethod
    def min(self, a: Any) -> float: ...

    @abstractmethod
    def abs(self, a: Any) -> Any: ...

    @abstractmethod
    def sqrt(self, a: Any) -> Any: ...

    @abstractmethod
    def clip(self, a: Any, lo: Any, hi: Any) -> Any: ...

    @abstractmethod
    def dot(self, a: Any, b: Any) -> Any: ...

    @abstractmethod
    def inf(self) -> float: ...

    @abstractmethod
    def finfo_max(self) -> float: ...

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @abstractmethod
    def flatnonzero(self, a: Any) -> Any: ...

    @abstractmethod
    def argsort(self, a: Any) -> Any: ...

    @abstractmethod
    def set_item(self, a: Any, idx: Any, value: Any) -> Any:
        """Return array with ``a[idx] = value``.

        For mutable backends (NumPy, CuPy) this is done in-place and the
        same array is returned. For immutable backends (JAX) a new array
        is returned via ``a.at[idx].set(value)``.
        """
        ...

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    @abstractmethod
    def cholesky_lower(self, a: Any) -> Any:
        """Return lower-triangular Cholesky factor L such that L @ L.T == a."""
        ...

    @abstractmethod
    def solve_triangular_l(self, L: Any, b: Any) -> Any:
        """Solve L x = b where L is lower-triangular."""
        ...

    @abstractmethod
    def solve_triangular_u(self, U: Any, b: Any) -> Any:
        """Solve U x = b where U is upper-triangular."""
        ...

    @abstractmethod
    def solve(self, A: Any, b: Any) -> Any:
        """Solve A x = b (general square system)."""
        ...

    @abstractmethod
    def diag(self, a: Any) -> Any: ...

    @abstractmethod
    def fill_diagonal(self, a: Any, val: float) -> Any:
        """Return array with diagonal set to val (functional for JAX)."""
        ...

    @abstractmethod
    def diff(self, a: Any, axis: int = 0) -> Any: ...

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @abstractmethod
    @contextlib.contextmanager
    def errstate(self, **kwargs: Any) -> Generator[None, None, None]: ...

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @abstractmethod
    def to_numpy(self, a: Any) -> np_real.ndarray: ...

    @abstractmethod
    def from_numpy(self, a: np_real.ndarray, type_as: Any = None) -> Any: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: list[Backend] = []


def register_backend(backend: Backend) -> None:
    """Register a backend instance. Last registered = lowest priority."""
    _REGISTRY.append(backend)


def get_backend(*arrays: Any) -> Backend:
    """Return the backend matching the type of the given arrays.

    Ignores ``None`` entries. Raises ``TypeError`` if arrays span multiple
    backends or if no backend is registered for their type.
    """
    live = [a for a in arrays if a is not None]
    if not live:
        raise ValueError("At least one non-None array is required.")

    for backend in _REGISTRY:
        if all(isinstance(a, backend.__type__) for a in live):
            return backend

    types = [type(a).__name__ for a in live]
    raise TypeError(
        f"No registered backend for array types {types}. "
        "Register one via lbfgsb.backend.register_backend()."
    )


# ---------------------------------------------------------------------------
# NumPy backend (always available)
# ---------------------------------------------------------------------------


class NumpyBackend(Backend):
    """Backend for plain NumPy arrays."""

    __name__ = "numpy"
    __type__ = np_real.ndarray

    # creation
    def zeros(self, shape, dtype=np.float64):
        return np.zeros(shape, dtype=dtype)

    def zeros_like(self, a):
        return np.zeros_like(a)

    def empty_like(self, a):
        return np.empty_like(a)

    def full(self, shape, fill_value, dtype=np.float64):
        return np.full(shape, fill_value, dtype=dtype)

    def copy(self, a):
        return a.copy()

    # element-wise
    def where(self, cond, x, y):
        return np.where(cond, x, y)

    def isfinite(self, a):
        return np.isfinite(a)

    def isinf(self, a):
        return np.isinf(a)

    def isnan(self, a):
        return np.isnan(a)

    def all(self, a):
        return bool(np.all(a))

    def any(self, a):
        return bool(np.any(a))

    def max(self, a):
        return float(np.max(a))

    def min(self, a):
        return float(np.min(a))

    def abs(self, a):
        return np.abs(a)

    def sqrt(self, a):
        return np.sqrt(a)

    def clip(self, a, lo, hi):
        return np.clip(a, lo, hi)

    def dot(self, a, b):
        return np.dot(a, b)

    def inf(self):
        return np.inf

    def finfo_max(self):
        return float(np.finfo(np.float64).max)

    # indexing
    def flatnonzero(self, a):
        return np.flatnonzero(a)

    def argsort(self, a):
        return np.argsort(a)

    def set_item(self, a, idx, value):
        a[idx] = value  # in-place, fine for numpy
        return a

    # linalg
    def cholesky_lower(self, a):
        return _scipy_la.cholesky(a, lower=True)

    def solve_triangular_l(self, L, b):
        return _scipy_la.solve_triangular(L, b, lower=True)

    def solve_triangular_u(self, U, b):
        return _scipy_la.solve_triangular(U, b, lower=False)

    def solve(self, A, b):
        return np.linalg.solve(A, b)

    def diag(self, a):
        return np.diag(a)

    def fill_diagonal(self, a, val):
        np.fill_diagonal(a, val)
        return a

    def diff(self, a, axis=0):
        return np.diff(a, axis=axis)

    # context
    @contextlib.contextmanager
    def errstate(self, **kwargs):
        with np.errstate(**kwargs):
            yield

    # conversion
    def to_numpy(self, a):
        return a

    def from_numpy(self, a, type_as=None):
        if type_as is None:
            return np.asarray(a, dtype=np.float64)
        return np.asarray(a, dtype=type_as.dtype)


# ---------------------------------------------------------------------------
# Numba backend — same arrays as NumPy, JIT inner loops
# ---------------------------------------------------------------------------


class NumbaBackend(NumpyBackend):
    """NumPy arrays with Numba-JIT inner loops for hot paths.

    All array creation / linalg delegates to NumpyBackend.
    The solver entry points (cauchy, subspacemin, bfgsmats) check
    ``isinstance(nx, NumbaBackend)`` and call the ``@njit`` kernels.
    """

    __name__ = "numpy+numba"
    # same __type__ as NumpyBackend so isinstance checks still work


# ---------------------------------------------------------------------------
# CuPy backend
# ---------------------------------------------------------------------------


def _make_cupy_backend() -> Backend:
    """Build and return a CupyBackend instance (deferred import)."""

    import cupy as cp
    import cupy.linalg as cpla
    import cupyx

    class CupyBackend(Backend):
        __name__ = "cupy"
        __type__ = cp.ndarray

        # creation
        def zeros(self, shape, dtype=np.float64):
            return cp.zeros(shape, dtype=dtype)

        def zeros_like(self, a):
            return cp.zeros_like(a)

        def empty_like(self, a):
            return cp.empty_like(a)

        def full(self, shape, fill_value, dtype=np.float64):
            return cp.full(shape, fill_value, dtype=dtype)

        def copy(self, a):
            return a.copy()

        # element-wise
        def where(self, cond, x, y):
            return cp.where(cond, x, y)

        def isfinite(self, a):
            return cp.isfinite(a)

        def isinf(self, a):
            return cp.isinf(a)

        def isnan(self, a):
            return cp.isnan(a)

        def all(self, a):
            return bool(cp.all(a))

        def any(self, a):
            return bool(cp.any(a))

        def max(self, a):
            return float(cp.max(a))

        def min(self, a):
            return float(cp.min(a))

        def abs(self, a):
            return cp.abs(a)

        def sqrt(self, a):
            return cp.sqrt(a)

        def clip(self, a, lo, hi):
            return cp.clip(a, lo, hi)

        def dot(self, a, b):
            return cp.dot(a, b)

        def inf(self):
            return cp.inf

        def finfo_max(self):
            return float(np_real.finfo(np_real.float64).max)

        # indexing
        def flatnonzero(self, a):
            return cp.flatnonzero(a)

        def argsort(self, a):
            return cp.argsort(a)

        def set_item(self, a, idx, value):
            a[idx] = value
            return a

        # linalg — CuPy lacks solve_triangular; fall back via host
        def cholesky_lower(self, a):
            return cpla.cholesky(a)  # CuPy returns lower by default

        def solve_triangular_l(self, L, b):
            return cp.asarray(
                _scipy_la.solve_triangular(cp.asnumpy(L), cp.asnumpy(b), lower=True)
            )

        def solve_triangular_u(self, U, b):
            return cp.asarray(
                _scipy_la.solve_triangular(cp.asnumpy(U), cp.asnumpy(b), lower=False)
            )

        def solve(self, A, b):
            return cpla.solve(A, b)

        def diag(self, a):
            return cp.diag(a)

        def fill_diagonal(self, a, val):
            cp.fill_diagonal(a, val)
            return a

        def diff(self, a, axis=0):
            return cp.diff(a, axis=axis)

        @contextlib.contextmanager
        def errstate(self, **kwargs):
            with cupyx.errstate(**kwargs):
                yield

        def to_numpy(self, a):
            return cp.asnumpy(a)

        def from_numpy(self, a, type_as=None):
            if type_as is None:
                return cp.asarray(a, dtype=cp.float64)
            with cp.cuda.Device(type_as.device):
                return cp.asarray(a, dtype=type_as.dtype)

    return CupyBackend()


# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------


def _make_jax_backend() -> Backend:
    """Build and return a JaxBackend instance (deferred import)."""

    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jla

    class JaxBackend(Backend):
        __name__ = "jax"
        __type__ = jax.Array

        # creation
        def zeros(self, shape, dtype=np.float64):
            return jnp.zeros(shape, dtype=dtype)

        def zeros_like(self, a):
            return jnp.zeros_like(a)

        def empty_like(self, a):
            return jnp.zeros_like(a)  # JAX has no empty

        def full(self, shape, fill_value, dtype=np.float64):
            return jnp.full(shape, fill_value, dtype=dtype)

        def copy(self, a):
            return jnp.array(a)

        # element-wise
        def where(self, cond, x, y):
            return jnp.where(cond, x, y)

        def isfinite(self, a):
            return jnp.isfinite(a)

        def isinf(self, a):
            return jnp.isinf(a)

        def isnan(self, a):
            return jnp.isnan(a)

        def all(self, a):
            return bool(jnp.all(a))

        def any(self, a):
            return bool(jnp.any(a))

        def max(self, a):
            return float(jnp.max(a))

        def min(self, a):
            return float(jnp.min(a))

        def abs(self, a):
            return jnp.abs(a)

        def sqrt(self, a):
            return jnp.sqrt(a)

        def clip(self, a, lo, hi):
            return jnp.clip(a, lo, hi)

        def dot(self, a, b):
            return jnp.dot(a, b)

        def inf(self):
            return float(jnp.inf)

        def finfo_max(self):
            return float(np_real.finfo(np_real.float64).max)

        # indexing
        def flatnonzero(self, a):
            return jnp.flatnonzero(a)

        def argsort(self, a):
            return jnp.argsort(a)

        def set_item(self, a, idx, value):
            return a.at[idx].set(value)  # functional, returns new array

        # linalg
        def cholesky_lower(self, a):
            return jla.cholesky(a, lower=True)

        def solve_triangular_l(self, L, b):
            return jla.solve_triangular(L, b, lower=True)

        def solve_triangular_u(self, U, b):
            return jla.solve_triangular(U, b, lower=False)

        def solve(self, A, b):
            return jnp.linalg.solve(A, b)

        def diag(self, a):
            return jnp.diag(a)

        def fill_diagonal(self, a, val):
            return a.at[jnp.diag_indices(a.shape[0])].set(val)

        def diff(self, a, axis=0):
            return jnp.diff(a, axis=axis)

        @contextlib.contextmanager
        def errstate(self, **kwargs):
            yield  # JAX has no errstate; no-op

        def to_numpy(self, a):
            return np_real.array(a)

        def from_numpy(self, a, type_as=None):
            if type_as is None:
                return jnp.asarray(a, dtype=jnp.float64)
            return jnp.asarray(a, dtype=type_as.dtype)

    return JaxBackend()


# ---------------------------------------------------------------------------
# Auto-register available backends at import time
# ---------------------------------------------------------------------------

_numpy_backend = NumpyBackend()
_numba_backend = NumbaBackend()

register_backend(_numpy_backend)
register_backend(_numba_backend)  # checked after numpy; same __type__

try:
    register_backend(_make_cupy_backend())
except Exception:
    pass  # CuPy not installed

try:
    register_backend(_make_jax_backend())
except Exception:
    pass  # JAX not installed


def get_numpy_backend() -> NumpyBackend:
    """Return the singleton NumpyBackend."""
    return _numpy_backend


def get_numba_backend() -> NumbaBackend:
    """Return the singleton NumbaBackend (use for explicit opt-in)."""
    return _numba_backend
