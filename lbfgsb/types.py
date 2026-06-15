# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Array type aliases for lbfgsb.

Two tiers:
- ``NDArrayFloat`` / ``NDArrayInt`` — concrete NumPy arrays.
  Used exclusively for numba-JIT kernels and internal numpy-only helpers.
  ``ty`` / mypy will reject CuPy or JAX arrays here.

- ``AnyArray`` — a TypeVar bound to ``ArrayProtocol``.
  Used on all public API and backend-agnostic algorithm functions.
  Tells static checkers the function works for any compliant array type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    # These imports only run inside the type checker, never at runtime,
    # so missing packages do not cause ImportError.
    # ruff: noqa: TCH002
    try:
        import cupy as cp

        CupyArray = cp.ndarray
    except ImportError:
        CupyArray = np.ndarray  # fallback so the Union is valid

    try:
        import jax

        JaxArray = jax.Array
    except ImportError:
        JaxArray = np.ndarray  # fallback

# ---------------------------------------------------------------------------
# Concrete NumPy types — numba JIT paths only
# ---------------------------------------------------------------------------

NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int64]

# ---------------------------------------------------------------------------
# Backend-agnostic type — public API and algorithm functions
# ---------------------------------------------------------------------------


@runtime_checkable
class ArrayProtocol(Protocol):
    """Structural interface that every backend array satisfies.

    We don't use ``typing.Protocol`` + ``@runtime_checkable`` here because
    ``isinstance`` checks against a Protocol are slow and we already rely on
    ``Backend.__type__`` for dispatch. This class exists for ``ty``/mypy
    documentation only.
    """

    dtype: np.dtype
    size: int
    ndim: int

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __add__(self, other: Any) -> ArrayProtocol: ...
    def __sub__(self, other: Any) -> ArrayProtocol: ...
    def __mul__(self, other: Any) -> ArrayProtocol: ...
    def __truediv__(self, other: Any) -> ArrayProtocol: ...
    def __neg__(self) -> ArrayProtocol: ...
    def __lt__(self, other: Any) -> ArrayProtocol: ...
    def __gt__(self, other: Any) -> ArrayProtocol: ...
    def __le__(self, other: Any) -> ArrayProtocol: ...
    def __ge__(self, other: Any) -> ArrayProtocol: ...
    def __matmul__(self, other: Any) -> ArrayProtocol: ...
    def __getitem__(self, key: Any) -> ArrayProtocol: ...
    def copy(self) -> ArrayProtocol: ...


# TypeVar used to annotate backend-agnostic public functions.
# ``bound=ArrayProtocol`` lets ty/mypy check that .shape, .dtype, etc. exist.
AnyArray = TypeVar("AnyArray", bound=ArrayProtocol)
