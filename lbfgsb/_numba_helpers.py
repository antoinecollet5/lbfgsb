# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

"""
Provide njit from numba.

If numba is not available provide an effectless decorator instead.
"""

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def wrapper(func: F) -> F:
            return func

        return wrapper

    def prange(*args, **kwargs):  # type: ignore
        return range(*args, **kwargs)


__all__ = ["njit", "NUMBA_AVAILABLE"]
