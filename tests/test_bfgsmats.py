# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

import logging
from collections import deque
from contextlib import nullcontext as does_not_raise
from typing import Deque, Tuple

import numpy as np
import pytest
from lbfgsb.bfgsmats import (
    LBFGSB_MATRICES,
    bmv,
    bmv_numba,
    is_update_X_and_G,
    make_X_and_G_respect_strong_wolfe,
    update_X_and_G,
)
from lbfgsb.types import NDArrayFloat

logger: logging.Logger = logging.getLogger("L-BFGS-B")
logger.setLevel(logging.INFO)
logging.info("this is a logging test")


@pytest.mark.parametrize(
    "n, exception",
    (
        (0, pytest.raises(ValueError, match=r"n must be an integer > 0.")),
        (1, does_not_raise()),
        (100, does_not_raise()),
    ),
)
def test_matrices(n: int, exception) -> None:
    with exception:
        LBFGSB_MATRICES(n)


@pytest.mark.parametrize(
    "xk,gk,x_old,g_old,expected",
    (
        (
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            np.False_,
        ),
    ),
)
def test_is_update_X_and_G(
    xk: NDArrayFloat,
    gk: NDArrayFloat,
    x_old: NDArrayFloat,
    g_old: NDArrayFloat,
    expected: bool,
) -> None:
    assert is_update_X_and_G(xk, gk, x_old, g_old) is expected


@pytest.mark.parametrize(
    "xk,gk,X,G,maxcor,expected",
    (
        (
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            deque([np.array([1.0, 1.0])]),
            deque([np.array([0.0, 0.0])]),
            5,
            False,
        ),
    ),
)
def test_update_X_and_G(
    xk: NDArrayFloat,
    gk: NDArrayFloat,
    X: Deque[NDArrayFloat],
    G: Deque[NDArrayFloat],
    maxcor: int,
    expected: bool,
) -> None:
    for is_use_numba_jit in [False, True]:
        assert (
            update_X_and_G(xk, gk, X, G, maxcor, is_use_numba_jit=is_use_numba_jit)
            is expected
        )


@pytest.mark.parametrize(
    "X,G,expected",
    (
        (
            deque([np.array([1.0, 1.0]), np.array([1.0, 1.0])]),
            deque([np.array([0.0, 0.0]), np.array([0.0, 0.0])]),
            (deque([np.array([1.0, 1.0])]), deque([np.array([0.0, 0.0])])),
        ),
    ),
)
def test_make_X_and_G_respect_strong_wolfe(
    X: Deque[NDArrayFloat],
    G: Deque[NDArrayFloat],
    expected: Tuple[Deque[NDArrayFloat], Deque[NDArrayFloat]],
) -> None:
    np.testing.assert_allclose(
        make_X_and_G_respect_strong_wolfe(X, G, logger=logger), expected
    )


def random_triangular(
    n: int,
    seed: int = 0,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Generate random triangular LU factorizations."""
    rng = np.random.default_rng(seed)

    L = rng.standard_normal((n, n))
    L = np.tril(L)
    np.fill_diagonal(L, np.abs(np.diag(L)) + 1.0)

    U = rng.standard_normal((n, n))
    U = np.triu(U)
    np.fill_diagonal(U, np.abs(np.diag(U)) + 1.0)

    return L, U


@pytest.mark.parametrize("n", [1, 2, 5, 10, 20, 40])
def test_bmv_matches_scipy(n: int) -> None:
    """Test that both bmv and the numba version return the same results."""
    L, U = random_triangular(n, seed=n)
    v: NDArrayFloat = np.random.randn(n)

    ref: NDArrayFloat = bmv((L, U), v)
    out: NDArrayFloat = bmv_numba(L, U, v)

    np.testing.assert_allclose(
        out,
        ref,
        rtol=1e-12,
        atol=1e-12,
    )
