import logging
from contextlib import nullcontext as does_not_raise
from typing import Deque, Tuple

import numpy as np
import pytest
from lbfgsb.bfgsmats import (
    LBFGSB_MATRICES,
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
            False,
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
            Deque([np.array([1.0, 1.0])]),
            Deque([np.array([0.0, 0.0])]),
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
    assert update_X_and_G(xk, gk, X, G, maxcor) is expected


@pytest.mark.parametrize(
    "X,G,expected",
    (
        (
            Deque([np.array([1.0, 1.0]), np.array([1.0, 1.0])]),
            Deque([np.array([0.0, 0.0]), np.array([0.0, 0.0])]),
            (Deque([np.array([1.0, 1.0])]), Deque([np.array([0.0, 0.0])])),
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
