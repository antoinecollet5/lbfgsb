# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

"""Tests for base functions."""

import logging
import re
from contextlib import nullcontext as does_not_raise
from typing import Sequence

import numpy as np
import pytest
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
from lbfgsb.types import NDArrayFloat


@pytest.mark.parametrize(
    "x0, bounds, expected_exception",
    (
        [np.array([0, 0, 0]), np.array([[0, 0], [0, 0], [0, 0]]), does_not_raise()],
        [np.array([0, 0, 0]), None, does_not_raise()],
        [
            np.array([]),
            np.array([]),
            pytest.raises(ValueError, match="x0 cannot be an empty vector!"),
        ],
        [
            np.array([]),
            np.array([1.0]),
            pytest.raises(ValueError, match="x0 cannot be an empty vector!"),
        ],
        [
            np.array([1.0]),
            np.array([]),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Bounds have shape ((0,)), while shape (1, 2) is expected!"
                ),
            ),
        ],
        [
            np.array([0, 0, 0]),
            np.array([[0, 1.0], [-1, 0], [1.0, -1.0]]),
            pytest.raises(
                ValueError,
                match="One of the lower bounds is greater than an upper bound.",
            ),
        ],
        [np.array([1e-10]), np.array([(1e-10, 1000000)]), does_not_raise()],
        [np.array([1e6]), np.array([(1e-10, 1000000)]), does_not_raise()],
        [
            np.array([1e-15]),
            np.array([(1e-10, 1000000)]),
            pytest.raises(
                ValueError,
                match="There are 1 values violating the lower bounds and"
                " 0 values violating the upper bounds!",
            ),
        ],
        [
            np.array([1e7]),
            np.array([(1e-10, 1000000)]),
            pytest.raises(
                ValueError,
                match="There are 0 values violating the lower bounds "
                "and 1 values violating the upper bounds!",
            ),
        ],
    ),
)
def test_get_bounds(x0, bounds, expected_exception) -> None:
    with expected_exception:
        get_bounds(x0, bounds)


@pytest.mark.parametrize(
    "arrs, expected",
    [
        ([], False),
        ([np.array([])], False),
        ([np.array([1.0, 2.0])], False),
        ([np.array([1.0, 2.0]), np.array([1.0])], False),
        ([np.array([1.0, 2.0]), np.array([1.0])], False),
        ([np.array([np.inf])], True),
        ([np.array([-np.inf, 2.0])], True),
        ([np.array([1.0, 2.0]), np.array([np.inf, 1.0])], True),
    ],
)
def test_is_any_inf(arrs: Sequence[NDArrayFloat], expected: bool) -> None:
    """Test the function is_any_inf."""
    assert expected == is_any_inf(arrs)


@pytest.mark.parametrize(
    "x, bounds, expected",
    [
        (np.array([0, 0, 0]), np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0, 0])),
    ],
)
def test_clip2bounds(
    x: NDArrayFloat, bounds: NDArrayFloat, expected: NDArrayFloat
) -> None:
    """Test the function clip2bounds."""
    np.testing.assert_array_equal(clip2bounds(x, bounds[:, 0], bounds[:, 1]), expected)


@pytest.mark.parametrize(
    "x, bounds, expected",
    [
        (np.array([0, 0, 0]), np.array([[0, 0], [0, 0], [0, 0]]), 3),
    ],
)
def test_count_var_at_bounds(
    x: NDArrayFloat, bounds: NDArrayFloat, expected: int
) -> None:
    """Test the function count_var_at_bounds."""
    assert count_var_at_bounds(x, bounds[:, 0], bounds[:, 1]) == expected


@pytest.mark.parametrize(
    "iprint, logger", ([0, None], [100, None], [100, logging.Logger("L-BFGS-B")])
)
def test_display_start(iprint: int, logger: logging.Logger) -> None:
    """Test the function display_start."""
    display_start(
        epsmch=1e-16,
        n=10,
        m=100,
        nvar_at_b=2,
        iprint=iprint,
        logger=logger,
    )


def test_projgr() -> None:
    """Test the function projgr."""
    assert projgr is not None


@pytest.mark.parametrize(
    "iprint",
    (-1, 0, 1, 2, 100, 120),
)
@pytest.mark.parametrize("logger", (None, logging.Logger("L-BFGS-B")))
def test_display_iter(iprint: int, logger: logging.Logger) -> None:
    """Test the function display_iter."""
    display_iter(
        niter=3,
        sbgnrm=4.5,
        f=38.9,
        iprint=iprint,
        logger=logger,
    )


@pytest.mark.parametrize(
    "is_final_display",
    (True, False),
)
@pytest.mark.parametrize(
    "iprint",
    (-1, 0, 1, 2, 100, 120),
)
@pytest.mark.parametrize("logger", (None, logging.Logger("L-BFGS-B")))
def test_display_results(
    is_final_display: bool, iprint: int, logger: logging.Logger
) -> None:
    """Test the function display_results."""
    display_results(
        n_iterations=3,
        max_iter=10,
        x=np.array([0.0, 1.0, 7.0]),
        grad=np.array([0.0, 1.0, 7.0]),
        lb=np.zeros(3),
        ub=np.ones(3) * 30.0,
        f0=1e-3,
        gtol=1e-5,
        is_final_display=is_final_display,
        iprint=iprint,
        logger=logger,
    )
