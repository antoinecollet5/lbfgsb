# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from lbfgsb.benchmarks import rosenbrock, rosenbrock_grad
from lbfgsb.scalar_function import FD_METHODS, ScalarFunction, prepare_scalar_function


@pytest.mark.parametrize(
    "grad, exception",
    (
        (rosenbrock_grad, does_not_raise()),
        ("2-point", does_not_raise()),
        ("3-point", does_not_raise()),
        ("cs", does_not_raise()),
        (
            None,
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"`grad` must be either callable or one of {FD_METHODS}."
                ),
            ),
        ),
        (
            "something_weird",
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"`grad` must be either callable or one of {FD_METHODS}."
                ),
            ),
        ),
    ),
)
def test_ScalarFunction(grad, exception) -> None:
    with exception:
        ScalarFunction(rosenbrock, np.array([0.0, 5.0]), (), grad, 0.1, 0.1, 0.1)


@pytest.mark.parametrize(
    "jac, exception",
    (
        (rosenbrock_grad, does_not_raise()),
        ("2-point", does_not_raise()),
        ("3-point", does_not_raise()),
        ("cs", does_not_raise()),
        (None, does_not_raise()),
        (
            "something_weird",
            pytest.raises(
                ValueError,
                match=(
                    "jac must be callable, None or among ['2-point', '3-point', 'cs']."
                ),
            ),
        ),
    ),
)
def test_scalar_function(jac, exception) -> None:
    with exception:
        sf = prepare_scalar_function(rosenbrock, np.array([0.0, 5.0]), jac)
        sf.fun(np.array([0.0, 10.0]))
        sf.grad(np.array([12.0, 10.0]))


def test_scalar_function_numpy() -> None:
    def wrapper(x):
        return np.array(rosenbrock(x))

    sf = prepare_scalar_function(wrapper, np.array([0.0, 5.0]), "cs")
    sf.fun(np.array([0.0, 10.0]))


def test_scalar_function_str() -> None:
    def wrapper(x):
        return [[2.0], []]

    sf = prepare_scalar_function(wrapper, np.array([0.0, 5.0]), "cs")
    with pytest.raises(
        ValueError,
        match="The user-provided objective function must return a scalar value.",
    ):
        sf.fun(np.array([0.0, 10.0]))
