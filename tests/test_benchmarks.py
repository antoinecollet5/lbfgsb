"""Test the benchmark functions and their gradients."""

from typing import Callable

import numdifftools as nd
import numpy as np
import pytest
from lbfgsb import (
    ackley,
    ackley_grad,
    beale,
    beale_grad,
    griewank,
    griewank_grad,
    quartic,
    quartic_grad,
    rastrigin,
    rastrigin_grad,
    rosenbrock,
    rosenbrock_grad,
    sphere,
    sphere_grad,
    styblinski_tang,
    styblinski_tang_grad,
)


@pytest.mark.parametrize(
    "function,gradient",
    (
        (ackley, ackley_grad),
        (beale, beale_grad),
        (griewank, griewank_grad),
        (quartic, quartic_grad),
        (rastrigin, rastrigin_grad),
        (rosenbrock, rosenbrock_grad),
        (sphere, sphere_grad),
        (styblinski_tang, styblinski_tang_grad),
    ),
)
def test_benchmark_functions(function: Callable, gradient: Callable) -> None:
    """Test the functions and their gradient."""
    x = np.array([7.0, 1.0, 2.0, 3.0, 9.0])

    np.testing.assert_allclose(
        gradient(x), nd.Gradient(function, step=1e-5)(x), atol=1e-5, rtol=1e-5
    )
