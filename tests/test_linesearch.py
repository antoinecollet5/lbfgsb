import numpy as np
import pytest
from lbfgsb.linesearch import max_allowed_steplength
from lbfgsb.types import NDArrayFloat


@pytest.mark.parametrize(
    "x,d,lb,ub,max_steplength,n_iter,expected",
    (
        (np.array([]), np.array([]), np.array([]), np.array([]), 0.0, 0, 1.0),
        (
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([-np.inf, -np.inf]),
            np.array([np.inf, np.inf]),
            100.0,
            10,
            100.0,
        ),
        (
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([0.5, 0.5]),
            np.array([1.5, 1.5]),
            100.0,
            10,
            0.5,
        ),
        (
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([-np.inf, np.inf]),
            np.array([1.6, np.inf]),
            100.0,
            10,
            0.6,
        ),
        (
            np.array([1.0, 1.0]),
            np.array([1.0, -1.0]),
            np.array([-np.inf, 0.3]),
            np.array([np.inf, np.inf]),
            100.0,
            10,
            0.7,
        ),
    ),
)
def test_max_allowed_steplength(
    x: NDArrayFloat,
    d: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    max_steplength: float,
    n_iter: int,
    expected: float,
) -> None:
    np.testing.assert_allclose(
        max_allowed_steplength(x, d, lb, ub, max_steplength, n_iter), expected
    )
