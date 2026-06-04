r"""
Provide finite difference approxation for the gradient and hessian.

This is basically the same as provided by the library
`numdifftools <https://numdifftools.readthedocs.io/en/latest/index.html.>`_ but it
supports multiprocessing for gradient approximation which is quite useful when
working with large problems (python is slow).

grad = finite_gradient(np.array([1, 1]), rosen)

Chapitre très intéressant:
    https://pythonnumericalmethods.berkeley.edu/notebooks/chapter20.02-Finite-Difference-Approximating-Derivatives.html

@author: acollet
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from hytecio.utils.types import NDArrayFloat, getInfoLogger


def rosen(x):
    """Rosenbrock function."""
    return (1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def rosen_gradient(x):
    """Rosenbrock function first derivative."""
    return np.array(
        [400 * x[0] ** 3 + (2 - 400 * x[1]) * x[0] - 2, 200 * (x[1] - x[0] ** 2)]
    )


def rosen_hessian(x):
    """Rosenbrock function second derivative."""
    return np.array(
        [[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200.0]]
    )


def is_jacobian_correct(
    x: np.ndarray,
    fm: Callable,
    jac: Callable,
    fm_args: Optional[Union[Tuple[Any], List[Any]]] = None,
    fm_kwargs: Optional[Dict[str, Any]] = None,
    jac_args: Optional[Tuple[Any]] = None,
    jac_kwargs: Optional[Dict[str, Any]] = None,
    accuracy: int = 0,
    eps: Optional[float] = None,
    max_workers: int = 1,
    comptor_kw: Optional[str] = None,
) -> bool:
    """
    Check by finite difference if the Jacobian matrix is correct.

    Parameters
    ----------
    x : np.ndarray
        The input parameters vector.
    fm : Callable
        Forward model.
    jac : Callable
        Jacobian model.
    fm_args: Tuple[Any]
        Positional arguments for the forward model.
    fm_kwargs : Dict[Any, Any]
        Keyword arguments for the forward model.
    grad_args: Tuple[Any]
        Positional arguments for the gradient model.
    grad_kwargs : Dict[Any, Any]
        Keyword arguments for the gradient model.
    accuracy : int, optional
        Number of points to use for the finite difference approximation.
        Possible values are 0 (2 points), 1 (4 points), 2 (6 points),
        3 (8 points). The default is 0 which corresponds to the central
        difference scheme (2 points).
    eps: float, optional
        The epsilon for the computation (h). The default value has been
        taken from the C++ implementation of
        :cite:`wieschollek2016cppoptimizationlibrary`, and should correspond
        to the optimal h taking into account the roundoff errors due to
        the machine precision. The default is -2.2204e-6.
    max_workers: int
        Number of workers used. If different from one, the calculation relies on
        multi-processing to decrease the computation time. The default is 1.
    comptor_kw: Optional[str] = None

    Returns
    -------
    bool
        True if the Jacobian is correct, false otherwise.

    """
    if jac_args is None:
        _jac_args: Tuple[()] = ()
    else:
        _jac_args = jac_args
    if jac_kwargs is None:
        jac_kwargs = {}
    actual_jac = jac(x, *_jac_args, **jac_kwargs)
    expected_jac = finite_jacobian(
        x, fm, fm_args, fm_kwargs, accuracy, eps, max_workers, comptor_kw=comptor_kw
    )
    # Handle the gradient case
    # if actual_jac.ndim != expected_jac.ndim:
    #     expected_jac = expected_jac.reshape(*actual_jac.shape)
    return is_all_close(actual_jac, expected_jac)


def is_gradient_correct(
    x: np.ndarray,
    fm: Callable,
    grad: Callable,
    fm_args: Optional[Union[Tuple[Any], List[Any]]] = None,
    fm_kwargs: Optional[Dict[str, Any]] = None,
    grad_args: Optional[Tuple[Any]] = None,
    grad_kwargs: Optional[Dict[str, Any]] = None,
    accuracy: int = 0,
    eps: Optional[float] = None,
    max_workers: int = 1,
    comptor_kw: Optional[str] = None,
) -> bool:
    """
    Check by finite difference if the gradient is correct.

    Parameters
    ----------
    x : np.ndarray
        The input parameters vector.
    fm : Callable
        Forward model.
    grad : Callable
        Gradient model.
    fm_args: Tuple[Any]
        Positional arguments for the forward model.
    fm_kwargs : Dict[Any, Any]
        Keyword arguments for the forward model.
    grad_args: Tuple[Any]
        Positional arguments for the gradient model.
    grad_kwargs : Dict[Any, Any]
        Keyword arguments for the gradient model.
    accuracy : int, optional
        Number of points to use for the finite difference approximation.
        Possible values are 0 (2 points), 1 (4 points), 2 (6 points),
        3 (4 points). The default is 0 which corresponds to the central
        difference scheme (2 points).
    eps: float, optional
        The epsilon for the computation (h). The default value has been
        taken from the C++ implementation of
        :cite:`wieschollek2016cppoptimizationlibrary`, and should correspond
        to the optimal h taking into account the roundoff errors due to
        the machine precision. The default is -2.2204e-6.
    max_workers: int
        Number of workers used. If different from one, the calculation relies on
        multi-processing to decrease the computation time. The default is 1.
    ppe_kwargs: Dict[str, Any]
        Keywords arguments for the ProcessPoolExecutor instance. This is used only if
        max_workers is above 1.

    Returns
    -------
    bool
        True if the gradient is correct, false otherwise.

    """
    return is_jacobian_correct(
        x=x,
        fm=fm,
        jac=grad,
        fm_args=fm_args,
        fm_kwargs=fm_kwargs,
        jac_args=grad_args,
        jac_kwargs=grad_kwargs,
        accuracy=accuracy,
        eps=eps,
        max_workers=max_workers,
        comptor_kw=comptor_kw,
    )


def is_all_close(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-2) -> bool:
    """Return whether the two vectors are approximately equal."""
    scale = np.maximum(np.maximum(np.abs(v1), np.abs(v2)), 1.0)
    return bool(np.less_equal((np.abs(v1 - v2)), eps * scale).all())


@dataclass
class FDParams:
    x0: NDArrayFloat
    shape: Tuple[int]
    inner_steps: int
    coeff: List[List[float]]
    coeff2: List[List[float]]
    fm: Callable
    fm_args: Sequence[Any]
    fm_kwargs: Dict[str, Any]
    dd_val: float
    accuracy: int
    eps: float
    comptor_kw: Optional[str]


def approximate_elt_jac(elt_id: int, fd_params: FDParams) -> NDArrayFloat:
    _res = []
    for s in range(fd_params.inner_steps):
        fd_params.x0[elt_id] += fd_params.coeff2[fd_params.accuracy][s] * fd_params.eps
        if fd_params.comptor_kw is not None:
            fd_params.fm_kwargs[fd_params.comptor_kw] = (
                elt_id * fd_params.inner_steps + s
            )
        _res.append(
            fd_params.coeff[fd_params.accuracy][s]
            * fd_params.fm(
                fd_params.x0.reshape(fd_params.shape),
                *fd_params.fm_args,
                **fd_params.fm_kwargs,
            )
        )
        fd_params.x0[elt_id] -= fd_params.coeff2[fd_params.accuracy][s] * fd_params.eps
    return np.sum(_res, axis=0) / fd_params.dd_val


def finite_jacobian(
    x: np.ndarray,
    fm: Callable,
    fm_args: Optional[Sequence[Any]] = None,
    fm_kwargs: Optional[Dict[str, Any]] = None,
    accuracy: int = 0,
    eps: Optional[float] = None,
    max_workers: int = 1,
    comptor_kw: Optional[str] = None,
) -> NDArrayFloat:
    r"""
    Compute the Jacobian by finite difference.

    The Jacobian is computed by using Taylor Series. For instance, if
    accurcy = 1, we use 4 points, which means that
    we take the Taylor series of $f$ around $a = x_j$ and compute the series
    at $x = x_{j-2}, x_{j-1}, x_{j+1}, x_{j+2}$.

    .. math::
        \begin{eqnarray*}
        f(x_{j-2}) &=& f(x_j) - 2hf^{\prime}(x_j) + \frac{4h^2f''(x_j)}{2} -
        \frac{8h^3f'''(x_j)}{6} + \frac{16h^4f''''(x_j)}{24}
        - \frac{32h^5f'''''(x_j)}{120} + \cdots\\
        f(x_{j-1}) &=& f(x_j) - hf^{\prime}(x_j) + \frac{h^2f''(x_j)}{2}
        - \frac{h^3f'''(x_j)}{6} + \frac{h^4f''''(x_j)}{24}
        - \frac{h^5f'''''(x_j)}{120} + \cdots\\
        f(x_{j+1}) &=& f(x_j) + hf^{\prime}(x_j) + \frac{h^2f''(x_j)}{2}
        + \frac{h^3f'''(x_j)}{6} + \frac{h^4f''''(x_j)}{24}
        + \frac{h^5f'''''(x_j)}{120} + \cdots\\
        f(x_{j+2}) &=& f(x_j) + 2hf^{\prime}(x_j) + \frac{4h^2f''(x_j)}{2}
        + \frac{8h^3f'''(x_j)}{6} + \frac{16h^4f''''(x_j)}{24}
        + \frac{32h^5f'''''(x_j)}{120} + \cdots
        \end{eqnarray*}

    To get the $h^2, h^3$, and $h^4$ terms to cancel out, we can compute

    .. math::
        f(x_{j-2}) - 8f(x_{j-1}) + 8f(x_{j-1}) - f(x_{j+2})
        = 12hf^{\prime}(x_j) - \frac{48h^5f'''''(x_j)}{120}

    which can be rearranged to

    .. math::
        f^{\prime}(x_j) = \frac{f(x_{j-2}) - 8f(x_{j-1})
        + 8f(x_{j-1}) - f(x_{j+2})}{12h} + O(h^4).

    This formula is a better approximation for the derivative at $x_j$
    than the central difference formula, but requires twice as many
    calculations.

    Parameters
    ----------
    x : np.ndarray
        The input parameters array.
    fm : Callable
        Forward model.
    fm_args: Tuple[Any]
        Positional arguments for the forward model.
    fm_kwargs : Dict[Any, Any]
        Keyword arguments for the forward model.
    accuracy : int, optional
        Number of points to use for the finite difference approximation.
        Possible values are 0 (2 points), 1 (4 points), 2 (6 points),
        3 (8 points). The default is 0 which corresponds to the central
        difference scheme (2 points).
    eps: float, optional
        The epsilon for the computation (h). By default, it take 1e-6 times
        the maximum absolute value of the input data.

        .. :math:

            \delta s = \begin{cases}
            \epsilon^{1/3} min\Big(max\left(\lvert s\rvert\right),
            1\Big) & \text{if} \; max\left(\lvert s\rvert\right) > 0\\
            \epsilon^{1/3} & \text{otherwise}
            \end{cases}

    max_workers: int
        Number of workers used. If different from one, the calculation relies on
        multi-processing to decrease the computation time. The default is 1.
    ppe_kwargs: Dict[str, Any]
        Keywords arguments for the ProcessPoolExecutor instance. This is used only if
        max_workers is above 1.

    Returns
    -------
    NDArrayFloat
        The finite difference Jacobian matrix.

    """
    x0 = np.array(x).astype(np.float64)
    if eps is None:
        _eps = np.power(np.finfo(float).eps, 1 / 3) * min(1, np.max(np.abs(x0)))
        if _eps == 0:
            _eps = np.power(np.finfo(float).eps, 1 / 3)
    else:
        _eps = eps
    getInfoLogger("Finite differences").info(f"eps = {_eps:.2e}")
    if accuracy not in [0, 1, 2, 3]:
        raise ValueError("The accuracy should be 0, 1, 2 or 3!")
    dd = np.array([2.0, 12.0, 60.0, 840.0])

    fd_params = FDParams(
        x0=x0.ravel(),
        shape=x0.shape,
        inner_steps=2 * (accuracy + 1),
        coeff=[
            [1.0, -1.0],
            [1, -8.0, 8.0, -1.0],
            [-1, 9, -45.0, 45.0, -9.0, 1.0],
            [3.0, -32.0, 168.0, -672.0, 672.0, -168.0, 32.0, -3.0],
        ],
        coeff2=[
            [1.0, -1.0],
            [-2.0, -1.0, 1.0, 2.0],
            [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            [-4, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0],
        ],
        fm=fm,
        fm_args=fm_args if fm_args is not None else [],
        fm_kwargs=fm_kwargs if fm_kwargs is not None else {},
        dd_val=dd[accuracy] * _eps,
        accuracy=accuracy,
        eps=_eps,
        comptor_kw=comptor_kw,
    )

    def get_fd_params() -> Generator:
        while True:
            yield fd_params

    results: List[NDArrayFloat] = []
    # Single worker (no multi-processing)
    if max_workers == 1:
        for i in range(x0.size):
            results.append(approximate_elt_jac(i, fd_params))
    # Multi-processing enabled
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    approximate_elt_jac,
                    range(x0.size),
                    get_fd_params(),
                )
            )

    # (*output shape, *input shape) -> we do not flatten
    return np.stack(results, axis=-1).reshape(*results[0].shape, *fd_params.shape)


def finite_gradient(
    x: np.ndarray,
    fm: Callable,
    fm_args: Optional[Sequence[Any]] = None,
    fm_kwargs: Optional[Dict[str, Any]] = None,
    accuracy: int = 0,
    eps: Optional[float] = None,
    max_workers: int = 1,
    comptor_kw: Optional[str] = None,
) -> NDArrayFloat:
    r"""
    Compute the gradient by finite difference.

    The gradient is computed by using Taylor Series. For instance, if
    accurcy = 1, we use 4 points, which means that
    we take the Taylor series of $f$ around $a = x_j$ and compute the series
    at $x = x_{j-2}, x_{j-1}, x_{j+1}, x_{j+2}$.

    .. math::
        \begin{eqnarray*}
        f(x_{j-2}) &=& f(x_j) - 2hf^{\prime}(x_j) + \frac{4h^2f''(x_j)}{2} -
        \frac{8h^3f'''(x_j)}{6} + \frac{16h^4f''''(x_j)}{24}
        - \frac{32h^5f'''''(x_j)}{120} + \cdots\\
        f(x_{j-1}) &=& f(x_j) - hf^{\prime}(x_j) + \frac{h^2f''(x_j)}{2}
        - \frac{h^3f'''(x_j)}{6} + \frac{h^4f''''(x_j)}{24}
        - \frac{h^5f'''''(x_j)}{120} + \cdots\\
        f(x_{j+1}) &=& f(x_j) + hf^{\prime}(x_j) + \frac{h^2f''(x_j)}{2}
        + \frac{h^3f'''(x_j)}{6} + \frac{h^4f''''(x_j)}{24}
        + \frac{h^5f'''''(x_j)}{120} + \cdots\\
        f(x_{j+2}) &=& f(x_j) + 2hf^{\prime}(x_j) + \frac{4h^2f''(x_j)}{2}
        + \frac{8h^3f'''(x_j)}{6} + \frac{16h^4f''''(x_j)}{24}
        + \frac{32h^5f'''''(x_j)}{120} + \cdots
        \end{eqnarray*}

    To get the $h^2, h^3$, and $h^4$ terms to cancel out, we can compute

    .. math::
        f(x_{j-2}) - 8f(x_{j-1}) + 8f(x_{j-1}) - f(x_{j+2})
        = 12hf^{\prime}(x_j) - \frac{48h^5f'''''(x_j)}{120}

    which can be rearranged to

    .. math::
        f^{\prime}(x_j) = \frac{f(x_{j-2}) - 8f(x_{j-1})
        + 8f(x_{j-1}) - f(x_{j+2})}{12h} + O(h^4).

    This formula is a better approximation for the derivative at $x_j$
    than the central difference formula, but requires twice as many
    calculations.

    Parameters
    ----------
    x : np.ndarray
        The input parameters array.
    fm : Callable
        Forward model.
    fm_args: Tuple[Any]
        Positional arguments for the forward model.
    fm_kwargs : Dict[Any, Any]
        Keyword arguments for the forward model.
    accuracy : int, optional
        Number of points to use for the finite difference approximation.
        Possible values are 0 (2 points), 1 (4 points), 2 (6 points),
        3 (4 points). The default is 0 which corresponds to the central
        difference scheme (2 points).
    eps: float, optional
        The epsilon for the computation (h). By default, it take 1e-6 times
        the maximum absolute value of the input data.

        .. :math:

            \delta s = \begin{cases}
            \epsilon^{1/3} min\Big(max\left(\lvert s\rvert\right),
            1\Big) & \text{if} \; max\left(\lvert s\rvert\right) > 0\\
            \epsilon^{1/3} & \text{otherwise}
            \end{cases}

    max_workers: int
        Number of workers used. If different from one, the calculation relies on
        multi-processing to decrease the computation time. The default is 1.
    ppe_kwargs: Dict[str, Any]
        Keywords arguments for the ProcessPoolExecutor instance. This is used only if
        max_workers is above 1.

    Returns
    -------
    NDArrayFloat
        The finite difference gradient vector.
    """
    return finite_jacobian(
        x, fm, fm_args, fm_kwargs, accuracy, eps, max_workers, comptor_kw=comptor_kw
    ).reshape(x.shape)
