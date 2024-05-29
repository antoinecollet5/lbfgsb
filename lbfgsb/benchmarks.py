"""
Provide the following benchmark functions and their gradients.
- ackley
- griewank
- quadratic
- rastrigin
- rosenbrook
- sphere
- styblinski_tang
"""

import numpy as np

from lbfgsb.types import NDArrayFloat


def ackley(x: NDArrayFloat) -> float:
    """
    The Ackley function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Ackley function is to be computed.

    Returns
    -------
    float
        The value of the Ackley function.

    """
    x = np.asarray(x)
    ndim = x.size
    e = 2.7182818284590451
    sum1 = np.sqrt(1.0 / ndim * np.square(x).sum())
    sum2 = 1.0 / ndim * np.cos(2.0 * np.pi * x).sum()
    return 20.0 + e - 20.0 * np.exp(-0.2 * sum1) - np.exp(sum2)


def ackley_grad(x: NDArrayFloat) -> NDArrayFloat:
    """
    The gradient of the Ackley function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Ackley function is to be derivated.

    Returns
    -------
    NDArrayFloat
        The gradient of the Ackley function.

    """
    x = np.asarray(x)
    ndim = x.size
    square_sum = np.square(x).sum()
    return (
        4.0
        * x
        * np.sqrt(square_sum / ndim)
        * np.exp(-0.2 * np.sqrt(square_sum / ndim))
        / square_sum
    ) - 2.0 * np.pi / ndim * np.sin(2.0 * np.pi * x)


def griewank(x: NDArrayFloat) -> float:
    """
    The Griewank function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Griewank function is to be computed.

    Returns
    -------
    float
        The value of the Griewank function.

    """
    x = np.asarray(x)
    ndim = x.size
    sum1 = np.square(x).sum() / 4000.0
    prod1 = np.prod(np.cos(x / np.sqrt(np.arange(1, ndim + 1))))
    return 1.0 + sum1 - prod1


def griewank_grad(x: NDArrayFloat) -> NDArrayFloat:
    """
    The gradient of the Griewank function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Griewank function is to be derivated.

    Returns
    -------
    NDArrayFloat
        The gradient of the Griewank function.

    """
    x = np.asarray(x)
    ndim = x.size
    den = np.sqrt(np.arange(1, ndim + 1))
    return (
        x / 2000.0 + np.sin(x / den) * np.prod(np.cos(x / den)) / np.cos(x / den) / den
    )


def quartic(x: NDArrayFloat) -> float:
    """
    The Quartic function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Quartic function is to be computed.

    Returns
    -------
    float
        The value of the Quartic function.

    """
    x = np.asarray(x)
    ndim = x.size
    return (np.arange(1, ndim + 1) * np.power(x, 4)).sum()


def quartic_grad(x: NDArrayFloat) -> NDArrayFloat:
    """
    The gradient of the Quartic function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Quartic function is to be derivated.

    Returns
    -------
    NDArrayFloat
        The gradient of the Quartic function.

    """
    x = np.asarray(x)
    ndim = x.size
    return np.arange(1, ndim + 1) * 4 * np.power(x, 3)


def rastrigin(x: NDArrayFloat) -> float:
    """
    The Rastrigin function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rastrigin function is to be computed.

    Returns
    -------
    float
        The value of the Rastrigin function.

    """
    x = np.asarray(x)
    ndim = x.size
    sum1 = (np.square(x) - 10.0 * np.cos(2.0 * np.pi * x)).sum()
    return 10.0 * ndim + sum1


def rastrigin_grad(x: NDArrayFloat) -> NDArrayFloat:
    """
    The gradient of the Rastrigin function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rastrigin function is to be derivated.

    Returns
    -------
    NDArrayFloat
        The gradient of the Rastrigin function.

    """
    x = np.asarray(x)
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


def rosenbrock(x: NDArrayFloat) -> float:
    """
    The Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rosenbrock function is to be computed.

    Returns
    -------
    float
        The value of the Rosenbrock function.

    """
    x = np.asarray(x)
    sum1 = ((x[1:] - x[:-1] ** 2.0) ** 2.0).sum()
    sum2 = np.square(1.0 - x[:-1]).sum()
    return 100.0 * sum1 + sum2


def rosenbrock_grad(x: NDArrayFloat) -> NDArrayFloat:
    """
    The gradient of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rosenbrock function is to be derivated.

    Returns
    -------
    NDArrayFloat
        The gradient of the Rosenbrock function.
    """
    x = np.asarray(x)
    g = np.zeros(x.size)
    # derivation of sum1
    g[1:] += 100.0 * (2.0 * x[1:] - 2.0 * x[:-1] ** 2.0)
    g[:-1] += 100.0 * (-4.0 * x[1:] * x[:-1] + 4.0 * x[:-1] ** 3.0)
    # derivation of sum2
    g[:-1] += 2.0 * (x[:-1] - 1.0)
    return g


def sphere(x: NDArrayFloat) -> float:
    """
    The Sphere function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Sphere function is to be computed.

    Returns
    -------
    float
        The value of the Sphere function.

    """
    return np.square(x).sum()


def sphere_grad(x: NDArrayFloat) -> NDArrayFloat:
    """
    The gradient of the Sphere function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Sphere function is to be derivated.

    Returns
    -------
    NDArrayFloat
        The gradient of the Rosenbrock function.
    """
    return 2 * np.asarray(x)


def styblinski_tang(x: NDArrayFloat) -> float:
    """
    The Styblinski-Tang function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Styblinski-Tang function is to be computed.

    Returns
    -------
    float
        The value of the Styblinski-Tang function.

    """
    x = np.asarray(x)
    sum1 = (np.power(x, 4) - 16.0 * np.square(x) + 5.0 * x).sum()
    return 0.5 * sum1 + 39.16599 * x.size


def styblinski_tang_grad(x: NDArrayFloat) -> NDArrayFloat:
    """
    The gradient of the Styblinski-Tang function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Styblinski-Tang function is to be derivated.

    Returns
    -------
    NDArrayFloat
        The gradient of the Styblinski-Tang function.

    """
    x = np.asarray(x)
    return 2.0 * np.power(x, 3) - 16.0 * x + 2.5
