# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
from lbfgsb import extract_hess_inv_diag
from numpy.testing import assert_allclose
from scipy.optimize import minimize


def test_1():
    def f(x):
        return x**4, 4 * x**3

    for gtol in [1e-8, 1e-12, 1e-20]:
        for maxcor in range(20, 35):
            result = minimize(
                fun=f,
                jac=True,
                method="L-BFGS-B",
                x0=20,
                options={"gtol": gtol, "maxcor": maxcor},
            )

            H1 = result.hess_inv(np.array([1])).reshape(1, 1)
            H2 = result.hess_inv.todense()

            assert_allclose(H1, H2)
            assert_allclose(extract_hess_inv_diag(result.hess_inv), np.diagonal(H1))


def test_2():
    H0 = [[3, 0], [1, 2]]

    def f(x):
        return np.dot(x, np.dot(scipy.linalg.inv(H0), x))

    result1 = minimize(fun=f, method="L-BFGS-B", x0=[10, 20])
    result2 = minimize(fun=f, method="BFGS", x0=[10, 20])

    H1 = result1.hess_inv.todense()

    H2 = np.vstack(
        (result1.hess_inv(np.array([1, 0])), result1.hess_inv(np.array([0, 1])))
    )

    assert_allclose(
        result1.hess_inv(np.array([1, 0]).reshape(2, 1)).reshape(-1),
        result1.hess_inv(np.array([1, 0])),
    )
    assert_allclose(H1, H2)
    assert_allclose(H1, result2.hess_inv, rtol=1e-2, atol=0.03)
    assert_allclose(
        np.diagonal(H1), extract_hess_inv_diag(result1.hess_inv), rtol=1e-2, atol=0.03
    )
