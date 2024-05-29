"""Provide optimization utilities."""

import numpy as np
from scipy.optimize import LbfgsInvHessProduct

from lbfgsb.types import NDArrayFloat


def extract_hess_inv_diag(hess_inv: LbfgsInvHessProduct) -> NDArrayFloat:
    """
    Extract efficiently the diagonal of the L-BFGS approximate inverse Hessian.

    It relies on the linear operator `matvec` operation and consequenlty does not
    require to build the dense matrix which is much longer and generally untractable
    for large-scale problems.

    Parameters
    ----------
    hess_inv : LbfgsInvHessProduct
        Linear operator for the L-BFGS approximate inverse Hessian.

    Returns
    -------
    NDArrayFloat
        The diagonal of the L-BFGS approximated inverse Hessian.
    """
    n_params = hess_inv.shape[0]
    hess_inv_diag = np.zeros(n_params)
    for i in range(n_params):
        v = np.zeros(n_params)
        v[i] = 1.0
        hess_inv_diag[i] = hess_inv.matvec(v)[i]
    return hess_inv_diag
