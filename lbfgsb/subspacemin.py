# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

"""
Subspace minimization procedure of the L-BFGS-B algorithm,
mainly for internal use.

The target of subspace minimization is to minimize the quadratic function m(x)
over the free variables, subject to the bound condition.
Free variables stand for coordinates that are not at the boundary in xcp,
the generalized Cauchy point.

In the classical implementation of L-BFGS-B [1], the minimization is done by first
ignoring the box constraints, followed by a line search.

Functions
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    get_freev
    form_k_from_wm
    form_k_from_za
    factorize_k
    subspace_minimization

Reference:
[1] R. H. Byrd, P. Lu, and J. Nocedal (1995). A limited memory algorithm for bound
constrained optimization.
[2] C. Voglis and I. E. Lagaris (2004). BOXCQP: An algorithm for bound constrained
convex quadratic problems.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import scipy as sp

from lbfgsb._numba_helpers import njit
from lbfgsb.bfgsmats import LBFGSB_MATRICES, bmv, bmv_numba
from lbfgsb.types import NDArrayFloat, NDArrayInt


def get_freev(
    x_cp: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    iter: int,
    free_vars_old: Optional[NDArrayInt] = None,
    iprint: int = -1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[NDArrayInt, NDArrayInt]:
    """
    Get the free variables and build sparse Z and A matrices.

    Parameters
    ----------
    x_cp : NDArrayFloat
        Generalized cauchy point.
    lb : NDArrayFloat
        Lower bounds.
    ub : NDArrayFloat
        Upper bounds.
    free_vars_old : NDArrayInt
        Free variables at x_cp at the previous iteration.
    iter : int
        Iteration number.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint >= 99``   print details of every iteration except n-vectors;
    logger: Optional[Logger], optional
        :class:`logging.Logger` instance. If None, nothing is displayed, no matter the
        value of `iprint`, by default None.

    Returns
    -------
    Tuple[NDArrayInt, spmatrix, spmatrix]
        The free variables and sparse matrices Z and A.
    """
    # number of variables
    n: int = x_cp.size

    # Array of free variable and active variable indices (from 0 to n-1)
    free_vars: NDArrayInt = ((x_cp != ub) & (x_cp != lb)).nonzero()[0]
    active_vars: NDArrayInt = (
        ~np.isin(np.arange(n), free_vars)  # type: ignore
    ).nonzero()[0]

    # Some display
    # 1) Indicate which variable is leaving the free variables and which is
    # entering the free variables -> Not for the first iteration
    if iprint > 100 and iter > 0 and free_vars_old is not None and logger is not None:
        # Variables leaving the free variables
        leaving_vars = active_vars[np.isin(active_vars, free_vars_old)]
        logger.info(f"Variables leaving the free variables set = {leaving_vars}")
        entering_vars = free_vars[~np.isin(free_vars, free_vars_old)]

        logger.info(f"Variables entering the free variables set = {entering_vars}")
        logger.info(
            f"N variables leaving = {leaving_vars.size} \t,"
            f" N variables entering = {entering_vars.size}"
        )
    # 2) Display the total of free variables at x_cp
    if iprint > 99 and logger is not None:
        logger.info(f"{free_vars.size} variables are free at GCP, iter = {iter + 1}")

    return free_vars, active_vars


def form_k(
    free_vars: NDArrayInt,
    active_vars: NDArrayInt,
    WTZ: NDArrayFloat,
    mats: LBFGSB_MATRICES,
    is_assert_correct: bool = True,
) -> NDArrayFloat:
    """ """
    # Construct K = M^{-1}(I - 1/theta M WT Z @ ZT @ W))
    K = form_k_from_za(
        free_vars, active_vars, mats.Y, mats.S, mats.D, mats.L, mats.theta
    )
    if is_assert_correct:
        K_wm = form_k_from_wm(WTZ, mats.invMfactors, mats.theta)
        np.testing.assert_allclose(K, K_wm, atol=1e-8)
    return K


def form_k_from_za(
    free_vars: NDArrayInt,
    active_vars: NDArrayInt,
    Y: NDArrayFloat,
    S: NDArrayFloat,
    D: NDArrayFloat,
    L: NDArrayFloat,
    theta: float,
) -> NDArrayFloat:
    r"""
    Form the matrix K.

    The matrix K is defined by

    .. math::
        \mathbf{M}^{-1} \mathbf{K} = \left(\mathbf{I} - \dfrac{1}{\theta}
            \mathbf{MW}^{\mathrm{T}}
            \mathbf{ZZ}^{\mathrm{T}}\mathbf{W}\right)  =
            \begin{bmatrix} -\mathbf{D} - \dfrac{1}{\theta} \mathbf{Y}^{\mathrm{T}}
            \mathbf{ZZ}^{\mathrm{T}}\mathbf{Y} & \mathbf{L}_A^{\mathrm{T}}
            - \mathbf{R}_Z^{\mathrm{T}} \\ \mathbf{L}_A - \mathbf{R}_Z & \theta
            \mathbf{S}^{\mathrm{T}}\mathbf{AA}^{\mathrm{T}}\mathbf{S} \end{bmatrix}

    Parameters
    ----------
    """
    if len(free_vars) == 0:
        YTZZTY = np.zeros((Y.shape[1], Y.shape[1]))
        STZZTY = np.zeros((Y.shape[1], Y.shape[1]))
    else:
        ZZTY = np.zeros(np.shape(Y), dtype=np.float64)
        ZZTY[free_vars, :] = Y[free_vars, :]
        YTZZTY = Y.T @ ZZTY
        STZZTY = S.T @ ZZTY

    if len(active_vars) == 0:
        STAATS = np.zeros((S.shape[1], S.shape[1]))
    else:
        AATS = np.zeros(np.shape(S), dtype=np.float64)
        AATS[active_vars, :] = S[active_vars, :]
        STAATS = S.T @ AATS

    m = L.shape[0]
    K = np.zeros((m * 2, m * 2))

    K[:m, :m] = -D - (1.0 / theta) * YTZZTY
    K[:m, m:] = (L - STZZTY).T
    K[m:, :m] = L - STZZTY
    K[m:, m:] = theta * STAATS

    return K


def form_k_from_wm(
    WTZ: NDArrayFloat,
    invMfactors: Tuple[NDArrayFloat, NDArrayFloat],
    theta: float,
) -> NDArrayFloat:
    r"""
    Form the matrix K.

    The matrix K is defined as

    .. math::

        mathbf{K} = \mathbf{M}^{-1} \left(\mathbf{I} -
        \dfrac{1}{\theta} \mathbf{MW}^{\mathrm{T}}
        \mathbf{ZZ}^{\mathrm{T}}\mathbf{W}\right)

    Parameters
    ----------
    WTZ : NDArrayFloat
        Matrix WTZ (TODO: add the shape.)
    invMfactors : Tuple[NDArrayFloat, NDArrayFloat]
        LU factorization of the inverse of the middle matrix M.
    theta : float
        L-BFGS float parameter (multiply the identity matrix).

    Returns
    -------
    NDArrayFloat
        Matrix K.
    """
    # Instead we build K directly as M^{-1}(I - 1/theta M WT Z @ ZT @ W))
    K = invMfactors[0] @ invMfactors[1]
    N = -1 / theta * bmv(invMfactors, WTZ.dot(np.transpose(WTZ)))
    np.fill_diagonal(N, N.diagonal() + 1)
    return K @ N


@njit(cache=True)
def solve_triangular_numba(
    L: NDArrayFloat, v: NDArrayFloat, lower: bool = True
) -> NDArrayFloat:
    """
    Numba replacement for:
        solve_triangular(U, lower=False)
    """
    n = v.size
    y = np.empty(n)

    if lower:
        # Forward solve: L y = v
        for i in range(n):
            s = v[i]
            for j in range(i):
                s -= L[i, j] * y[j]
            y[i] = s / L[i, i]
    else:
        # Backward solve: U p = y
        for i in range(n - 1, -1, -1):
            s = v[i]
            for j in range(i + 1, n):
                s -= L[i, j] * y[j]
            y[i] = s / L[i, i]

    return y


def factorize_k(
    K: NDArrayFloat,
    is_assert_correct: bool = True,
) -> Optional[NDArrayFloat]:
    """
    Return the L with LEL^T factorization of the indefinite matrix K.

    K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
        [L_a -R_z           theta*S'AA'S ]

    where

    E = [-I  0]
        [ 0  I]

    Parameters
    ----------
    K: NDArrayFloat
        Indefinite matrix to factorize.
    is_assert_correct : bool, optional
        Whether to test the correctness of the factorization, by default True

    Returns
    -------
    Optional[NDArrayFloat]
        LK, the lower triangle of the matrix factorization K = LK @ E @ LK.T
    """
    # The factorization only makes sense if K is at least (2, 2).
    if K.size < 4:
        assert K.size == 1
        return np.sqrt(K)

    # Extract the subblocks of K with K12 = K21.T (K is symmetric)
    m = int(K.shape[0] / 2)
    K11 = -K[:m, :m]
    K12 = -K[:m, m:]
    K22 = K[m:, m:]

    # LK is a lower triangle of the matrix factorization LK @ E @ LK.T
    # Initiate the array
    LK = np.zeros((2 * m, 2 * m), dtype=np.float64)

    # Form L, the lower part of LL' = D+Y' ZZ'Y/theta
    L11 = sp.linalg.cholesky(K11, lower=True, overwrite_a=False)
    # Top-left
    LK[:m, :m] = L11

    # then form L^-1(-L_a'+R_z') in the (1,2) block.
    L12 = sp.linalg.solve_triangular(L11, K12, lower=True, trans="N")
    # Top-right
    LK[m:, :m] = L12.T

    # Form L22 from S'AA'S*theta + (L^-1(-L_a'+R_z'))'L^-1(-L_a'+R_z')
    # Bottom-right
    LK[m:, m:] = sp.linalg.cholesky(K22 + L12.T @ L12, lower=True)

    # Test the factorization
    if is_assert_correct:
        E = np.identity(n=2 * m)
        E[:m, :m] *= -1
        np.testing.assert_allclose(LK @ E @ LK.T, K, atol=1e-8)
    return LK


def subspace_minimization(
    x: NDArrayFloat,
    xc: NDArrayFloat,
    free_vars: NDArrayInt,
    active_vars: NDArrayInt,
    c: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    mats: LBFGSB_MATRICES,
    is_check_factorizations: bool = False,
    is_use_numba_jit: bool = False,
) -> NDArrayFloat:
    r"""
    Computes an approximate solution of the subspace problem.

    This is following section 5.1 in Byrd et al. (1995).

    .. math::

        \begin{aligned}
            \min& &\langle r, (x-xcp)\rangle + 1/2 \langle x-xcp, B (x-xcp)\rangle\\
            \text{s.t.}& &l<=x<=u\\
                       & & x_i=xcp_i \text{for all} i \in A(xcp)
        \end{aligned}

    along the subspace unconstrained Newton direction :math:`d = -(Z'BZ)^(-1) r`.

    Parameters
    ----------
    x : NDArrayFloat
        Starting point for the GCP computation
    xc : NDArrayFloat
        Cauchy point.
    c : NDArrayFloat
        W^T(xc-x), computed with the Cauchy point.
    grad : NDArrayFloat
        Gradient of f(x). grad must be a nonzero vector.
    lb : NDArrayFloat
        Lower bound vector.
    ub : NDArrayFloat
        Upper bound vector.
    mats: LBFGSB_MATRICES
        Wrapper for BFGS matrices.
    is_check_factorizations: bool
        Whether to check the different factorizations performed. The default is False.
    is_use_numba_jit: bool
        Whether to use `numba` just-in-time compilation to speed-up the computation
        intensive part of the algorithm. The default is False.
        .. versionadded:: 1.0

    Returns
    -------
    NDArrayFloat
        xbar

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing, 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (2011),
      ACM Transactions on Mathematical Software, 38, 1.
    """
    # Direct primal method

    invThet = 1.0 / mats.theta

    # d = (1/theta)r + (1/theta*2) Z'WK^(-1)W'Z r.

    if free_vars.size == 0:
        return xc

    # ------------------------------------------------------------------
    # Pre-slice everything ONCE (major speedup)
    # ------------------------------------------------------------------
    xc_free = xc[free_vars]
    ub_free = ub[free_vars]
    lb_free = lb[free_vars]

    # Same as W.T.dot(Z) but numpy does not handle correctly
    # numpy_array.dot(sparce_matrix), so we give the responsibility to the
    # sparse matrix
    # Note that here, Z is suppose to have a shape (t, n) with t the number
    # of free_vars and n the number of variables.
    # WTZ = W.T.dot(Z.todense()) works but this is much less efficient
    W_free = mats.W[free_vars, :]
    WTZ = W_free.T  # shape (m, t)

    r = grad + mats.theta * (xc - x)
    # At iter 0, M is [[0.0]] and so is invMfactors
    if mats.use_factor:
        r -= mats.W.dot(bmv(mats.invMfactors, c))

    rHat = r[free_vars]
    v = WTZ @ rHat

    # Factorization of M^{-1}(I - 1/theta M WT Z @ ZT @ W))
    if mats.use_factor:
        K = form_k(
            free_vars,
            active_vars,
            WTZ,
            mats,
            is_assert_correct=is_check_factorizations,
        )
        # The assertion includes minor overhead
        LK: Optional[NDArrayFloat] = factorize_k(
            K, is_assert_correct=is_check_factorizations
        )
    else:
        LK = None

    if LK is not None:
        # LK is the lowest triangle of the cholesky factorization
        # of (I - 1/theta M WT Z @ ZT @ W)^{-1} M.
        if is_use_numba_jit:
            v = solve_triangular_numba(LK, v, lower=True)
        else:
            v = sp.linalg.solve_triangular(LK, v, lower=True)
        v[: int(LK.shape[0] / 2)] *= -1
        if is_use_numba_jit:
            v = solve_triangular_numba(LK.T, v, lower=False)
        else:
            v = sp.linalg.solve_triangular(LK.T, v, lower=False)
    else:
        # This is less efficient but it should only happen if LK is None, i.e., at
        # iteration 0
        if mats.use_factor:
            if is_use_numba_jit:
                v = bmv_numba(*mats.invMfactors, v)
                N = -bmv_numba(*mats.invMfactors, invThet * (WTZ @ WTZ.T))
            else:
                v = bmv(mats.invMfactors, v)
                N = -bmv(mats.invMfactors, invThet * (WTZ @ WTZ.T))
        else:
            M = mats.invMfactors[0] @ mats.invMfactors[1]
            v = M @ v
            N = -M @ (invThet * WTZ @ WTZ.T)
        # Add the identity matrix: this is the same as N = np.eye(N.shape[0]) - M.dot(N)
        # but much faster
        np.fill_diagonal(N, N.diagonal() + 1)
        v = np.linalg.solve(N, v)

    # Careful, there is an error in the original paper (the negative sign is
    # missing) !
    dHat = -invThet * (rHat + invThet * (WTZ.T @ v))

    # We can then backtrack towards the feasible region, if necessary, to obtain
    # alpha a positive scalar between 0 and 1 -> Eq (5.8)
    mask = dHat != 0.0
    if mask.any():
        d = dHat[mask]
        xc_m = xc_free[mask]

        step = np.empty_like(d)

        pos = d > 0.0
        step[pos] = (ub_free[mask][pos] - xc_m[pos]) / d[pos]
        step[~pos] = (lb_free[mask][~pos] - xc_m[~pos]) / d[~pos]

        alpha_star = min(1.0, step.min())
    else:
        alpha_star = 1.0

    # Eq (5.2) -> update free variables only
    xc[free_vars] = xc_free + alpha_star * dHat
    return xc
