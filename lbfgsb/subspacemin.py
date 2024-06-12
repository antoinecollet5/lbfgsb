"""
Subspace minimization procedure of the L-BFGS-B algorithm,
mainly for internal use.

The target of subspace minimization is to minimize the quadratic function m(x)
over the free variables, subject to the bound condition.
Free variables stand for coordinates that are not at the boundary in xcp,
the generalized Cauchy point.

In the classical implementation of L-BFGS-B [1], the minimization is done by first
ignoring the box constraints, followed by a line search.

TODO: Our implementation is
an exact minimization subject to the bounds, based on the BOXCQP algorithm [2].

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
from scipy.sparse import lil_matrix, spmatrix

from lbfgsb.bfgsmats import LBFGSB_MATRICES, bmv
from lbfgsb.types import NDArrayFloat, NDArrayInt


def get_freev(
    x_cp: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    iter: int,
    free_vars_old: Optional[NDArrayInt] = None,
    iprint: int = -1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[NDArrayInt, spmatrix, spmatrix]:
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

    nb_free_vars: int = free_vars.size
    nb_active_vars: int = active_vars.size

    # See section 5 of [1]: We define Z to be the (n , t) matrix whose columns are
    # unit vectors (i.e., columns of the identity matrix) that span the subspace of the
    # free variables at zc.Similarly A denotes the (n, (n- t)) matrix of active
    # constraint gradients at zc,which consists of n - t unit vectors.
    # Note that A^{T}Z = 0 and that  AA^T + ZZ^T == I.

    # We use sparse formats to save memory and get faster matrix products
    Z = lil_matrix((n, nb_free_vars))
    A = lil_matrix((n, nb_active_vars))
    # Affect one
    Z[free_vars, np.arange(nb_free_vars)] = 1
    A[active_vars, np.arange(nb_active_vars)] = 1

    # Test: we should have Z @ Z.T + A @ A.T == I

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

    return free_vars, Z.tocsc(), A.tocsc()


def form_k(
    Z: spmatrix,
    A: spmatrix,
    WTZ: NDArrayFloat,
    mats: LBFGSB_MATRICES,
    is_assert_correct: bool = True,
) -> NDArrayFloat:
    """ """
    # Construct K = M^{-1}(I - 1/theta M WT Z @ ZT @ W))
    K = form_k_from_za(Z, A, mats)
    if is_assert_correct:
        K_wm = form_k_from_wm(WTZ, mats.invMfactors, mats.theta)
        np.testing.assert_allclose(K, K_wm, atol=1e-8)
    return K


def form_k_from_za(
    Z: spmatrix,
    A: spmatrix,
    mats: LBFGSB_MATRICES,
    logger: Optional[logging.Logger] = None,
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
    if Z.shape[0] == 0:
        YTZZTY = np.zeros((mats.Y.shape[1], mats.Y.shape[1]))
        STZZTY = np.zeros((mats.Y.shape[1], mats.Y.shape[1]))
    else:
        YTZZTY = mats.Y.T @ Z @ Z.T @ mats.Y
        STZZTY = mats.S.T @ Z @ Z.T @ mats.Y
    if A.shape[0] == 0:
        STAATS = np.zeros((mats.S.shape[1], mats.S.shape[1]))
    else:
        STAATS = mats.S.T @ A @ A.T @ mats.S

    m = mats.L.shape[0]
    K = np.zeros((m * 2, m * 2))

    K[:m, :m] = -mats.D - (1 / mats.theta) * YTZZTY
    K[:m, m:] = (mats.L - STZZTY).T
    K[m:, :m] = mats.L - STZZTY
    K[m:, m:] = mats.theta * STAATS

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
        _description_
    invMfactors : Tuple[NDArrayFloat, NDArrayFloat]
        _description_
    theta : float
        _description_

    Returns
    -------
    NDArrayFloat
        _description_
    """
    # Instead we build K directly as M^{-1}(I - 1/theta M WT Z @ ZT @ W))
    K = invMfactors[0] @ invMfactors[1]
    N = -1 / theta * bmv(invMfactors, WTZ.dot(np.transpose(WTZ)))
    np.fill_diagonal(N, N.diagonal() + 1)
    return K @ N


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
    X : Deque[NDArrayFloat]
        _description_
    G : Deque[NDArrayFloat]
        _description_
    Z : spmatrix
        _description_
    A : spmatrix
        _description_
    WTZ : NDArrayFloat
        _description_
    invMfactors : Tuple[NDArrayFloat, NDArrayFloat]
        _description_
    theta : float
        _description_
    is_assert_correct : bool, optional
        _description_, by default True

    Returns
    -------
    Optional[NDArrayFloat]
        _description_
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

    # Form L, the lower part of LL' = D+Y' ZZ'Y/theta
    L11 = sp.linalg.cholesky(K11, lower=True, overwrite_a=False)

    # then form L^-1(-L_a'+R_z') in the (1,2) block.
    L12 = sp.linalg.solve_triangular(L11, K12, lower=True, trans="N")

    # Form L22 from S'AA'S*theta + (L^-1(-L_a'+R_z'))'L^-1(-L_a'+R_z')
    L22 = sp.linalg.cholesky(K22 + L12.T @ L12, lower=True)

    # LK is a lower triangle of the matrix factorization LK @ E @ LK.T
    LK = np.hstack([np.vstack([L11, L12.T]), np.vstack([np.zeros(L12.shape), L22])])

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
    Z: spmatrix,
    A: spmatrix,
    c: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    mats: LBFGSB_MATRICES,
    n_iterations: int,
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
        TODO.
    Z: spmatrix
        Warning: it has shape (n, t)

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

    if len(free_vars) == 0:
        return xc

    # Same as W.T.dot(Z) but numpy does not handle correctly
    # numpy_array.dot(sparce_matrix), so we give the responsibility to the
    # sparse matrix
    # Note that here, Z is suppose to have a shape (t, n) with t the number
    # of free_vars and n the number of variables.
    # WTZ = W.T.dot(Z.todense()) works but this is much less efficient
    WTZ = Z.T.dot(mats.W).T

    r = grad + mats.theta * (xc - x)
    # At iter 0, M is [[0.0]] and so is invMfactors
    if n_iterations != 0:
        r -= mats.W.dot(bmv(mats.invMfactors, c))

    rHat = [r[ind] for ind in free_vars]
    v = WTZ.dot(rHat)

    # Factorization of M^{-1}(I - 1/theta M WT Z @ ZT @ W))
    if n_iterations != 0:
        K = form_k(Z, A, WTZ, mats)
        # The assertion includes minor overhead
        LK: Optional[NDArrayFloat] = factorize_k(K, is_assert_correct=True)
    else:
        LK = None

    if LK is not None:
        # LK is the lowest triangle of the cholesky factorization
        # of (I - 1/theta M WT Z @ ZT @ W)^{-1} M.
        v = sp.linalg.solve_triangular(LK, v, lower=True)
        v[: int(LK.shape[0] / 2)] *= -1
        v = sp.linalg.solve_triangular(LK.T, v, lower=False)
    else:
        # This is less efficient but it should only happen if LK is None, i.e., at
        # iteration 0
        if n_iterations != 0:
            v = bmv(mats.invMfactors, v)
            N = -bmv(mats.invMfactors, invThet * WTZ.dot(np.transpose(WTZ)))
        else:
            M = mats.invMfactors[0] @ mats.invMfactors[1]
            v = M.dot(v)
            N = -M.dot(invThet * WTZ.dot(np.transpose(WTZ)))
        # Add the identity matrix: this is the same as N = np.eye(N.shape[0]) - M.dot(N)
        # but much faster
        np.fill_diagonal(N, N.diagonal() + 1)
        v = np.linalg.solve(N, v)

    # Careful, there is an error in the original paper (the negative sign is
    # missing) !
    dHat = -invThet * (rHat + invThet * np.transpose(WTZ).dot(v))

    # We can then backtrack towards the feasible region, if necessary, to obtain
    # alpha a positive scalar between 0 and 1 -> Eq (5.8)
    mask = dHat != 0

    alpha_star = min(
        1.0,
        np.nanmin(
            np.where(
                dHat[mask] > 0, (ub - xc)[free_vars][mask], (lb - xc)[free_vars][mask]
            )
            / dHat[mask]
            if dHat[mask].size != 0
            else 1.0
        ),
    )
    # Eq (5.2) -> update free variables only
    return xc + alpha_star * Z @ dHat
