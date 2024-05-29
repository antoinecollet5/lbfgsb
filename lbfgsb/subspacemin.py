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

    freev
    form_k_from_xgza
    form_k_from_wm
    formk
    direct_primal_subspace_minimization

Reference:
[1] R. H. Byrd, P. Lu, and J. Nocedal (1995). A limited memory algorithm for bound
constrained optimization.
[2] C. Voglis and I. E. Lagaris (2004). BOXCQP: An algorithm for bound constrained
convex quadratic problems.
"""

from typing import Deque, Optional, Tuple

import numpy as np
import scipy as sp
from scipy.sparse import lil_matrix, spmatrix

from lbfgsb.bfgsmats import bmv
from lbfgsb.types import NDArrayFloat, NDArrayInt


def freev(
    x_cp: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    iprint: int,
    iter: int,
    free_vars_old: Optional[NDArrayInt] = None,
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
    iprint : int
        Level of display.
    iter : int
        Iteration number.

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
    if iprint > 100 and iter > 0 and free_vars_old is not None:
        # Variables leaving the free variables
        leaving_vars = active_vars[np.isin(active_vars, free_vars_old)]
        print(f"Variables leaving the free variables set = {leaving_vars}")
        entering_vars = free_vars[~np.isin(free_vars, free_vars_old)]
        print(f"Variables entering the free variables set = {entering_vars}")
        print(
            f"N variables leaving = {leaving_vars.size} \t,"
            f" N variables entering = {entering_vars.size}"
        )
    # 2) Display the total of free variables at x_cp
    if iprint > 99:
        print(f"{free_vars.size} variables are free at GCP, iter = {iter + 1}")

    return free_vars, Z.tocsc(), A.tocsc()


def form_k_from_xgza(
    X: Deque[NDArrayFloat],
    G: Deque[NDArrayFloat],
    Z: spmatrix,
    A: spmatrix,
    theta: float,
) -> NDArrayFloat:
    """
    Form the matrix K.

    The matrix K is defined by:

    K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
        [L_a -R_z           theta*S'AA'S ]

    Parameters
    ----------
    """
    # This was the old approach that did not work:
    # form S and Y
    S = np.diff(np.array(X), axis=0).T
    Y = np.diff(np.array(G), axis=0).T
    D: NDArrayFloat = np.diag(np.diag(S.T @ Y))

    # 1) RZ is the upper triangular part of S'ZZ'Y.
    if Z.size == 0:
        YTZZTY = np.zeros((Y.shape[1], Y.shape[1]))
        RZ = YTZZTY.copy()
    else:
        YTZZTY = Y.T @ Z @ Z.T @ Y
        print(YTZZTY)
        RZ = sp.linalg.cholesky(
            YTZZTY, lower=False, overwrite_a=False
        )  # np.triu(YTZZTY)

    # 2) LA is the strict lower triangle of S^{T}AA^{T}S
    if A.size == 0:
        STAATS = np.zeros((S.shape[1], S.shape[1]))
        LA = STAATS.copy()
    else:
        STAATS = S.T @ A @ A.T @ S
        LA = sp.linalg.cholesky(
            STAATS, lower=True, overwrite_a=False
        )  # np.tril(STAATS, -1)

    m = LA.shape[0]
    K = np.zeros((m * 2, m * 2))

    K[:m, :m] = -D - (1 / theta) * YTZZTY
    K[:m, m:] = (LA - RZ).T
    K[m:, :m] = LA - RZ
    K[m:, m:] = theta * STAATS

    return K


def form_k_from_wm(
    WTZ: NDArrayFloat,
    invMfactors: Tuple[NDArrayFloat, NDArrayFloat],
    theta: float,
) -> NDArrayFloat:
    """
    Form the matrix K.

    The matrix K is defined as M^{-1}(I - 1/theta M WT Z @ ZT @ W)).

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


def formk(
    X: Deque[NDArrayFloat],
    G: Deque[NDArrayFloat],
    Z: spmatrix,
    A: spmatrix,
    WTZ: NDArrayFloat,
    invMfactors: Tuple[NDArrayFloat, NDArrayFloat],
    theta: float,
    is_assert_correct: bool = True,
) -> Optional[NDArrayFloat]:
    """
    Form mk.

    Form  the LEL^T factorization of the indefinite matrix

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
    K = form_k_from_wm(WTZ, invMfactors, theta)

    # TODO: K2 is not exactly equal to what it should
    # We don't understand why for now...
    # K2 = form_k_from_xgza(X, G, Z, A, theta)
    # np.testing.assert_allclose(K, K2, atol=1e-8)
    # print(f"K = {K}")
    # print(f"K2 = {K2}")

    # The factorization only makes sense if K is at least (2, 2).
    if K.size < 4:
        return None

    m = int(K.shape[0] / 2)
    K11 = -K[:m, :m]
    K12 = -K[:m, m:]
    K22 = K[m:, m:]

    # print(f"K.shape = {K.shape}")
    # print(f"K11.shape = {K11.shape}")
    # print(f"K12.shape = {K12.shape}")
    # print(f"K22.shape = {K22.shape}")

    # Form L, the lower part of LL' = D+Y' ZZ'Y/theta
    L11 = sp.linalg.cholesky(K11, lower=True, overwrite_a=False)

    # then form L^-1(-L_a'+R_z') in the (1,2) block.
    L12 = sp.linalg.solve_triangular(L11, K12, lower=True, trans="N")

    # Form L22 from S'AA'S*theta + (L^-1(-L_a'+R_z'))'L^-1(-L_a'+R_z')
    L22 = sp.linalg.cholesky(K22 + L12.T @ L12, lower=True)

    # K is a lower triangle matrix
    LK = np.hstack([np.vstack([L11, L12.T]), np.vstack([np.zeros(L12.shape), L22])])

    # Test the factorization # TODO: create a specific function
    if is_assert_correct:
        K2 = np.hstack([np.vstack([-K11, -K12.T]), np.vstack([-K12, K22])])
        E = np.identity(n=K2.shape[0])
        E[: int(E.shape[0] / 2), : int(E.shape[0] / 2)] *= -1
        # print(f"K11 = {K11}")
        # print(f"K = {K}")
        # print(f"E = {E}")
        # print(f"LK = {LK}")
        np.testing.assert_allclose(LK @ E @ LK.T, K2, atol=1e-8)
        np.testing.assert_allclose(LK @ E @ LK.T, K, atol=1e-8)
    return LK


# There are three methods for this one and we need to find the correct one.
def direct_primal_subspace_minimization(
    X,
    G,
    x: NDArrayFloat,
    xc: NDArrayFloat,
    free_vars: NDArrayInt,
    Z: spmatrix,
    A: spmatrix,
    c: NDArrayFloat,
    grad: NDArrayFloat,
    lb: NDArrayFloat,
    ub: NDArrayFloat,
    W: NDArrayFloat,
    invMfactors: Tuple[NDArrayFloat, NDArrayFloat],
    theta: float,
    n_iterations: int,
) -> NDArrayFloat:
    r"""
    Computes an approximate solution of the subspace problem.

    This is following section 5.1 in Byrd et al. (1995).

    .. math::
        :nowrap:

       \[\begin{aligned}
            \min& &\langle r, (x-xcp)\rangle + 1/2 \langle x-xcp, B (x-xcp)\rangle\\
            \text{s.t.}& &l<=x<=u\\
                       & & x_i=xcp_i \text{for all} i \in A(xcp)
        \]

    along the subspace unconstrained Newton direction
    .. math:: $d = -(Z'BZ)^(-1) r.$

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
    W : NDArrayFloat
        Part of limited memory BFGS Hessian approximation.
    M : NDArrayFloat
        Part of limited memory BFGS Hessian approximation.
    theta : float
        Part of limited memory BFGS Hessian approximation.
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

    invThet = 1.0 / theta

    # d = (1/theta)r + (1/theta*2) Z'WK^(-1)W'Z r.

    if len(free_vars) == 0:
        return xc

    # Same as W.T.dot(Z) but numpy does not handle correctly
    # numpy_array.dot(sparce_matrix), so we give the responsibility to the
    # sparse matrix
    # Note that here, Z is suppose to have a shape (t, n) with t the number
    # of free_vars and n the number of variables.
    # WTZ = W.T.dot(Z.todense())
    WTZ = Z.T.dot(W).T

    r = grad + theta * (xc - x)
    # At iter 0, M is [[0.0]] and so is invMfactors
    if n_iterations != 0:
        r -= W.dot(bmv(invMfactors, c))

    rHat = [r[ind] for ind in free_vars]
    v = WTZ.dot(rHat)

    # Factorization of M^{-1}(I - 1/theta M WT Z @ ZT @ W))
    if n_iterations != 0:
        LK: Optional[NDArrayFloat] = formk(X, G, Z, A, WTZ, invMfactors, theta)
    else:
        LK = None

    if LK is not None:
        # LK is the lowest triangle of the cholesky factorization
        # of (I - 1/theta M WT Z @ ZT @ W)^{-1} M.
        v = sp.linalg.solve_triangular(LK, v, lower=True)
        v[: int(LK.shape[0] / 2)] *= -1
        v = sp.linalg.solve_triangular(LK.T, v, lower=False)
    else:
        if n_iterations != 0:
            v = bmv(invMfactors, v)
            N = -bmv(invMfactors, invThet * WTZ.dot(np.transpose(WTZ)))
        else:
            M = invMfactors[0] @ invMfactors[1]
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
