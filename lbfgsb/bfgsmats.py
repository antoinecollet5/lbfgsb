# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Antoine COLLET

"""
An *implicit* representation of the BFGS approximation to the Hessian matrix B

B = theta * I - W * M * W'
H = inv(B)

Classes
^^^^^^^

.. autosummary::
   :toctree: _autosummary

    LBFGSB_MATRICES

Functions
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    bmv
    form_invMfactors
    update_lbfgs_matrices
    update_X_and_G

Reference:
:cite:`nocedalUpdatingQuasiNewtonMatrices1980`
:cite:`byrdRepresentationsQuasiNewtonMatrices1994`
:cite:`byrdLimitedMemoryAlgorithm1995`
"""

import logging
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import scipy as sp

from lbfgsb._numba_helpers import njit
from lbfgsb.types import NDArrayFloat


class LBFGSB_MATRICES:
    """
    Represent the L-BFGS matrices.

    Attributes
    ----------
    S : NDArrayFloat
        # shape (n, m)
    Y : NDArrayFloat
        # shape (n, m)
    D : NDArrayFloat
        # shape (m, m)
    L : NDArrayFloat
        # shape (m, m)
    W : NDArrayFloat
        # shape (m, 2m)
    invMfactors: Tuple[NDArrayFloat, NDArrayFloat]
        # shape (2m, 2m)
    theta : float
        L-BFGS float parameter (multiply the identity matrix).
    """

    __slots__ = ["S", "Y", "D", "L", "W", "invMfactors", "theta"]

    def __init__(self, n: int) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        n : int
            Number of adjusted variables.
        """
        if n < 1:
            raise ValueError("n must be an integer > 0.")
        self.S: NDArrayFloat = np.zeros([n, 1])
        self.Y: NDArrayFloat = np.zeros([n, 1])
        self.D: NDArrayFloat = np.zeros([n, 1])
        self.L: NDArrayFloat = np.zeros([n, 1])
        self.W: NDArrayFloat = np.zeros([n, 1])
        self.invMfactors: Tuple[NDArrayFloat, NDArrayFloat] = (
            np.zeros([1, 1]),
            np.zeros([1, 1]),
        )
        self.theta: float = 1.0

    @property
    def use_factor(self) -> bool:
        """If the factors are null then matrix factorization will fail."""
        return self.invMfactors[0].size != 1 or self.invMfactors[0][0, 0] != 0


def bmv(
    invMfactors: Tuple[NDArrayFloat, NDArrayFloat], v: NDArrayFloat
) -> NDArrayFloat:
    """
    Return the product of the 2m x 2m middle matrix with a vector v.

    In the compact L-BFGS formula of B and a 2m vector `v`;
    it returns the product in `p`.

    Parameters
    ----------
    invMfactors : Tuple[NDArrayFloat, NDArrayFloat]
        LU factorization of the inverse of the middle matrix M.
    v : NDArrayFloat
        Vector with size 2m.

    Returns
    -------
    NDArrayFloat
        Product of the 2m x 2m middle matrix with a vector v.
    """
    # PART I: solve [  D^(1/2)      O ] [ p1 ] = [ v1 ]
    #               [ -L*D^(-1/2)   J ] [ p2 ]   [ v2 ].
    # sp.linalg.solve_triangular(invMfactors[0], v, lower=True)
    # PART II: solve [ -D^(1/2)   D^(-1/2)*L'  ] [ p1 ] = [ p1 ]
    #                [  0         J'           ] [ p2 ]   [ p2 ].
    return sp.linalg.solve_triangular(
        invMfactors[1],
        sp.linalg.solve_triangular(invMfactors[0], v, lower=True),
        lower=False,
    )


@njit(cache=True)
def bmv_numba(L: NDArrayFloat, U: NDArrayFloat, v: NDArrayFloat) -> NDArrayFloat:
    """
    Numba replacement for:
        solve_triangular(U, solve_triangular(L, v, lower=True), lower=False)
    """
    n = v.size
    y = np.empty(n)
    p = np.empty(n)

    # Forward solve: L y = v
    for i in range(n):
        s = v[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s / L[i, i]

    # Backward solve: U p = y
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= U[i, j] * p[j]
        p[i] = s / U[i, i]

    return p


def form_invMfactors(theta, STS, L, D) -> Tuple[NDArrayFloat, NDArrayFloat]:
    r"""
    Return upper triangle of the cholesky factorization of the inverse of M_k.

    This is defined in eq. (3.4) [1].

    Although Mk is not positive definite, but its inverse reads

    .. math::

        \mathbf{M}^{-1} = \begin{bmatrix} -\mathbf{D} & \mathbf{L}^{\mathrm{T}} \\
        \mathbf{L} & \theta \mathbf{S}^{\mathrm{T}}\mathbf{S} \end{bmatrix}


    as given in (3.4) of [1].

    Hence its inverse can be factorized almost `symmetrically` by using Cholesky
    factorizations of the submatrices.
    Now, the inverse of Mk, the middle matrix in B reads:

    [  D^(1/2)      O ] [ -D^(1/2)  D^(-1/2)*L' ]
    [ -L*D^(-1/2)   J ] [  0        J'          ]

    With J @ J' = T = theta*Ss + L*D^(-1)*L'; T being definite positive,
    J is obtained by Cholesky factorization of T.

    REF: see algo 3.2 in :cite:t:`byrdRepresentationsQuasiNewtonMatrices1994`.
    """
    invD = np.zeros_like(D)
    # Add 1/D on diagonal
    invD.flat[:: D.shape[0] + 1] = 1 / np.diag(D)

    # Cholesky factorization
    J = sp.linalg.cholesky(theta * STS + L @ invD @ L.T, lower=True)

    m = D.shape[0]
    dtype = D.dtype

    sqrtD = np.sqrt(D)
    sqrtInvD = np.sqrt(invD)

    # Precompute products
    L_sqrtInvD = L @ sqrtInvD  # (m, m)
    sqrtInvD_LT = sqrtInvD @ L.T  # (m, m)

    L = np.zeros((2 * m, 2 * m), dtype=dtype)
    U = np.zeros((2 * m, 2 * m), dtype=dtype)

    # -------- Lower matrix --------
    L[:m, :m] = sqrtD
    L[m:, :m] = -L_sqrtInvD
    L[m:, m:] = J

    # -------- Upper matrix --------
    U[:m, :m] = -sqrtD
    U[:m, m:] = sqrtInvD_LT
    U[m:, m:] = J.T

    # Note we form the upper triangle and then transpose it to get the lower one
    return L, U


def update_lbfgs_matrices(
    xk: NDArrayFloat,
    gk: NDArrayFloat,
    X: Deque[NDArrayFloat],
    G: Deque[NDArrayFloat],
    maxcor: int,
    mats: LBFGSB_MATRICES,
    is_force_update: bool,
    eps: float = 2.2e-16,
    is_check_factorization: bool = False,
    is_use_numba_jit: bool = False,
) -> LBFGSB_MATRICES:
    r"""
    Update lists S and Y, and form the L-BFGS Hessian approximation.

    Instead of storing sk and yk, we store the gradients and the parameters.

    2 conditions for update
    - The current step update is accepted
    - The all sequence of x and g has been modified (reg case)

    Parameters
    ----------
    xk : NDArrayFloat
        New x parameter.
    gk : NDArrayFloat
        New gradient parameter g.
    X : Deque[NDArrayFloat]
        List of successive parameters x.
    G : Deque[NDArrayFloat]
        List of successive gradients.
    maxcor : int
        The maximum number of variable metric corrections used to
        define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms
        in an approximation to it.)
    is_force_update: bool
        Whether to perform an update even if the current step update is rejected.
        This is useful if the sequence of X and G has been modified during the
        optimization. See TODO: add ref, for the use.
    eps : float, optional
        Positive stability parameter for accepting current step for updating.
        By default 2.2e-16.
    is_force_update: bool
        This is to recompute fully the matrices.
    is_check_factorization: bool
        Whether to check the cholesky factorization of M. The default is False.
    is_use_numba_jit: bool
        Whether to use `numba` just-in-time compilation to speed-up the computation
        intensive part of the algorithm. The default is False.

    Returns
    -------
    Tuple[NDArrayFloat, Tuple[NDArrayFloat, NDArrayFloat], float]
        Updated [W, M, invMfactors, theta]

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
    # Case of a vector
    is_current_update_accepted: bool = update_X_and_G(
        xk, gk, X, G, maxcor, eps, is_use_numba_jit
    )

    # two conditions to update the inverse Hessian approximation
    if is_force_update or is_current_update_accepted:
        # yk and sk: These correction pairs contain information about the curvature of
        # the
        # function and, in conjunction with the BFGS formula, define the limited-memory
        # iteration matrix Bk. The question is how to best represent these matrices
        # without explicitly forming them. In [6] it is proposed to use a compact
        # (or outer product) form to define the limited-memory matrix Bk in terms of
        # the n x m correction matrices

        # 1) Update theta
        sk = X[-1] - X[-2]
        yk = G[-1] - G[-2]
        # sk = X[-1] - X[-2]
        sTy = sk @ yk  # type: ignore
        yTy = yk @ yk  # type: ignore
        mats.theta = yTy / sTy

        m = len(X) - 1
        n = np.size(X[-1])

        # Update the lbfgsb matrices
        if is_force_update:
            mats.S = np.diff(np.array(X), axis=0).T  # shape (n, m)
            mats.Y = np.diff(np.array(G), axis=0).T  # shape (n ,m)
        else:
            if np.shape(mats.S)[1] == m:
                # shift left
                mats.S[:, :-1] = mats.S[:, 1:]
                mats.Y[:, :-1] = mats.Y[:, 1:]
            else:
                _S = np.zeros((n, m), dtype=np.float64)
                _Y = np.zeros((n, m), dtype=np.float64)
                _S[:, :-1] = mats.S[:, :]
                _Y[:, :-1] = mats.Y[:, :]
                mats.S = _S
                mats.Y = _Y
            # append newest column
            mats.S[:, -1] = sk
            mats.Y[:, -1] = yk

        STS = mats.S.T @ mats.S  # shape (m, m)
        SY = mats.S.T @ mats.Y
        # We can build a dense matrix because shape is (m, m) with m usually small ~10
        mats.D = np.diag(np.diag(SY))  # shape (m, m)
        mats.L = np.tril(SY, -1)  # shape (m, m)

        # W = [Yk, \theta Sk]
        n, m = np.shape(mats.S)
        mats.W = np.zeros((n, 2 * m))
        mats.W[:, :m] = mats.Y
        mats.W[:, m:] = mats.theta * mats.S  # shape (n, 2m)

        # To avoid forming the limited-memory iteration matrix Bk and allow fast
        # matrix vector products, we represent it as eq. (3.2) [1].
        # B = theta * I  - W @ M @ W.T

        # M (or Mk) can be obtained with
        # M = np.linalg.inv(
        #     np.hstack([np.vstack([-D, L]), np.vstack([L.T, theta * STS])])
        # )
        # However, we can also factorize its inverse and obtain very fast matrix
        # products: lower triangle of M inverse
        mats.invMfactors = form_invMfactors(mats.theta, STS, mats.L, mats.D)

        # Test the factorization on the fly.
        if is_check_factorization:
            np.testing.assert_allclose(
                mats.invMfactors[0] @ mats.invMfactors[1],
                np.hstack(
                    [
                        np.vstack([-mats.D, mats.L]),
                        np.vstack([mats.L.T, mats.theta * STS]),
                    ]
                ),
            )

    return mats


def is_update_X_and_G(
    xk: NDArrayFloat,
    gk: NDArrayFloat,
    x_old: NDArrayFloat,
    g_old: NDArrayFloat,
    eps: float = 2.2e-16,
) -> bool:
    """
    Update the sequence of parameters X and gradients G with a strong wolfe condition.

    Parameters
    ----------
    xk : NDArrayFloat
        New adjusted values vector (at iteration k).
    gk : NDArrayFloat
        New gradient vector (at iteration k).
    x_old : NDArrayFloat
        Previous adjusted values vector (at iteration k-1).
    g_old : NDArrayFloat
        Previous gradient vector (at iteration k-1).
    maxcor : int
        Maximum number of corrections stored (m).
    eps : float, optional
        Positive stability parameter for accepting current step for updating.
        By default 2.2e-16.
    Returns
    -------
    bool
        Whether the current step as been accepted.
    """
    yk = gk - g_old
    sTy = (xk - x_old).dot(yk)  # type: ignore
    yTy = (yk).dot(yk)  # type: ignore

    # See eq. (3.9) in [1].
    # One can show that BFGS update (2.19) generates positive definite approximations
    # whenever the initial approximation B0 is positive definite and sT k yk > 0.
    # We discuss these issues further in Chapter 6. (See Numerical optimization in
    # Noecedal and Wright)
    return sTy > eps * yTy


@njit(cache=True)
def is_update_X_and_G_numba(
    xk: NDArrayFloat,
    gk: NDArrayFloat,
    x_old: NDArrayFloat,
    g_old: NDArrayFloat,
    eps: float = 2.2e-16,
) -> bool:
    """
    Update the sequence of parameters X and gradients G with a strong wolfe condition.

    Parameters
    ----------
    xk : NDArrayFloat
        New adjusted values vector (at iteration k).
    gk : NDArrayFloat
        New gradient vector (at iteration k).
    x_old : NDArrayFloat
        Previous adjusted values vector (at iteration k-1).
    g_old : NDArrayFloat
        Previous gradient vector (at iteration k-1).
    maxcor : int
        Maximum number of corrections stored (m).
    eps : float, optional
        Positive stability parameter for accepting current step for updating.
        By default 2.2e-16.
    Returns
    -------
    bool
        Whether the current step as been accepted.
    """
    sTy = 0.0
    yTy = 0.0
    for i in range(xk.size):
        y = gk[i] - g_old[i]
        sTy += (xk[i] - x_old[i]) * y
        yTy += y * y

    # See eq. (3.9) in [1].
    # One can show that BFGS update (2.19) generates positive definite approximations
    # whenever the initial approximation B0 is positive definite and sT k yk > 0.
    # We discuss these issues further in Chapter 6. (See Numerical optimization in
    # Noecedal and Wright)
    return sTy > eps * yTy


def update_X_and_G(
    xk: NDArrayFloat,
    gk: NDArrayFloat,
    X: Deque[NDArrayFloat],
    G: Deque[NDArrayFloat],
    maxcor: int,
    eps: float = 2.2e-16,
    is_use_numba_jit: bool = False,
) -> bool:
    """

    Parameters
    ----------
    xk : NDArrayFloat
        New adjusted values vector.
    gk : NDArrayFloat
        New gradient vector.
    X : Deque[NDArrayFloat]
        Sequence of past adjusted values vectors respecting the strong wolfe conditions.
    G : Deque[NDArrayFloat]
        Sequence of past gradient vectors respecting the strong wolfe conditions.
    maxcor : int
        Maximum number of corrections stored (m).
    eps : float, optional
        Positive stability parameter for accepting current step for updating.
        By default 2.2e-16.
    is_use_numba_jit: bool
        Whether to use `numba` just-in-time compilation to speed-up the computation
        intensive part of the algorithm. The default is False.

    Returns
    -------
    bool
        True if the matrices have been updated, False otherwise.
    """
    if is_use_numba_jit:
        if not is_update_X_and_G_numba(xk, gk, X[-1], G[-1], eps):
            return False
    else:
        if not is_update_X_and_G(xk, gk, X[-1], G[-1], eps):
            return False

    X.append(xk)
    G.append(gk)
    # maxcor is the number of corrections m (see S and Y shapes),
    # so we must keep one more gradient and parameter vectors.
    if len(X) > maxcor + 1:
        X.popleft()
        G.popleft()

    return True


def make_X_and_G_respect_strong_wolfe(
    X: Deque[NDArrayFloat],
    G: Deque[NDArrayFloat],
    eps: float = 2.2e-16,
    logger: Optional[logging.Logger] = None,
    is_use_numba_jit: bool = False,
) -> Tuple[Deque[NDArrayFloat], Deque[NDArrayFloat]]:
    """

    Parameters
    ----------
    X : Deque[NDArrayFloat]
        Sequence of past adjusted values vectors respecting the strong wolfe conditions.
    G : Deque[NDArrayFloat]
        Sequence of past gradient vectors respecting the strong wolfe conditions.
    eps : float, optional
        Positive stability parameter for accepting current step for updating.
        By default 2.2e-16.
    logger: Optional[Logger], optional
        :class:`logging.Logger` instance. If None, nothing is displayed, no matter the
        value of `iprint`, by default None.
    is_use_numba_jit: bool
        Whether to use `numba` just-in-time compilation to speed-up the computation
        intensive part of the algorithm. The default is False.

    Returns
    -------
    Tuple[Deque[NDArrayFloat], Deque[NDArrayFloat]]
        Updated X and G.
    """

    ncor: int = len(X) - 1
    _X, _G = deque([X[-1]]), deque([G[-1]])
    for i in range(ncor):
        k = ncor - i - 1  # start at 1
        if not is_update_X_and_G(X[k], G[k], _X[0], _G[0], eps):
            if logger is not None:
                logger.info(f"Dropping update #{-i - 2}")
        else:
            _X.appendleft(X[k])
            _G.appendleft(G[k])

    # This is for debug
    if len(_G) != len(G) and logger is not None:
        # if logger is not None:
        logger.info(f"len(newG) = {len(_G)}, len(oldG) = {len(G)}")
    return _X, _G
