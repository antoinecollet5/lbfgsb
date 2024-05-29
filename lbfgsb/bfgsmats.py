"""
An *implicit* representation of the BFGS approximation to the Hessian matrix B

B = theta * I - W * M * W'
H = inv(B)

Functions
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

    bmv
    form_invMfactors
    update_lbfgs_matrices
    update_X_and_G

Reference:
[1] D. C. Liu and J. Nocedal (1989). On the limited memory BFGS method for large scale
optimization.
[2] R. H. Byrd, P. Lu, and J. Nocedal (1995). A limited memory algorithm for bound
constrained optimization.
"""

from typing import Deque, Tuple

import numpy as np
import scipy as sp

from lbfgsb.types import NDArrayFloat


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
        _description_
    v : NDArrayFloat
        _description_

    Returns
    -------
    NDArrayFloat
        _description_
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


def form_invMfactors(theta, STS, L, D) -> Tuple[NDArrayFloat, NDArrayFloat]:
    r"""
    Return upper triangle of the cholesky factorization of the inverse of M_k.

    This is defined in eq. (3.4) [1].

    Although Mk is not positive definite, but its inverse reads.

    [-D  L'          ]
    [L   theta * S'*S]

    Hence its inverse can be factorized almost `symmetrically` by using Cholesky
    factorizations of the submatrices TODO: add ref to the phd manuscript.
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

    # Note we form the upper triangle and then transpose it to get the lower one
    return (
        np.hstack(
            [
                np.vstack([np.sqrt(D), -(np.sqrt(invD) @ L.T).T]),  # upper row
                np.vstack([np.zeros(D.shape), J]),  # lower row
            ]
        ),
        np.hstack(
            [
                np.vstack([-np.sqrt(D), np.zeros(D.shape)]),  # upper row
                np.vstack([np.sqrt(invD) @ L.T, J.T]),  # lower row
            ]
        ),
    )


def update_lbfgs_matrices(
    xk: NDArrayFloat,
    gk: NDArrayFloat,
    X: Deque[NDArrayFloat],
    G: Deque[NDArrayFloat],
    maxcor: int,
    W: NDArrayFloat,
    M: NDArrayFloat,
    invMfactors: Tuple[NDArrayFloat, NDArrayFloat],
    theta: float,
    is_force_update: bool,
    eps: float = 2.2e-16,
) -> Tuple[NDArrayFloat, NDArrayFloat, Tuple[NDArrayFloat, NDArrayFloat], float]:
    r"""
    Update lists S and Y, and form the L-BFGS Hessian approximation thet, W and M.

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
    X : deque
        List of successive parameters x.
    G : deque
        List of successive gradients.
    maxcor : int
        The maximum number of variable metric corrections used to
        define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms
        in an approximation to it.)
    W : NDArrayFloat
        L-BFGS matrices.
    M : NDArrayFloat
        L-BFGS matrices.
    thet : float
        L-BFGS float parameter (multiply the identity matrix).
    is_force_update: bool
        Whether to perform an update even if the current step update is rejected.
        This is useful if the sequence of X and G has been modified during the
        optimization. See TODO: add ref, for the use.
    eps : float, optional
        Positive stability parameter for accepting current step for updating.
        By default 2.2e-16.

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat, Tuple[NDArrayFloat, NDArrayFloat], float]
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
    if xk.ndim == 2:
        # Case of an ensemble (xk and gk)
        is_current_update_accepted = False
        # TODO: Can we do a bit better than a for loop ?
        for i, _x in enumerate(xk.T):
            if update_X_and_G(_x, gk[:, i], X, G, maxcor, eps):
                is_current_update_accepted = True
    else:
        # Case of a vector
        is_current_update_accepted: bool = update_X_and_G(xk, gk, X, G, maxcor, eps)

    # two conditions to update the inverse Hessian approximation
    if is_force_update or is_current_update_accepted:
        # 1) Update theta
        yk = G[-1] - G[-2]
        sTy = (X[-1] - X[-2]).dot(yk)  # type: ignore
        yTy = (yk).dot(yk)  # type: ignore
        theta = yTy / sTy

        # Update the lbfgsb matrices
        Sarray = np.diff(np.array(X), axis=0).T  # shape (n, m - 1)
        Yarray = np.diff(np.array(G), axis=0).T  # shape (n ,m - 1)
        STS = Sarray.T @ Sarray
        L = Sarray.T @ Yarray
        # We can build a dense matrix because shape is (m, m) with m usually small ~10
        D = np.diag(np.diag(L))
        L = np.tril(L, -1)

        W = np.hstack([Yarray, theta * Sarray])

        # To avoid forming the limited-memory iteration matrix Bk and allow fast
        # matrix vector products, we represent it as eq. (3.2) [1].
        # B = theta * I  - W @ M @ W.T

        # M (or Mk) can be obtained with
        M = np.linalg.inv(
            np.hstack([np.vstack([-D, L]), np.vstack([L.T, theta * STS])])
        )
        # However, we can also factorize its inverse and obtain very fast matrix
        # products: lower triangle of M inverse
        invMfactors = form_invMfactors(theta, STS, L, D)

        np.testing.assert_allclose(
            invMfactors[0] @ invMfactors[1],
            np.hstack([np.vstack([-D, L]), np.vstack([L.T, theta * STS])]),
        )

    return W, M, invMfactors, theta


def update_X_and_G(
    xk: NDArrayFloat,
    gk: NDArrayFloat,
    X: Deque[NDArrayFloat],
    G: Deque[NDArrayFloat],
    maxcor: int,
    eps: float = 2.2e-16,
) -> bool:
    """
    Update the sequence of parameters X and gradients G with a strong wolfe condition.

    Parameters
    ----------
    xk : NDArrayFloat
        _description_
    gk : NDArrayFloat
        _description_
    X : Deque[NDArrayFloat]
        _description_
    G : Deque[NDArrayFloat]
        _description_
    maxcor : int
        _description_
    eps : float, optional
        _description_, by default 2.2e-16

    Returns
    -------
    bool
        Whether the current step as been accepted.
    """
    yk = gk - G[-1]
    sTy = (xk - X[-1]).dot(yk)  # type: ignore
    yTy = (yk).dot(yk)  # type: ignore

    # See eq. (3.9) in [1].
    # One can show that BFGS update (2.19) generates positive definite approximations
    # whenever the initial approximation B0 is positive definite and sT k yk > 0.
    # We discuss these issues further in Chapter 6. (See Numerical optimization in
    # Noecedal and Wright)
    if sTy > eps * yTy:
        X.append(xk)
        G.append(gk)
        if len(X) > maxcor:
            X.popleft()
            G.popleft()
        return True
    return False
