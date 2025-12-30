import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr

def _qr_cols(A):
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q

def _project_perp(Q, M):
    if M.ndim == 2:
        return M - Q @ (Q.T @ M)
    Ns, d, N = M.shape
    M2 = M.reshape(Ns, d*N)
    return (M2 - Q @ (Q.T @ M2)).reshape(Ns, d, N)

def _make_Fop_K_only(Phi, Phi0_tilde, dt, m):
    Ns, d, N = Phi.shape
    if m < 1:
        raise ValueError("m must be >= 1")
    Q = _qr_cols(Phi0_tilde)
    Phi_perp = _project_perp(Q, Phi)
    n_eq = Ns * d * N
    nK = d * d * m

    def matvec(x):
        K = x.reshape(d, d, m)
        Y = np.zeros((Ns, d, N), dtype=float)
        for n in range(N):
            jmax = min(m-1, n)
            acc = Y[:, :, n]
            for j in range(jmax + 1):
                w = (0.5 * dt) if j == 0 else dt
                acc += w * (Phi_perp[:, :, n - j] @ K[:, :, j].T)
        return Y.ravel()

    def rmatvec(y):
        Y = y.reshape(Ns, d, N)
        gK = np.zeros((d, d, m), dtype=Y.dtype)
        for n in range(N):
            jmax = min(m-1, n)
            Yn = Y[:, :, n]
            for j in range(jmax + 1):
                w = (0.5 * dt) if j == 0 else dt
                gK[:, :, j] += w * (Phi_perp[:, :, n - j].T @ Yn).T
        return gK.ravel()

    return LinearOperator((n_eq, nK), matvec=matvec, rmatvec=rmatvec, dtype=float)

def lsqrKB(Phi, Phi0_tilde, Z, dt, m, rcond, maxiter=500, tol=1e-12):
    """
    Returns:
      K : (d, d, m)
      B : (d, e, N)
      result : lsqr tuple
    """
    Ns, d, N = Phi.shape
    Q = _qr_cols(Phi0_tilde)
    Z_perp = _project_perp(Q, Z)

    Fop = _make_Fop_K_only(Phi, Phi0_tilde, dt, m)
    result = lsqr(Fop, Z_perp.ravel(), atol=tol, btol=tol, iter_lim=maxiter)

    nK = d * d * m
    K = result[0].reshape(d, d, m)

    e = Phi0_tilde.shape[1]
    B = np.zeros((d, e, N), dtype=float)
    for n in range(N):
        Rn = Z[:, :, n].copy()
        jmax = min(m-1, n)
        for j in range(jmax + 1):
            w = (0.5 * dt) if j == 0 else dt
            Rn -= w * (Phi[:, :, n - j] @ K[:, :, j].T)
        X, *_ = np.linalg.lstsq(Phi0_tilde, Rn, rcond=rcond)  # X: (e,d)
        B[:, :, n] = X.T
    return K, B, result
