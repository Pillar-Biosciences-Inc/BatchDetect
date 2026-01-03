from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .mixture import HeavyMixture


@dataclass
class LMRResult:
    lr: float  # raw LR = 2*(ll_alt - ll_null)
    lmr_lr: float  # adjusted LMR statistic
    df: int  # parameter count difference
    p_value: float  # LMR p-value


def _pack_params(model: HeavyMixture) -> np.ndarray:
    """
    Pack HeavyMixture params into unconstrained theta:
      theta = [alpha_1..alpha_{K-1}, means_flat (K*d), log_scales (K)]
    where weights = softmax([alpha..., 0]).
    """
    w = np.clip(model.weights_, 1e-15, 1.0)
    w = w / w.sum()
    alpha = np.log(w[:-1]) - np.log(w[-1])  # alpha_K fixed to 0
    means = np.asarray(model.means_, dtype=float).reshape(-1)
    log_scales = np.log(np.maximum(model.scales_, 1e-15)).reshape(-1)
    return np.concatenate([alpha, means, log_scales], axis=0)


def _unpack_params(theta: np.ndarray, K: int, d: int):
    """
    Inverse of _pack_params: map unconstrained theta -> (weights, means, scales).
    """
    theta = np.asarray(theta, dtype=float)
    alpha = theta[: K - 1]
    means = theta[K - 1 : K - 1 + K * d].reshape(K, d)
    log_scales = theta[K - 1 + K * d :].reshape(K)

    # stable softmax([alpha..., 0])
    if alpha.size:
        m = float(max(0.0, np.max(alpha)))
        ex = np.exp(alpha - m)
        Z = 1.0 + ex.sum()
        w = np.empty(K, dtype=float)
        w[:-1] = ex / Z
        w[-1] = 1.0 / Z
    else:
        w = np.ones(1, dtype=float)

    scales = np.exp(log_scales)
    return w, means, scales


def _set_params_inplace(model: HeavyMixture, theta: np.ndarray):
    K = model.n_components
    d = model.n_features_in_
    w, means, scales = _unpack_params(theta, K, d)
    model.weights_ = w
    model.means_ = means
    model.scales_ = scales


def _logprob_per_sample(
    model: HeavyMixture, theta: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """
    Return per-sample log p(x_i | theta). Mutates model parameters.
    """
    _set_params_inplace(model, theta)
    return model.score_samples(X)


def _score_matrix_fd(
    model: HeavyMixture, theta: np.ndarray, X: np.ndarray, fd_eps: float
) -> np.ndarray:
    """
    Per-sample score vectors via central finite differences:
      S[i, j] = d/dtheta_j log p(x_i | theta)
    Returns shape (n, p).
    """
    theta = np.asarray(theta, dtype=float)
    n = X.shape[0]
    p = theta.size
    S = np.empty((n, p), dtype=float)

    base = _logprob_per_sample(model, theta, X)  # also sets model to theta
    # step sizes
    eps = fd_eps * (1.0 + np.abs(theta))

    for j in range(p):
        tj_p = theta.copy()
        tj_m = theta.copy()
        tj_p[j] += eps[j]
        tj_m[j] -= eps[j]
        lp = _logprob_per_sample(model, tj_p, X)
        lm = _logprob_per_sample(model, tj_m, X)
        S[:, j] = (lp - lm) / (2.0 * eps[j])

    # restore base theta
    _set_params_inplace(model, theta)
    _ = base  # keep lint happy
    return S


def _hessian_avg_loglik_fd(
    model: HeavyMixture, theta: np.ndarray, X: np.ndarray, fd_eps: float
) -> np.ndarray:
    """
    Hessian of average log-likelihood L(theta) = mean_i log p(x_i | theta)
    via central second differences.

    Returns A_hat with shape (p, p), approximating:
      A = E[ d^2 / dtheta dtheta^T log p(X | theta) ].
    """
    theta = np.asarray(theta, dtype=float)
    p = theta.size
    eps = fd_eps * (1.0 + np.abs(theta))

    def L(t):
        return float(_logprob_per_sample(model, t, X).mean())

    H = np.empty((p, p), dtype=float)
    L0 = L(theta)

    # diagonal
    for i in range(p):
        tp = theta.copy()
        tm = theta.copy()
        tp[i] += eps[i]
        tm[i] -= eps[i]
        Lp = L(tp)
        Lm = L(tm)
        H[i, i] = (Lp - 2.0 * L0 + Lm) / (eps[i] ** 2)

    # off-diagonal (symmetric)
    for i in range(p):
        for j in range(i + 1, p):
            tpp = theta.copy()
            tpm = theta.copy()
            tmp = theta.copy()
            tmm = theta.copy()

            tpp[i] += eps[i]
            tpp[j] += eps[j]
            tpm[i] += eps[i]
            tpm[j] -= eps[j]
            tmp[i] -= eps[i]
            tmp[j] += eps[j]
            tmm[i] -= eps[i]
            tmm[j] -= eps[j]

            Lpp = L(tpp)
            Lpm = L(tpm)
            Lmp = L(tmp)
            Lmm = L(tmm)

            hij = (Lpp - Lpm - Lmp + Lmm) / (4.0 * eps[i] * eps[j])
            H[i, j] = hij
            H[j, i] = hij

    # restore base theta
    _set_params_inplace(model, theta)
    return H


def _imhof_cdf_weighted_chisq(
    y: float, lambdas: np.ndarray, eps: float = 1e-7
) -> float:
    """
    Imhof inversion for Q = sum_i lambda_i * chi2_1, return P(Q < y).

    Matches LMR paper eq (13)-(14), including the u->0 limit.
    """
    lam = np.asarray(lambdas, dtype=float).ravel()
    lam = lam[np.abs(lam) > 1e-12]
    if lam.size == 0:
        return 1.0 if y >= 0.0 else 0.0

    m = lam.size

    # Choose a finite truncation U (paper suggests truncation control).
    # Use a conservative heuristic based on the envelope of 1/(u*rho(u)).
    C = float(np.prod(np.sqrt(np.abs(lam))))
    if (not np.isfinite(C)) or C <= 0.0:
        U = 500.0
    else:
        # Tail ~ O(U^{-m/2}); solve for eps roughly
        U = (1.0 / (C * max(1.0, (m / 2.0)) * eps)) ** (2.0 / m)
        U = float(max(50.0, min(5000.0, U)))

    lam_sum = float(lam.sum())

    def integrand(u: float) -> float:
        if u == 0.0:
            # lim_{u->0} sin(delta(u)) / (u*rho(u)) = 0.5*sum(lam) - 0.5*y
            return 0.5 * lam_sum - 0.5 * y

        lu = lam * u
        delta = 0.5 * np.sum(np.arctan(lu)) - 0.5 * y * u
        rho = float(np.prod((1.0 + lu * lu) ** 0.25))
        return float(np.sin(delta) / (u * rho))

    val, _err = quad(integrand, 0.0, U, epsabs=eps, epsrel=eps, limit=2000)
    cdf = 0.5 - (1.0 / np.pi) * val
    return float(min(1.0, max(0.0, cdf)))


def lmr_test_heavymixture(
    X,
    L: int,
    K: int,
    *,
    component_distribution: str = "gennorm",
    fit_kwargs: dict | None = None,
    fd_eps: float = 1e-4,
    ridge_A: float = 1e-8,
    imhof_eps: float = 1e-7,
    correction: bool = True,
    random_state: int | None = 0,
) -> LMRResult:
    """
    Paper-consistent LMR/Vuong-style test for #components: L (null) vs K (alt), K > L.

    - Fits HeavyMixture(L) and HeavyMixture(K) on X.
    - Computes LR = 2*(ll_K - ll_L), where ll_* are total log-likelihoods.
    - Estimates W using Af, Ag, Bf, Bg, Bfg, Bgf as in LMR (via Vuong).
    - Uses Imhof inversion to get p-value for the weighted chi-square sum.
    - Optionally applies LMR small-sample correction:
        LR* = LR / (1 + 1 / ((p-q) * log(n))).

    Returns
    -------
    LMRResult(lr, lmr_lr, df, p_value)
    """
    if K <= L:
        raise ValueError("Require K > L.")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]

    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)

    # Fit models
    null_model = HeavyMixture(
        n_components=L,
        component_distribution=component_distribution,
        **fit_kwargs,
    )
    alt_model = HeavyMixture(
        n_components=K,
        component_distribution=component_distribution,
        **fit_kwargs,
    )

    if random_state is not None:
        # HeavyMixture uses np.random.default_rng internally in sample(), not fit(),
        # but users often pass random seeds via fit_kwargs (n_init etc).
        pass

    null_model.fit(X)
    alt_model.fit(X)

    # Total log-likelihoods (sum over samples)
    ll_L = float(null_model.score_samples(X).sum())
    ll_K = float(alt_model.score_samples(X).sum())

    lr = 2.0 * (ll_K - ll_L)

    # Pack MLE params (theta_f for alt, theta_g for null)
    theta_f = _pack_params(alt_model)
    theta_g = _pack_params(null_model)

    p = int(theta_f.size)
    q = int(theta_g.size)
    df = p - q

    # Save and restore original fitted params to avoid side effects outside this function
    alt_backup = (
        alt_model.weights_.copy(),
        alt_model.means_.copy(),
        alt_model.scales_.copy(),
    )
    nul_backup = (
        null_model.weights_.copy(),
        null_model.means_.copy(),
        null_model.scales_.copy(),
    )

    try:
        # Scores (per-sample gradients)
        Sf = _score_matrix_fd(alt_model, theta_f, X, fd_eps=fd_eps)  # (n, p)
        Sg = _score_matrix_fd(null_model, theta_g, X, fd_eps=fd_eps)  # (n, q)

        # B matrices (mean outer products)
        Bf = (Sf.T @ Sf) / n
        Bg = (Sg.T @ Sg) / n
        Bfg = (Sf.T @ Sg) / n
        Bgf = Bfg.T

        # A matrices (mean Hessians of loglik)
        Af = _hessian_avg_loglik_fd(alt_model, theta_f, X, fd_eps=fd_eps)
        Ag = _hessian_avg_loglik_fd(null_model, theta_g, X, fd_eps=fd_eps)

        # Stabilize inversions
        Af_inv = np.linalg.inv(Af + ridge_A * np.eye(p))
        Ag_inv = np.linalg.inv(Ag + ridge_A * np.eye(q))

        # W block matrix (LMR/Vuong)
        W11 = -Bf @ Af_inv
        W12 = -Bfg @ Ag_inv
        W21 = Bgf @ Af_inv
        W22 = Bg @ Ag_inv
        W = np.block([[W11, W12], [W21, W22]])

        eig = np.linalg.eigvals(W)
        lambdas = eig.real  # numerical imaginary parts should be tiny

        # Apply LMR correction if requested
        lmr_lr = float(lr)
        if correction:
            if df <= 0:
                # Correction is defined for p-q > 0 in the paper; if not, skip.
                lmr_lr = float(lr)
            else:
                denom = 1.0 + 1.0 / (df * np.log(n))
                lmr_lr = float(lr / denom)

        cdf = _imhof_cdf_weighted_chisq(lmr_lr, lambdas, eps=imhof_eps)
        p_value = float(1.0 - cdf)

        return LMRResult(
            lr=float(lr),
            lmr_lr=float(lmr_lr),
            df=int(df),
            p_value=float(p_value),
        )

    finally:
        alt_model.weights_, alt_model.means_, alt_model.scales_ = alt_backup
        null_model.weights_, null_model.means_, null_model.scales_ = nul_backup
