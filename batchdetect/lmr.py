from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.integrate import quad
from scipy.stats import chi2

from .mixture import HeavyMixture


@dataclass
class LMRResult:
    lr: float  # raw LR = 2*(ll_alt - ll_null)
    lmr_lr: float  # adjusted LMR statistic
    df: int  # parameter count difference
    p_value: float  # LMR p-value


def lo_mendell_rubin_lrt(
    n: int,
    ll_null: float,
    ll_alt: float,
    n_params_null: int,
    n_params_alt: int,
    k_null: int,
    k_alt: int,
) -> LMRResult:
    """
    Lo–Mendell–Rubin adjusted likelihood ratio test for K_null vs K_alt
    mixture / latent class models.

    Parameters
    ----------
    n : int
        Sample size (number of observations used in both models).
    ll_null : float
        Log-likelihood under the null model (K_null classes).
    ll_alt : float
        Log-likelihood under the alternative model (K_alt classes).
    n_params_null : int
        Number of free parameters in the null model.
    n_params_alt : int
        Number of free parameters in the alternative model.
    k_null : int
        Number of classes/components in the null model.
    k_alt : int
        Number of classes/components in the alternative model.

    Returns
    -------
    LMRResult
        Contains raw LR, adjusted LMR statistic, df, and p-value.
    """
    if k_alt <= k_null:
        raise ValueError(
            "Alternative model must have more classes than null model."
        )

    # Raw likelihood ratio statistic
    lr = 2.0 * (ll_alt - ll_null)

    # LMR adjustment: formula used in tidyLPA (Lo–Mendell–Rubin 2001, eq. 15)
    # modlr = LR / [1 + ((3*K_alt - 1) - (3*K_null - 1)) / log(n)]
    #       = LR / [1 + 3*(K_alt - K_null) / log(n)]
    delta_k = k_alt - k_null
    denom = 1.0 + 3.0 * delta_k / np.log(n)
    lmr_lr = lr / denom

    df = int(n_params_alt - n_params_null)
    if df <= 0:
        raise ValueError(
            "Alternative model must have more parameters than null model."
        )

    p_value = chi2.sf(lmr_lr, df)

    return LMRResult(lr=lr, lmr_lr=lmr_lr, df=df, p_value=p_value)


def davies_pvalue_weighted_chisq(lambdas, x, *, tol=1e-10, limit=500):
    """
    Deterministic tail probability for Q = sum_j lambdas_j * Z_j^2, Z_j ~ N(0,1).

    SciPy-only implementation via the Imhof inversion integral (commonly used in
    practice as a "Davies/Imhof-style" deterministic quadratic-form p-value).

    Returns p = P(Q >= x).
    """
    lambdas = np.asarray(lambdas, dtype=float).ravel()
    if lambdas.size == 0:
        raise ValueError("lambdas must be non-empty.")
    if not np.isfinite(x):
        raise ValueError("x must be finite.")

    # Imhof:
    # P(Q <= x) = 1/2 - (1/pi) * integral_0^inf sin(theta(u)) / (u*rho(u)) du
    # theta(u) = 0.5 * sum_j arctan(l_j*u) - 0.5*x*u
    # rho(u)   = prod_j (1 + (l_j*u)^2)^(1/4)
    def integrand(u):
        if u == 0.0:
            return 0.0
        lu = lambdas * u
        theta = 0.5 * np.sum(np.arctan(lu)) - 0.5 * x * u
        log_rho = 0.25 * np.sum(np.log1p(lu * lu))
        rho = np.exp(log_rho)
        return np.sin(theta) / (u * rho)

    val, _err = quad(
        integrand, 0.0, np.inf, epsabs=tol, epsrel=tol, limit=limit
    )
    cdf = 0.5 - (1.0 / np.pi) * val
    cdf = float(np.clip(cdf, 0.0, 1.0))
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def weighted_chisq_lrt_num_components(
    X,
    L,
    K,
    *,
    component_distribution="laplace",
    fit_kwargs=None,
    fd_eps=1e-4,
    ridge_I=1e-8,
    pvalue_method="davies",  # "davies" or "mc"
    davies_tol=1e-10,
    davies_limit=500,
    n_sim=100_000,
    random_state=0,
    return_simulated=False,  # only meaningful for MC
):
    """
    Approximate LRT for #components L vs K using weighted chi-square with lambdas from I^{-1}J.

    - Fits HeavyMixture for L and K components.
    - Computes LR = 2*(ll_K - ll_L).
    - Constructs J = sum_i s_i s_i^T from per-observation scores s_i by bulk finite differences
      of HeavyMixture.score_samples at the K-fit parameters.
    - Constructs I = -H (observed information) by finite-diff Hessian of total loglik.
    - Lambdas are eigvals of B = I^{-1/2} J I^{-1/2}.
    - p-value via either:
        * pvalue_method="davies": deterministic Imhof inversion integral (SciPy quad)
        * pvalue_method="mc": Monte Carlo simulation of sum lambdas_j * Z_j^2
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not (1 <= L < K):
        raise ValueError("Require 1 <= L < K.")

    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    # ---- fit models using your implementation ----
    null_model = HeavyMixture(
        n_components=L,
        component_distribution=component_distribution,
        **fit_kwargs,
    ).fit(X)
    alt_model = HeavyMixture(
        n_components=K,
        component_distribution=component_distribution,
        **fit_kwargs,
    ).fit(X)

    n = X.shape[0]
    ll_L = float(null_model.score(X)) * n
    ll_K = float(alt_model.score(X)) * n
    lr = 2.0 * (ll_K - ll_L)

    # ---- parameterization for K-component model ----
    # theta = [alpha (K-1 logits), means (K*d), log_scales (K)]
    def pack(model):
        w = np.clip(model.weights_, 1e-15, 1.0)
        w = w / w.sum()
        alpha = np.log(w[:-1]) - np.log(w[-1])  # alpha_K fixed to 0
        means = model.means_.reshape(-1)
        log_scales = np.log(np.maximum(model.scales_, 1e-15)).reshape(-1)
        return np.concatenate([alpha, means, log_scales], axis=0)

    def unpack(theta, K_, d_):
        alpha = theta[: K_ - 1]
        means = theta[K_ - 1 : K_ - 1 + K_ * d_].reshape(K_, d_)
        log_scales = theta[K_ - 1 + K_ * d_ :].reshape(K_)
        ex = np.exp(alpha - np.max(alpha))
        Z = 1.0 + ex.sum()
        w = np.empty(K_, dtype=float)
        w[:-1] = ex / Z
        w[-1] = 1.0 / Z
        scales = np.exp(log_scales)
        return w, means, scales

    def score_samples_given_theta(theta):
        K_ = alt_model.n_components
        d_ = alt_model.n_features_in_
        w, means, scales = unpack(theta, K_, d_)
        # overwrite params on the fitted object (fast)
        alt_model.weights_ = w
        alt_model.means_ = means
        alt_model.scales_ = scales
        return alt_model.score_samples(X)  # (n,)

    theta0 = pack(alt_model)
    p = theta0.size

    # ---- per-observation scores S via bulk central differences ----
    # S[i, j] = d/dtheta_j log p(x_i | theta)
    S = np.empty((n, p), dtype=float)
    for j in range(p):
        e = np.zeros(p, dtype=float)
        e[j] = fd_eps
        fp = score_samples_given_theta(theta0 + e)
        fm = score_samples_given_theta(theta0 - e)
        S[:, j] = (fp - fm) / (2.0 * fd_eps)

    J = S.T @ S

    # ---- observed information I = -H of total loglik via central differences ----
    def total_loglik(theta):
        return float(score_samples_given_theta(theta).sum())

    H = np.empty((p, p), dtype=float)
    f00 = total_loglik(theta0)

    for i in range(p):
        ei = np.zeros(p, dtype=float)
        ei[i] = fd_eps
        fpi = total_loglik(theta0 + ei)
        fmi = total_loglik(theta0 - ei)
        H[i, i] = (fpi - 2.0 * f00 + fmi) / (fd_eps**2)
        for j in range(i + 1, p):
            ej = np.zeros(p, dtype=float)
            ej[j] = fd_eps
            fpp = total_loglik(theta0 + ei + ej)
            fpm = total_loglik(theta0 + ei - ej)
            fmp = total_loglik(theta0 - ei + ej)
            fmm = total_loglik(theta0 - ei - ej)
            val = (fpp - fpm - fmp + fmm) / (4.0 * fd_eps**2)
            H[i, j] = val
            H[j, i] = val

    information_matrix = -0.5 * (H + H.T)

    # ---- lambdas from symmetric similarity transform ----
    evals_I, evecs_I = linalg.eigh(information_matrix)
    evals_I = np.maximum(evals_I, ridge_I)
    I_inv_sqrt = (evecs_I * (1.0 / np.sqrt(evals_I))) @ evecs_I.T

    B = I_inv_sqrt @ J @ I_inv_sqrt
    B = 0.5 * (B + B.T)
    lambdas = linalg.eigvalsh(B)

    # ---- p-value ----
    method = str(pvalue_method).lower()
    simulated_stats = None
    if method == "davies":
        p_value = davies_pvalue_weighted_chisq(
            lambdas, float(lr), tol=davies_tol, limit=davies_limit
        )
    elif method == "mc":
        Z = rng.standard_normal(size=(n_sim, lambdas.size))
        simulated_stats = (Z**2) @ lambdas
        p_value = float(np.mean(simulated_stats >= lr))
    else:
        raise ValueError("pvalue_method must be 'davies' or 'mc'.")

    out = {
        "lr": float(lr),
        "p_value": float(p_value),
        "lambdas": lambdas,
        "ll_L": float(ll_L),
        "ll_K": float(ll_K),
        "null_model": null_model,
        "alt_model": alt_model,
        "pvalue_method": method,
    }
    if return_simulated and method == "mc":
        out["simulated_stats"] = simulated_stats
    return out
