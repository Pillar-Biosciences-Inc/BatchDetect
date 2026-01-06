import numpy as np
from scipy.special import logsumexp



def _logmeanexp(a):
    """
    Compute log(mean(exp(a))) in a numerically stable way.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    scalar
        The log of the mean of the exponentials of the input elements.
    """
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    return m + np.log(np.mean(np.exp(a - m)))


def _effective_sample_size_from_logw(logw):
    """
    Compute Effective Sample Size (ESS) from log-weights.

    ESS = (sum w)^2 / sum w^2

    Parameters
    ----------
    logw : array_like
        Log-weights of particles.

    Returns
    -------
    float
        The effective sample size. Returns 0.0 if input is empty or sum of squared weights is 0.
    """
    # ESS = (sum w)^2 / sum w^2 computed stably in log-space
    logw = np.asarray(logw, dtype=float)
    if logw.size == 0:
        return 0.0
    m = np.max(logw)
    w = np.exp(logw - m)
    s1 = np.sum(w)
    s2 = np.sum(w * w)
    if s2 <= 0:
        return 0.0
    return float((s1 * s1) / s2)


def _log_complete_likelihood_isotropic_gaussian(X, z, mus, taus):
    """
    Compute the complete-data log-likelihood for an isotropic Gaussian Mixture Model.

    log p(X, z | mu, tau) = sum_n log N(x_n | mu_{z_n}, tau_{z_n}^{-1} I)

    Parameters
    ----------
    X : array_like, shape (N, D)
        Observed data.
    z : array_like, shape (N,)
        Component assignments.
    mus : array_like, shape (K, D)
        Component means.
    taus : array_like, shape (K,)
        Component precisions (1/variance).

    Returns
    -------
    float
        Total log-likelihood.
    """
    X = np.asarray(X, dtype=float)
    z = np.asarray(z, dtype=np.int64)
    N, D = X.shape
    K = mus.shape[0]
    ll = 0.0
    const = -0.5 * D * np.log(2.0 * np.pi)
    for k in range(K):
        idx = z == k
        nk = int(np.sum(idx))
        if nk == 0:
            continue
        diff = X[idx] - mus[k]
        sq = np.sum(diff * diff)
        ll += nk * (const + 0.5 * D * np.log(taus[k])) - 0.5 * taus[k] * sq
    return float(ll)


def _sample_dirichlet(rng, alpha):
    """
    Sample from a Dirichlet distribution, handling validity checks.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator instance.
    alpha : array_like
        Concentration parameters. Must be strictly positive.

    Returns
    -------
    ndarray
        Sampled probability vector summing to 1.

    Raises
    ------
    ValueError
        If any alpha <= 0.
    """
    alpha = np.asarray(alpha, dtype=float)
    # np.random.Generator.dirichlet requires all alpha > 0
    if np.any(alpha <= 0):
        raise ValueError("Dirichlet parameters must be > 0.")
    return rng.dirichlet(alpha)


def _sample_prior_state(rng, X, K, alpha0, m0, kappa0, a0, b0):
    """
    Sample initial state from the prior distribution (beta=0).

    The model is:
      pi ~ Dir(alpha0)
      tau_k ~ Gamma(a0, b0) (shape/rate parameterization)
      mu_k | tau_k ~ N(m0, (kappa0 * tau_k)^-1 I)
      z_n | pi ~ Cat(pi)

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    X : array_like
        Data (used for shape N, D only).
    K : int
        Number of components.
    alpha0 : float
        Dirichlet concentration parameter.
    m0 : array_like
        &Prior mean for component means.
    kappa0 : float
        Scaling factor for prior variance of means.
    a0 : float
        Gamma shape parameter for precision.
    b0 : float
        Gamma rate parameter for precision.

    Returns
    -------
    tuple
        (pi, mus, taus, z) sampled from the prior.
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape

    pi = _sample_dirichlet(rng, np.full(K, alpha0))

    taus = rng.gamma(shape=a0, scale=1.0 / b0, size=K)  # rate=b0
    mus = np.empty((K, D), dtype=float)
    for k in range(K):
        cov_scale = 1.0 / (kappa0 * taus[k])  # variance per dim
        mus[k] = m0 + rng.normal(loc=0.0, scale=np.sqrt(cov_scale), size=D)

    # sample z from pi
    z = rng.choice(K, size=N, p=pi)

    return pi, mus, taus, z


def _gibbs_sweep_tempered(
    rng, X, pi, mus, taus, z, beta, alpha0, m0, kappa0, a0, b0
):
    """
    Perform one tempered Gibbs sampling sweep.

    Targets the tempered posterior:
      f_beta(pi, mu, tau, z) ∝ p(pi) p(mu,tau) p(z|pi) [p(X|z,mu,tau)]^beta

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    X : array_like
        Observed data.
    pi : array_like
        Current mixture weights.
    mus : array_like
        Current component means.
    taus : array_like
        Current component precisions.
    z : array_like
        Current assignments.
    beta : float
        Inverse temperature (0 <= beta <= 1).
    alpha0, m0, kappa0, a0, b0 : float
        Hyperparameters.

    Returns
    -------
    tuple
        Updated (pi, mus, taus, z).
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    K = mus.shape[0]

    # ---- Sample pi | z  (independent of beta) ----
    Nk = np.bincount(z, minlength=K).astype(float)
    pi = _sample_dirichlet(rng, alpha0 + Nk)

    # ---- Sample (mu_k, tau_k) | z, X with tempered likelihood ----
    # Tempering corresponds to scaling sufficient statistics by beta.
    for k in range(K):
        nk = int(Nk[k])
        if nk == 0 or beta == 0.0:
            # If no assigned points or beta=0, posterior equals prior
            taus[k] = rng.gamma(shape=a0, scale=1.0 / b0)
            cov_scale = 1.0 / (kappa0 * taus[k])
            mus[k] = m0 + rng.normal(0.0, np.sqrt(cov_scale), size=D)
            continue

        idx = z == k
        Xk = X[idx]
        xbar = np.mean(Xk, axis=0)

        diff = Xk - xbar
        Sk = float(np.sum(diff * diff))  # sum ||x - xbar||^2

        # Tempered "effective sample size"
        nkb = beta * nk

        kappa_n = kappa0 + nkb
        m_n = (kappa0 * m0 + nkb * xbar) / kappa_n

        a_n = a0 + 0.5 * nkb * D

        mean_diff = xbar - m0
        mean_diff_sq = float(np.dot(mean_diff, mean_diff))

        b_n = b0 + 0.5 * (beta * Sk + (kappa0 * nkb / kappa_n) * mean_diff_sq)

        # Sample tau | ...  then mu | tau, ...
        taus[k] = rng.gamma(shape=a_n, scale=1.0 / b_n)
        cov_scale = 1.0 / (kappa_n * taus[k])
        mus[k] = m_n + rng.normal(0.0, np.sqrt(cov_scale), size=D)

    # ---- Sample z_n | pi, mu, tau with tempered likelihood ----
    # log p(z_n=k | ...) ∝ log pi_k + beta * log N(x_n | mu_k, tau_k^-1 I)
    # Compute vectorized log probs: N x K
    log_pi = np.log(pi + 1e-300)
    const = -0.5 * D * np.log(2.0 * np.pi)

    # Precompute per-k constants
    log_norm_k = const + 0.5 * D * np.log(taus)  # size K

    # For each k: -0.5 * tau_k * ||x - mu_k||^2
    # Compute squared distances efficiently
    # dist2[n,k] = ||X[n] - mus[k]||^2
    # Use (x^2 + m^2 - 2 x.m)
    x2 = np.sum(X * X, axis=1, keepdims=True)  # (N,1)
    m2 = np.sum(mus * mus, axis=1, keepdims=True).T  # (1,K)
    xm = X @ mus.T  # (N,K)
    dist2 = x2 + m2 - 2.0 * xm  # (N,K)
    dist2 = np.maximum(dist2, 0.0)

    log_like = log_norm_k[None, :] - 0.5 * (taus[None, :] * dist2)

    logp = log_pi[None, :] + beta * log_like
    logp -= logsumexp(logp, axis=1, keepdims=True)
    p = np.exp(logp)

    # sample categorical for each row
    # vectorized sampling by cumulative sums
    u = rng.random(size=N)
    cdf = np.cumsum(p, axis=1)
    z = np.sum(u[:, None] > cdf, axis=1).astype(np.int64)

    return pi, mus, taus, z


def ais_log_evidence_isotropic_gmm(
    X,
    K,
    n_particles=128,
    n_intermediate=200,
    n_gibbs_sweeps_per_beta=1,
    schedule_power=2.0,
    alpha0=1.0,
    kappa0=1e-2,
    a0=1e-2,
    b0=1e-2,
    m0=None,
    random_state=0,
):
    """
    Estimate log-evidence log p(X | K) using Annealed Importance Sampling (AIS).

    The model assumed is a Bayesian Isotropic Gaussian Mixture:
       pi ~ Dirichlet(alpha0 * 1_K)
       tau_k ~ Gamma(a0, b0)
       mu_k | tau_k ~ N(m0, (kappa0 * tau_k)^-1 I)

    The AIS path anneals from the prior (beta=0) to the posterior (beta=1).

    Parameters
    ----------
    X : array_like, shape (N, D)
        Observed data matrix.
    K : int
        Number of components to test.
    n_particles : int, default=128
        Number of AIS particles (chains).
    n_intermediate : int, default=200
        Number of annealing steps (beta distributions).
    n_gibbs_sweeps_per_beta : int, default=1
        Number of MCMC transitions per temperature step.
    schedule_power : float, default=2.0
        Power for the annealing schedule (beta_i = (i/T)^power).
    alpha0 : float, default=1.0
        Dirichlet concentration.
    kappa0 : float, default=0.01
        Prior mean precision scaling.
    a0 : float, default=0.01
        Gamma shape for precision.
    b0 : float, default=0.01
        Gamma rate for precision.
    m0 : array_like or None
        Prior mean for mu. If None, defaults to global mean of X.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    dict
        Results dictionary containing:
        - "logZ_hat": float, estimated log evidence.
        - "logw": array, shape (n_particles,), final log weights.
        - "ess": float, effective sample size of the weights.
        - "betas": array, annealing schedule used.
        - "config": dict, input configuration.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D: (n_samples, n_features).")
    N, D = X.shape
    K = int(K)

    rng = np.random.default_rng(random_state)

    if m0 is None:
        m0 = np.mean(X, axis=0)
    m0 = np.asarray(m0, dtype=float)
    if m0.shape != (D,):
        raise ValueError("m0 must have shape (n_features,)")

    # Annealing schedule: betas[0]=0, betas[-1]=1
    t = np.linspace(0.0, 1.0, n_intermediate + 1)
    betas = t**schedule_power
    betas[0] = 0.0
    betas[-1] = 1.0

    # Initialize particles from beta=0 distribution
    pis = np.empty((n_particles, K), dtype=float)
    mus = np.empty((n_particles, K, D), dtype=float)
    taus = np.empty((n_particles, K), dtype=float)
    zs = np.empty((n_particles, N), dtype=np.int64)

    for i in range(n_particles):
        pi_i, mu_i, tau_i, z_i = _sample_prior_state(
            rng, X, K, alpha0, m0, kappa0, a0, b0
        )
        pis[i] = pi_i
        mus[i] = mu_i
        taus[i] = tau_i
        zs[i] = z_i

    logw = np.zeros(n_particles, dtype=float)

    # AIS loop
    for j in range(len(betas) - 1):
        b0j = float(betas[j])
        b1j = float(betas[j + 1])
        db = b1j - b0j
        if db < 0:
            raise ValueError("Betas must be nondecreasing.")

        # Incremental weights using current state at beta=b0j
        if db != 0.0:
            for i in range(n_particles):
                ll = _log_complete_likelihood_isotropic_gaussian(
                    X, zs[i], mus[i], taus[i]
                )
                logw[i] += db * ll

        # Move each particle with tempered Gibbs targeting beta=b1j
        if b1j != b0j:
            for i in range(n_particles):
                for _ in range(n_gibbs_sweeps_per_beta):
                    pi_i, mu_i, tau_i, z_i = _gibbs_sweep_tempered(
                        rng,
                        X,
                        pis[i],
                        mus[i],
                        taus[i],
                        zs[i],
                        beta=b1j,
                        alpha0=alpha0,
                        m0=m0,
                        kappa0=kappa0,
                        a0=a0,
                        b0=b0,
                    )
                    pis[i], mus[i], taus[i], zs[i] = pi_i, mu_i, tau_i, z_i

    logZ_hat = _logmeanexp(logw)
    ess = _effective_sample_size_from_logw(logw)

    return {
        "logZ_hat": float(logZ_hat),
        "logw": logw,
        "ess": float(ess),
        "betas": betas,
        "config": {
            "K": K,
            "n_particles": n_particles,
            "n_intermediate": n_intermediate,
            "n_gibbs_sweeps_per_beta": n_gibbs_sweeps_per_beta,
            "schedule_power": schedule_power,
            "alpha0": float(alpha0),
            "kappa0": float(kappa0),
            "a0": float(a0),
            "b0": float(b0),
        },
    }


def ais_select_K(X, K_list, **ais_kwargs):
    """
    Perform Bayesian Model Selection for K (number of clusters) using AIS.

    Runs `ais_log_evidence_isotropic_gmm` for each K in `K_list` and ranks them
    by the estimated log-evidence.

    Parameters
    ----------
    X : array_like, shape (N, D)
        Observed data.
    K_list : list of int
        List of K values to test (e.g., [1, 2, 3]).
    **ais_kwargs
        Additional keyword arguments passed to `ais_log_evidence_isotropic_gmm`.

    Returns
    -------
    list of tuples
        Results sorted by logZ_hat descending. Each tuple is:
        (K, logZ_hat, ess, res_dict)
    """
    results = []
    for K in K_list:
        r = ais_log_evidence_isotropic_gmm(X, K, **ais_kwargs)
        results.append((K, r["logZ_hat"], r["ess"], r))
    results.sort(key=lambda t: t[1], reverse=True)
    return results

