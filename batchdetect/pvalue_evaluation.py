import numpy as np
from scipy.stats import beta, gamma


def alpha_posterior_credible_interval(
    x_samples,
    prior_shape,
    prior_rate,
    alpha,
):
    """
    Compute a central posterior credible interval for alpha in the model

        alpha ~ Gamma(prior_shape, prior_rate)   # shape-rate
        x_i | alpha ~ Beta(alpha, 1)

    using the conjugate Gamma posterior.

    Parameters
    ----------
    x_samples : array-like
        Observed x values in (0, 1).
    prior_shape : float
        Shape parameter of the Gamma prior on alpha.
    prior_rate : float
        Rate parameter of the Gamma prior on alpha.
    alpha : float
        Credible interval level parameter. The function returns the
        interval from alpha/2 to 1 - alpha/2.

    Returns
    -------
    lower : float
        Lower bound of the posterior credible interval.
    upper : float
        Upper bound of the posterior credible interval.
    """

    x = np.asarray(x_samples, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x_samples must contain at least one value.")

    if not np.all((x > 0.0) & (x < 1.0)):
        raise ValueError("All x_samples must be in the open interval (0, 1).")

    if prior_shape <= 0.0 or prior_rate <= 0.0:
        raise ValueError("prior_shape and prior_rate must be positive.")

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    n = float(x.size)
    sum_log_x = float(np.log(x).sum())

    # Posterior parameters
    post_shape = prior_shape + n
    post_rate = prior_rate - sum_log_x  # > prior_rate > 0 since log(x_i) < 0

    if post_rate <= 0.0:
        raise ValueError(
            "Posterior rate is not positive. " "Check your prior and data."
        )

    # SciPy gamma: shape = a, scale = 1 / rate
    post_scale = 1.0 / post_rate

    lower = gamma.ppf(alpha / 2.0, a=post_shape, loc=0.0, scale=post_scale)
    upper = gamma.ppf(
        1.0 - alpha / 2.0, a=post_shape, loc=0.0, scale=post_scale
    )

    return lower, upper


def conservativeness_bound(null_pvals, alpha=0.05, gamma=0.05):
    """
    Compute an upper confidence bound on theta(alpha) = P(p <= alpha | null)
    using the exact Clopper-Pearson binomial method.

    Parameters
    ----------
    null_pvals : array-like
        P-values obtained under the null (e.g., from negative controls or null simulations).
    alpha : float, optional
        Significance threshold at which to assess conservativeness. Default is 0.05.
    gamma : float, optional
        1 - confidence level for the upper bound. For a 95% CI, use gamma = 0.05.

    Returns
    -------
    results : dict
        Dictionary with keys:
            "m"              : number of null samples
            "X"              : number of null p-values <= alpha
            "theta_hat"      : empirical proportion X / m
            "theta_upper"    : (1 - gamma) upper confidence bound for theta(alpha)
            "is_conservative": bool, True if theta_upper <= alpha
    """
    null_pvals = np.asarray(null_pvals)
    m = null_pvals.size
    if m == 0:
        raise ValueError("null_pvals must contain at least one value.")

    # Count how many null p-values are <= alpha
    X = np.sum(null_pvals <= alpha)

    # Exact Clopper-Pearson upper one-sided bound:
    # theta_upper is the (1 - gamma) quantile of Beta(X + 1, m - X)
    if X == m:
        theta_upper = 1.0
    else:
        theta_upper = beta.ppf(1.0 - gamma, X + 1, m - X)

    theta_hat = X / float(m)
    is_conservative = theta_upper <= alpha

    return {
        "m": m,
        "X": int(X),
        "theta_hat": theta_hat,
        "theta_upper": theta_upper,
        "is_conservative": is_conservative,
    }
