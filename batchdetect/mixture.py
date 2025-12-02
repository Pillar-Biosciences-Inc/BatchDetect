from typing import Any, Callable, Dict, Union

from scipy.special import gammaln
import numpy as np

import numpy as np
from scipy.special import gammaln


class HeavyMixture:
    """Mixture of independent components with isotropic scales.

    This estimator uses expectation-maximization (EM) to fit a mixture of
    independent components with one scale parameter per component.

    Supported component distributions
    ---------------------------------
    component_distribution = "laplace" (default)
        p(x | k) = (1 / (2 * b_k)) ** d * exp(-||x - mu_k||_1 / b_k)

    component_distribution = "gaussian" or "normal"
        p(x | k) = (1 / (sqrt(2 * pi) * sigma_k) ** d)
                   * exp(-||x - mu_k||_2^2 / (2 * sigma_k^2))

    component_distribution = "student_t" or "t"
        Multivariate isotropic Student-t with degrees of freedom t_df (default 7):
        p(x | k) ∝ (1 + ||x - mu_k||_2^2 / (t_df * sigma_k^2))^{-(t_df + d)/2}

    component_distribution = "gennorm" or "generalized_normal"
        Product of 1D generalized normal distributions with shape gennorm_beta:
        f(z) = beta / (2 * alpha * Gamma(1 / beta)) * exp(-( |z| / alpha )**beta)
        with a single isotropic scale alpha per component.

    component_distribution = "hypsecant" or "hyperbolic_secant" or "sech"
        Product of 1D hyperbolic secant distributions:
        f(z) = 1 / (2 * s) * sech(pi * (z - mu) / (2 * s))

    Parameters
    ----------
    n_components : int, default=1
        Number of mixture components.
    component_distribution : {"laplace", "gaussian", "normal",
                              "student_t", "t",
                              "gennorm", "generalized_normal",
                              "hypsecant", "hyperbolic_secant", "sech"},
                              default="laplace"
        Component distribution family.
    t_df : float, default=7.0
        Degrees of freedom for Student-t components (if used). Must be > 2.
    gennorm_beta : float, default=1.5
        Shape parameter beta for generalized normal (if used). Must be > 0.
        beta = 1 -> Laplace-like; beta = 2 -> Gaussian.
    tol : float, default=1e-4
        Convergence threshold on the change in lower bound.
    max_iter : int, default=100
        Maximum number of EM iterations to perform for a single run.
    n_init : int, default=1
        Number of random initializations. The best run is kept.
    init_params : {"kmeans", "random"}, default="kmeans"
        Method used to initialize the component means.
    reg_b : float, default=1e-6
        Non-negative regularization for the scale parameters. Acts as a lower
        bound on the scale (b_k, sigma_k, etc.).
    warm_start : bool, default=False
        If True and the model was already fitted, reuse the solution of
        the previous call to fit as initialization, and perform a single
        EM run. If False, parameters are reinitialized n_init times.
    random_state : int or numpy.random.RandomState or None, default=None
        Random number generator used for initialization and sampling.
    verbose : int, default=0
        Verbosity level. Set to 1 to print basic convergence messages.

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
        Mixture weights for each component.
    means_ : ndarray of shape (n_components, n_features)
        Location vector of each component.
    scales_ : ndarray of shape (n_components,)
        Positive scale parameter of each component (b_k, sigma_k, etc.).
    n_iter_ : int
        Number of EM iterations performed for the best run.
    lower_bound_ : float
        Final average log likelihood of the training data under the model.
    n_features_in_ : int
        Number of features in the input passed to fit.
    """

    def __init__(
        self,
        n_components=1,
        component_distribution="laplace",
        t_df=7.0,
        gennorm_beta=1.5,
        tol=1e-4,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        reg_b=1e-6,
        warm_start=False,
        random_state=None,
        verbose=0,
    ):
        self.n_components = n_components
        self.component_distribution = component_distribution
        self.t_df = t_df
        self.gennorm_beta = gennorm_beta
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.reg_b = reg_b
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _check_random_state(self, random_state):
        """Return a RandomState instance from None, int, or RandomState."""
        if random_state is None:
            random_state = self.random_state
        if random_state is None or random_state is np.random:
            return np.random.mtrand._rand
        if isinstance(random_state, (int, np.integer)):
            return np.random.RandomState(random_state)
        if isinstance(random_state, np.random.RandomState):
            return random_state
        raise ValueError(
            "Invalid random_state argument. Expected None, int, or RandomState."
        )

    def _check_array(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        return X

    def _check_X(self, X):
        X = self._check_array(X)
        if (
            hasattr(self, "n_features_in_")
            and X.shape[1] != self.n_features_in_
        ):
            raise ValueError(
                "X has {} features, but this model was fitted with {} features.".format(
                    X.shape[1],
                    self.n_features_in_,
                )
            )
        return X

    def _logsumexp(self, a, axis=None):
        a = np.asarray(a)
        a_max = np.max(a, axis=axis, keepdims=True)
        tmp = np.exp(a - a_max)
        s = np.sum(tmp, axis=axis, keepdims=True)
        out = np.log(s) + a_max
        if axis is not None:
            out = np.squeeze(out, axis=axis)
        return out

    def _normalized_component_distribution(self):
        """Return normalized component distribution string."""
        cd = str(self.component_distribution).lower()

        if cd == "normal":
            cd = "gaussian"
        if cd in ("student_t", "studentt"):
            cd = "student_t"
        if cd in ("t",):
            cd = "student_t"
        if cd in ("gennorm", "generalized_normal", "gen_normal"):
            cd = "gennorm"
        if cd in ("hypsecant", "hyperbolic_secant", "sech"):
            cd = "hypsecant"

        if cd not in ("laplace", "gaussian", "student_t", "gennorm", "hypsecant"):
            raise ValueError(
                "Unsupported component_distribution {!r}. "
                "Expected one of: 'laplace', 'gaussian', 'normal', "
                "'student_t', 't', 'gennorm', 'generalized_normal', "
                "'hypsecant', 'hyperbolic_secant', 'sech'.".format(
                    self.component_distribution
                )
            )

        if cd == "student_t":
            if self.t_df <= 2.0:
                raise ValueError("t_df must be > 2 for Student-t.")
        if cd == "gennorm":
            if self.gennorm_beta <= 0.0:
                raise ValueError("gennorm_beta must be > 0.")

        return cd

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_parameters(self, X, rng):
        n_samples, n_features = X.shape
        K = self.n_components
        cd = self._normalized_component_distribution()

        # Initialize weights uniformly
        weights = np.full(K, 1.0 / K)

        # Initialize means
        if self.init_params == "random" or K == 1:
            indices = rng.randint(0, n_samples, size=K)
            means = X[indices].copy()
        elif self.init_params == "kmeans":
            # Simple k-means style initialization using L1 distance and medians
            means = X[rng.choice(n_samples, size=K, replace=False)].copy()
            for _ in range(10):
                diffs = X[:, None, :] - means[None, :, :]
                l1 = np.abs(diffs).sum(axis=2)  # (n_samples, K)
                labels = np.argmin(l1, axis=1)
                for k in range(K):
                    mask = labels == k
                    if np.any(mask):
                        means[k] = np.median(X[mask], axis=0)
        else:
            raise ValueError("init_params must be 'kmeans' or 'random'.")

        # Initialize scales
        diffs = X[:, None, :] - means[None, :, :]
        n_features_float = float(n_features)

        if cd == "laplace":
            # Average L1 deviation from component mean
            l1 = np.abs(diffs).sum(axis=2)
            scales = np.maximum(self.reg_b, np.mean(l1, axis=0) / n_features_float)
        elif cd in ("gaussian", "student_t", "hypsecant"):
            # Use average L2 distance to define initial sigma
            l2_sq = np.sum(diffs * diffs, axis=2)
            var = np.maximum(self.reg_b ** 2, np.mean(l2_sq, axis=0) / n_features_float)
            scales = np.sqrt(var)
        elif cd == "gennorm":
            beta = self.gennorm_beta
            abs_beta = np.abs(diffs) ** beta
            # Average |x - mu|^beta per feature
            m_beta = np.mean(abs_beta.sum(axis=2) / n_features_float, axis=0)
            # From E|Z|^beta = alpha^beta / beta => alpha^beta = beta * E|Z|^beta
            alpha_beta = np.maximum(self.reg_b ** beta, beta * m_beta)
            scales = alpha_beta ** (1.0 / beta)
        else:
            raise RuntimeError("Unexpected component_distribution in initialization.")

        return weights, means, scales

    # ------------------------------------------------------------------
    # E-step and helpers
    # ------------------------------------------------------------------
    def _estimate_log_prob(self, X):
        cd = self._normalized_component_distribution()
        diffs = X[:, None, :] - self.means_[None, :, :]  # (n_samples, K, d)
        n_features = self.n_features_in_

        if cd == "laplace":
            # Laplace with L1 norm
            l1 = np.abs(diffs).sum(axis=2)  # (n_samples, K)
            log_norm = -n_features * (np.log(2.0) + np.log(self.scales_))  # (K,)
            return log_norm[None, :] - l1 / self.scales_[None, :]

        if cd == "gaussian":
            # Gaussian with isotropic sigma
            l2_sq = np.sum(diffs * diffs, axis=2)
            var = self.scales_ ** 2  # (K,)
            log_norm = -0.5 * n_features * (np.log(2.0 * np.pi) + np.log(var))
            return log_norm[None, :] - 0.5 * l2_sq / var[None, :]

        if cd == "student_t":
            # Multivariate isotropic Student-t with df = t_df
            nu = float(self.t_df)
            l2_sq = np.sum(diffs * diffs, axis=2)
            s2 = self.scales_ ** 2  # (K,)

            # Normalization constant:
            # log C = gammaln((nu + d)/2) - gammaln(nu/2)
            #         - 0.5 * d * (log(nu * pi) + log(s2))
            # Note: log(s2) is per component
            d = float(n_features)
            log_C = (
                gammaln((nu + d) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * d * (np.log(nu * np.pi) + np.log(s2))
            )  # (K,)

            # Quadratic form inside the log term
            q = l2_sq / (nu * s2[None, :])  # (n_samples, K)
            return log_C[None, :] - 0.5 * (nu + d) * np.log1p(q)

        if cd == "gennorm":
            # Generalized normal, product of independent 1D
            beta = float(self.gennorm_beta)
            abs_diff_beta = np.abs(diffs) ** beta  # (n_samples, K, d)
            # sum over features of (|x - mu| / alpha)^beta
            # = sum_j |x_j - mu_j|^beta / alpha^beta
            alpha_beta = self.scales_ ** beta  # (K,)
            r_beta = abs_diff_beta.sum(axis=2) / alpha_beta[None, :]  # (n_samples, K)

            # log normalizing constant per dimension:
            # 1D: log f = log(beta) - log(2 * alpha) - log(Gamma(1 / beta)) - (|x|/alpha)^beta
            # d dims independent => multiply constants by d
            log_const_1d = np.log(beta) - np.log(2.0 * self.scales_) - gammaln(1.0 / beta)
            log_const = n_features * log_const_1d  # (K,)
            return log_const[None, :] - r_beta

        if cd == "hypsecant":
            # Product of 1D hyperbolic secant distributions:
            # f(z) = 1/(2 s) * sech(pi * (z - mu) / (2 s))
            # sech(u) = 1 / cosh(u)
            s = self.scales_[None, None, :]  # (1, 1, K)
            u = (np.pi / 2.0) * diffs / s  # (n_samples, K, d)
            # log sech(u) = -log(cosh(u))
            log_sech = -np.log(np.cosh(u))
            log_const = -n_features * np.log(2.0 * self.scales_)  # (K,)
            return log_const[None, :] + log_sech.sum(axis=2)

        raise RuntimeError("Unexpected component_distribution in _estimate_log_prob.")

    def _estimate_log_resp(self, X):
        log_prob = self._estimate_log_prob(X)  # (n_samples, K)
        log_prob_weighted = log_prob + np.log(self.weights_)[None, :]
        log_prob_norm = self._logsumexp(log_prob_weighted, axis=1)  # (n_samples,)
        log_resp = log_prob_weighted - log_prob_norm[:, None]
        return log_resp, log_prob_norm.mean()

    # ------------------------------------------------------------------
    # M-step helpers
    # ------------------------------------------------------------------
    def _weighted_median_1d(self, x, w):
        """Return weighted median of a 1D array.

        If all weights are zero, fall back to the unweighted median.
        """
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        if x.shape[0] != w.shape[0]:
            raise ValueError("x and w must have the same length.")
        total = w.sum()
        if total <= 0:
            return np.median(x)
        order = np.argsort(x)
        x_sorted = x[order]
        w_sorted = w[order]
        cum_w = np.cumsum(w_sorted)
        cutoff = 0.5 * total
        idx = np.searchsorted(cum_w, cutoff)
        idx = min(idx, x_sorted.size - 1)
        return x_sorted[idx]

    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        K = self.n_components
        nk = resp.sum(axis=0)  # responsibilities per component, shape (K,)
        cd = self._normalized_component_distribution()

        # Update weights
        weights = nk / n_samples
        eps = np.finfo(float).eps
        weights = np.maximum(weights, eps)
        weights = weights / weights.sum()

        means = np.empty((K, n_features), dtype=float)
        scales = np.empty(K, dtype=float)
        n_features_float = float(n_features)

        if cd == "laplace":
            # Update means via weighted medians per dimension
            global_median = np.median(X, axis=0)

            for k in range(K):
                w = resp[:, k]
                if nk[k] <= eps:
                    # Component effectively empty: fall back to global median
                    means[k] = global_median
                else:
                    for j in range(n_features):
                        means[k, j] = self._weighted_median_1d(X[:, j], w)

            # Update scales via weighted average absolute deviation
            diffs = X[:, None, :] - means[None, :, :]
            l1 = np.abs(diffs).sum(axis=2)  # (n_samples, K)

            # Precompute a global L1 scale as fallback
            l1_global = np.abs(X - global_median[None, :]).sum(axis=1)
            global_scale = np.maximum(self.reg_b, np.mean(l1_global) / n_features_float)

            for k in range(K):
                if nk[k] <= eps:
                    scales[k] = global_scale
                else:
                    scales[k] = np.maximum(
                        self.reg_b,
                        np.sum(resp[:, k] * l1[:, k]) / (n_features_float * nk[k]),
                    )

        else:
            # For all other distributions, we use weighted means for locations
            # and distribution-specific moment-based scales.
            global_mean = np.mean(X, axis=0)

            diffs_global = X - global_mean[None, :]
            l2_sq_global = np.sum(diffs_global * diffs_global, axis=1)
            # global variance per feature (average over features)
            var_global = np.mean(l2_sq_global) / n_features_float
            var_global = max(var_global, self.reg_b ** 2)

            if cd == "gaussian":
                global_scale = np.sqrt(var_global)
            elif cd == "student_t":
                nu = float(self.t_df)
                # For Student-t, var = nu / (nu - 2) * scale^2. Invert:
                # scale^2 = (nu - 2) / nu * var
                global_scale = np.sqrt((nu - 2.0) / nu * var_global)
            elif cd == "hypsecant":
                # For the chosen parameterization, we approximate var ~ scale^2
                global_scale = np.sqrt(var_global)
            elif cd == "gennorm":
                beta = float(self.gennorm_beta)
                abs_beta_global = np.abs(diffs_global) ** beta
                m_beta_global = np.mean(abs_beta_global)  # average over all dims
                alpha_beta_global = max(self.reg_b ** beta, beta * m_beta_global)
                global_scale = alpha_beta_global ** (1.0 / beta)
            else:
                raise RuntimeError("Unexpected distribution in M-step global scale.")

            for k in range(K):
                w = resp[:, k]
                if nk[k] <= eps:
                    means[k] = global_mean
                    scales[k] = global_scale
                    continue

                w_sum = nk[k]
                means_k = np.sum(w[:, None] * X, axis=0) / w_sum
                means[k] = means_k

                diffs_k = X - means_k[None, :]

                if cd in ("gaussian", "student_t", "hypsecant"):
                    l2_sq_k = np.sum(diffs_k * diffs_k, axis=1)
                    var_k = np.sum(w * l2_sq_k) / (n_features_float * w_sum)
                    var_k = max(var_k, self.reg_b ** 2)

                    if cd == "gaussian":
                        scale_k = np.sqrt(var_k)
                    elif cd == "student_t":
                        nu = float(self.t_df)
                        scale_k = np.sqrt((nu - 2.0) / nu * var_k)
                    else:  # hypsecant
                        scale_k = np.sqrt(var_k)

                    scales[k] = max(self.reg_b, scale_k)

                elif cd == "gennorm":
                    beta = float(self.gennorm_beta)
                    abs_beta_k = np.abs(diffs_k) ** beta
                    # Average |x - mu|^beta across samples and dimensions
                    m_beta_k = np.sum(w[:, None] * abs_beta_k) / (w_sum * n_features_float)
                    alpha_beta_k = max(self.reg_b ** beta, beta * m_beta_k)
                    scale_k = alpha_beta_k ** (1.0 / beta)
                    scales[k] = max(self.reg_b, scale_k)
                else:
                    raise RuntimeError("Unexpected distribution in M-step loop.")

        return weights, means, scales

    # ------------------------------------------------------------------
    # Single EM run
    # ------------------------------------------------------------------
    def _fit_single_em(self, X, rng):
        # Initialize
        if self.warm_start and hasattr(self, "weights_"):
            weights = self.weights_.copy()
            means = self.means_.copy()
            scales = self.scales_.copy()
        else:
            weights, means, scales = self._initialize_parameters(X, rng)

        self.weights_ = weights
        self.means_ = means
        self.scales_ = scales

        lower_bound = -np.inf
        for n_iter in range(1, self.max_iter + 1):
            # E-step
            log_resp, new_lower_bound = self._estimate_log_resp(X)
            resp = np.exp(log_resp)

            # M-step
            weights, means, scales = self._m_step(X, resp)
            self.weights_ = weights
            self.means_ = means
            self.scales_ = scales

            change = new_lower_bound - lower_bound
            lower_bound = new_lower_bound
            if self.verbose:
                print(
                    "[LaplaceMixture] EM iter {}, lower bound = {:.6f}, change = {:.6e}".format(
                        n_iter,
                        lower_bound,
                        change,
                    )
                )
            if abs(change) < self.tol:
                break

        return lower_bound, n_iter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """Fit the mixture model to data X using EM."""
        X = self._check_array(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        rng = self._check_random_state(self.random_state)

        best_lower_bound = -np.inf
        best_params = None
        best_n_iter = 0

        for init in range(self.n_init):
            # Use a different seed for each init to avoid identical runs
            rng_init = self._check_random_state(rng.randint(0, 2**31 - 1))
            lower_bound, n_iter = self._fit_single_em(X, rng_init)

            if self.verbose:
                print(
                    "[LaplaceMixture] Finished init {}, "
                    "lower bound = {:.6f}, n_iter = {:d}".format(
                        init + 1, lower_bound, n_iter
                    )
                )
            if lower_bound > best_lower_bound or best_params is None:
                best_lower_bound = lower_bound
                best_n_iter = n_iter
                best_params = (
                    self.weights_.copy(),
                    self.means_.copy(),
                    self.scales_.copy(),
                )

        self.weights_, self.means_, self.scales_ = best_params
        self.lower_bound_ = best_lower_bound
        self.n_iter_ = best_n_iter
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "weights_"):
            raise RuntimeError(
                "This LaplaceMixture instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    def score_samples(self, X):
        """Compute log probability of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        log_prob : ndarray of shape (n_samples,)
            Log probability of each sample under the mixture model.
        """
        self._check_is_fitted()
        X = self._check_X(X)
        log_prob = self._estimate_log_prob(X) + np.log(self.weights_)[None, :]
        return self._logsumexp(log_prob, axis=1)

    def score(self, X, y=None):
        """Compute average log likelihood of X under the fitted model."""
        return np.mean(self.score_samples(X))

    def predict_proba(self, X):
        """Compute posterior probabilities (responsibilities) for each sample."""
        self._check_is_fitted()
        X = self._check_X(X)
        log_resp, _ = self._estimate_log_resp(X)
        return np.exp(log_resp)

    def predict(self, X):
        """Assign each sample in X to the most likely component."""
        resp = self.predict_proba(X)
        return np.argmax(resp, axis=1)

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the fitted mixture model.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw.
        random_state : int or numpy.random.RandomState or None, default=None
            Random number generator.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Generated samples.
        y : ndarray of shape (n_samples,)
            Component labels for each sample.
        """
        self._check_is_fitted()
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        rng = self._check_random_state(random_state)

        n_features = self.n_features_in_
        K = self.n_components
        cd = self._normalized_component_distribution()

        # Draw component labels
        counts = rng.multinomial(n_samples, self.weights_)
        labels = np.repeat(np.arange(K), counts)

        # Generate samples for each component
        X = np.empty((n_samples, n_features), dtype=float)
        start = 0
        for k in range(K):
            nk = counts[k]
            if nk == 0:
                continue
            loc = self.means_[k]
            scale = self.scales_[k]

            if cd == "laplace":
                X_k = rng.laplace(loc=loc, scale=scale, size=(nk, n_features))
            elif cd == "gaussian":
                X_k = rng.normal(loc=loc, scale=scale, size=(nk, n_features))
            elif cd == "student_t":
                # Student-t via normal / sqrt(chi2/df)
                nu = float(self.t_df)
                # Shared chi2 per sample, independent dims
                v = rng.chisquare(df=nu, size=nk)  # (nk,)
                g = rng.normal(loc=0.0, scale=1.0, size=(nk, n_features))
                X_k = loc + scale * g / np.sqrt(v[:, None] / nu)
            elif cd == "gennorm":
                beta = float(self.gennorm_beta)
                # |X - mu|^beta ~ Gamma(shape=1/beta, scale=1)
                # X = mu + scale * S * Y^{1/beta}, S = +/- 1
                shape = 1.0 / beta
                Y = rng.gamma(shape=shape, scale=1.0, size=(nk, n_features))
                S = rng.choice([-1.0, 1.0], size=(nk, n_features))
                X_k = loc + scale * S * (Y ** (1.0 / beta))
            elif cd == "hypsecant":
                # Standard hyperbolic secant sampling:
                # If U ~ Uniform(0,1), then
                # Z = 2/pi * asinh( tan( pi*(U - 0.5) ) ) approx has sech distribution.
                U = rng.uniform(size=(nk, n_features))
                T = np.tan(np.pi * (U - 0.5))
                Z = (2.0 / np.pi) * np.arcsinh(T)
                X_k = loc + scale * Z
            else:
                raise RuntimeError("Unexpected distribution in sample().")

            X[start : start + nk, :] = X_k
            start += nk

        # Shuffle to avoid grouped components
        perm = rng.permutation(n_samples)
        return X[perm], labels[perm]


def parametric_bootstrap_lrt(
    X,
    null_model_factory: Callable[[], Any],
    alt_model_factory: Callable[[], Any],
    n_bootstrap: int = 500,
    random_state: Union[int, np.random.Generator, None] = None,
) -> Dict[str, Any]:
    """
    Parametric bootstrap likelihood ratio test for nested mixture models.

    This function tests a null mixture model against a more flexible
    alternative mixture model using a likelihood ratio statistic whose
    null distribution is approximated by parametric bootstrap.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_features)
        Observed data to fit both null and alternative models.
    null_model_factory : callable
        A zero-argument callable that returns a new, unfitted instance
        of the null model. The model must implement:
        - fit(X)
        - score(X): average log-likelihood per sample
        - sample(n_samples): returns either X_sim or (X_sim, labels)
    alt_model_factory : callable
        A zero-argument callable that returns a new, unfitted instance
        of the alternative model, with the same interface as the null model.
    n_bootstrap : int, default=500
        Number of bootstrap replicates used to approximate the null
        distribution of the likelihood ratio statistic.
    random_state : int, numpy.random.Generator, or None, default=None
        Random seed or Generator for reproducibility. This controls the
        randomness used to draw bootstrap seeds; the models themselves
        are created by the factories and may have their own random_state
        parameters.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - "statistic": float
            Observed likelihood ratio statistic:
            2 * (loglik_alt - loglik_null).
        - "p_value": float
            Parametric bootstrap p-value:
            (1 + sum(LR_boot >= LR_obs)) / (n_bootstrap + 1).
        - "lr_bootstrap": ndarray of shape (n_bootstrap,)
            Bootstrap likelihood ratio statistics under the null.
        - "null_model": fitted null model instance.
        - "alt_model": fitted alternative model instance.

    Notes
    -----
    - The function assumes the alternative model is a nested extension
      of the null model (for example, K components vs K+1 components).
    - Because mixture models are non-regular, the usual chi-square
      reference distribution is not used. The p-value is computed from
      the empirical null distribution obtained by parametric bootstrap.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    n_samples = X.shape[0]

    # Fit models on observed data
    null_model = null_model_factory()
    null_model.fit(X)

    alt_model = alt_model_factory()
    alt_model.fit(X)

    # Log-likelihoods and observed LR statistic
    # score is average log-likelihood per sample
    ll_null = float(null_model.score(X)) * n_samples
    ll_alt = float(alt_model.score(X)) * n_samples
    lr_obs = 2.0 * (ll_alt - ll_null)

    lr_bootstrap = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        # Sample from fitted null model
        Xb = null_model.sample(n_samples)
        if isinstance(Xb, tuple):
            # For models like sklearn.mixture.GaussianMixture
            Xb = Xb[0]
        Xb = np.asarray(Xb)
        if Xb.ndim == 1:
            Xb = Xb[:, None]

        # Refit null and alt on bootstrap sample
        null_b = null_model_factory()
        alt_b = alt_model_factory()

        null_b.fit(Xb)
        alt_b.fit(Xb)

        ll_null_b = float(null_b.score(Xb)) * n_samples
        ll_alt_b = float(alt_b.score(Xb)) * n_samples
        lr_bootstrap[b] = 2.0 * (ll_alt_b - ll_null_b)

    # Empirical p-value (one-sided, large LR means more evidence)
    p_value = (1.0 + np.sum(lr_bootstrap >= lr_obs)) / (n_bootstrap + 1.0)

    return {
        "statistic": lr_obs,
        "p_value": p_value,
        "lr_bootstrap": lr_bootstrap,
        "lr_obs": lr_obs,
        "null_model": null_model,
        "alt_model": alt_model,
    }
