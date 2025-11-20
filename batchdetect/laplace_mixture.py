import numpy as np


class LaplaceMixture:
    """Mixture of Laplace distributions with isotropic scales.

    This estimator uses expectation-maximization (EM) to fit a mixture of
    independent Laplace components with one scale parameter per component.

    The probability density of component k is

        p(x | k) = (1 / (2 * b_k)) ** d * exp(-||x - mu_k||_1 / b_k),

    where d is the number of features, mu_k is the location vector, and
    b_k is the positive scale parameter.

    Parameters
    ----------
    n_components : int, default=1
        Number of mixture components.
    tol : float, default=1e-4
        Convergence threshold on the EM lower bound.
    max_iter : int, default=100
        Maximum number of EM iterations to perform.
    n_init : int, default=1
        Number of different EM initializations to try. The best run in
        terms of training log likelihood is kept.
    init_params : {"kmeans", "random"}, default="kmeans"
        Method used to initialize the mixture parameters:
        - "kmeans" : use a simple k-means style procedure on the data to
          initialize the means (using L1 distance and medians).
        - "random" : sample initial means from random data points.
    reg_b : float, default=1e-6
        Non-negative regularization added to the scale estimates. This
        prevents numerical issues when a component collapses.
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
        Positive scale parameter of each component.
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
                "X has {} features, but LaplaceMixture was fitted with {} features".format(
                    X.shape[1], self.n_features_in_
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

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_parameters(self, X, rng):
        n_samples, n_features = X.shape
        K = self.n_components

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
                # Assign to nearest mean using L1 distance
                dists = np.sum(
                    np.abs(X[:, None, :] - means[None, :, :]), axis=2
                )
                labels = np.argmin(dists, axis=1)
                # Update means as component-wise medians
                for k in range(K):
                    mask = labels == k
                    if np.any(mask):
                        means[k] = np.median(X[mask], axis=0)
        else:
            raise ValueError("init_params must be 'kmeans' or 'random'.")

        # Initialize scales as average absolute deviation from own mean
        diffs = X[:, None, :] - means[None, :, :]
        l1 = np.abs(diffs).sum(axis=2)
        scales = np.maximum(self.reg_b, np.mean(l1, axis=0) / n_features)

        return weights, means, scales

    # ------------------------------------------------------------------
    # E-step and helpers
    # ------------------------------------------------------------------
    def _estimate_log_prob(self, X):
        # X: (n_samples, n_features)
        diffs = X[:, None, :] - self.means_[None, :, :]
        l1 = np.abs(diffs).sum(axis=2)  # (n_samples, K)
        log_norm = -self.n_features_in_ * (
            np.log(2.0) + np.log(self.scales_)
        )  # (K,)
        return log_norm[None, :] - l1 / self.scales_[None, :]

    def _estimate_log_resp(self, X):
        log_prob = self._estimate_log_prob(X)  # (n_samples, K)
        log_prob_weighted = log_prob + np.log(self.weights_)[None, :]
        log_prob_norm = self._logsumexp(
            log_prob_weighted, axis=1
        )  # (n_samples,)
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

        # Update weights
        weights = nk / n_samples
        eps = np.finfo(float).eps
        weights = np.maximum(weights, eps)
        weights = weights / weights.sum()

        # Update means via weighted medians per dimension
        means = np.empty((K, n_features), dtype=float)
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
        scales = np.empty(K, dtype=float)

        # Precompute a global L1 scale as fallback
        l1_global = np.abs(X - global_median[None, :]).sum(axis=1)
        global_scale = np.maximum(self.reg_b, np.mean(l1_global) / n_features)

        for k in range(K):
            if nk[k] <= eps:
                scales[k] = global_scale
            else:
                scales[k] = np.maximum(
                    self.reg_b,
                    np.sum(resp[:, k] * l1[:, k]) / (n_features * nk[k]),
                )

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
                    "[LaplaceMixture] Iteration {:d}, "
                    "lower bound = {:.6f}, change = {:.6e}".format(
                        n_iter, lower_bound, change
                    )
                )
            if abs(change) < self.tol:
                break

        return lower_bound, n_iter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : LaplaceMixture
            The fitted estimator.
        """
        X = self._check_array(X)
        self.n_features_in_ = X.shape[1]
        rng = self._check_random_state(self.random_state)

        if self.warm_start and hasattr(self, "weights_"):
            n_init = 1
        else:
            n_init = self.n_init

        best_lower_bound = -np.inf
        best_params = None
        best_n_iter = None

        for init in range(n_init):
            if self.verbose:
                print(
                    "[LaplaceMixture] EM init {} of {}".format(init + 1, n_init)
                )
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
        """Compute the average log likelihood of the data under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        avg_log_likelihood : float
            Average log probability of X under the fitted model.
        """
        return np.mean(self.score_samples(X))

    def predict_proba(self, X):
        """Compute posterior probabilities (responsibilities) for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        responsibilities : ndarray of shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            sample in X.
        """
        self._check_is_fitted()
        X = self._check_X(X)
        log_resp, _ = self._estimate_log_resp(X)
        return np.exp(log_resp)

    def predict(self, X):
        """Predict labels for input samples.

        The predicted label for each sample is the component with the
        highest posterior probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Component labels.
        """
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

        # Draw component labels
        counts = rng.multinomial(n_samples, self.weights_)
        labels = np.repeat(np.arange(K), counts)

        # Generate Laplace samples for each component
        X = np.empty((n_samples, n_features), dtype=float)
        start = 0
        for k in range(K):
            nk = counts[k]
            if nk == 0:
                continue
            loc = self.means_[k]
            scale = self.scales_[k]
            X_k = rng.laplace(loc=loc, scale=scale, size=(nk, n_features))
            X[start : start + nk, :] = X_k
            start += nk

        # Shuffle to avoid grouped components
        perm = rng.permutation(n_samples)
        return X[perm], labels[perm]
