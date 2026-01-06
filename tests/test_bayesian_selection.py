import numpy as np
import pytest
from scipy.special import logsumexp

from batchdetect.bayesian_selection import (
    _logmeanexp,
    _effective_sample_size_from_logw,
    _log_complete_likelihood_isotropic_gaussian,
    _sample_prior_state,
    _gibbs_sweep_tempered,
    ais_log_evidence_isotropic_gmm,
    ais_select_K,
)


def test_logmeanexp():
    a = np.array([0.0, 0.0])
    # log(mean(exp(0), exp(0))) = log(1) = 0
    assert np.isclose(_logmeanexp(a), 0.0)

    a = np.array([10.0, 10.0])
    # log(mean(exp(10), exp(10))) = 10
    assert np.isclose(_logmeanexp(a), 10.0)

    # Compare with scipy logsumexp - log(N)
    rng = np.random.default_rng(42)
    x = rng.normal(size=10)
    expected = logsumexp(x) - np.log(len(x))
    assert np.isclose(_logmeanexp(x), expected)


def test_effective_sample_size_from_logw():
    # Equal weights -> ESS = N
    logw = np.zeros(10)
    ess = _effective_sample_size_from_logw(logw)
    assert np.isclose(ess, 10.0)

    # One weight dominates -> ESS = 1
    logw = np.array([0.0, -100.0, -100.0])
    ess = _effective_sample_size_from_logw(logw)
    assert np.isclose(ess, 1.0, atol=1e-5)

    # Empty
    assert _effective_sample_size_from_logw([]) == 0.0


def test_log_complete_likelihood_isotropic_gaussian():
    # 2 points, 2 components
    X = np.array([[0.0], [10.0]])
    z = np.array([0, 1])
    mus = np.array([[0.0], [10.0]])
    taus = np.array([1.0, 1.0])  # sigma=1

    # LL = log N(0|0,1) + log N(10|10,1)
    # log N(x|mu,1) = -0.5 * log(2pi) - 0.5 * (x-mu)^2
    # here (x-mu)^2 = 0 for both
    # LL = 2 * (-0.5 * log(2pi))

    ll = _log_complete_likelihood_isotropic_gaussian(X, z, mus, taus)
    expected = 2 * (-0.5 * np.log(2 * np.pi))
    assert np.isclose(ll, expected)


def test_sample_prior_state():
    rng = np.random.default_rng(42)
    N, D = 50, 2
    X = np.zeros((N, D))
    K = 3
    alpha0 = 1.0
    m0 = np.zeros(D)
    kappa0 = 0.01
    a0 = 2.0
    b0 = 2.0

    pi, mus, taus, z = _sample_prior_state(
        rng, X, K, alpha0, m0, kappa0, a0, b0
    )

    assert pi.shape == (K,)
    assert np.isclose(pi.sum(), 1.0)
    assert mus.shape == (K, D)
    assert taus.shape == (K,)
    assert np.all(taus > 0)
    assert z.shape == (N,)
    assert np.all((z >= 0) & (z < K))


def test_gibbs_sweep_tempered():
    rng = np.random.default_rng(42)
    N, D = 10, 2
    X = rng.normal(size=(N, D))
    K = 2

    # Initial state
    pi = np.array([0.5, 0.5])
    mus = np.zeros((K, D))
    taus = np.ones(K)
    z = rng.choice(K, size=N)

    beta = 0.5
    alpha0 = 1.0
    m0 = np.zeros(D)
    kappa0 = 0.01
    a0 = 1.0
    b0 = 1.0

    pi_new, mus_new, taus_new, z_new = _gibbs_sweep_tempered(
        rng, X, pi, mus, taus, z, beta, alpha0, m0, kappa0, a0, b0
    )

    assert pi_new.shape == (K,)
    assert np.isclose(pi_new.sum(), 1.0)
    assert mus_new.shape == (K, D)
    assert taus_new.shape == (K,)
    assert np.all(taus_new > 0)
    assert z_new.shape == (N,)


def test_ais_log_evidence_isotropic_gmm():
    # Simple case: clearly 2 clusters
    rng = np.random.default_rng(42)
    X = np.concatenate(
        [
            rng.normal(loc=-5, scale=1, size=(20, 1)),
            rng.normal(loc=5, scale=1, size=(20, 1)),
        ]
    )

    res = ais_log_evidence_isotropic_gmm(
        X,
        K=2,
        n_particles=10,
        n_intermediate=10,
        n_gibbs_sweeps_per_beta=1,
        random_state=42,
    )

    assert "logZ_hat" in res
    assert "ess" in res
    assert "betas" in res
    assert res["betas"][0] == 0.0
    assert res["betas"][-1] == 1.0
    assert np.isfinite(res["logZ_hat"])


def test_ais_select_K():
    rng = np.random.default_rng(42)
    # Data from 2 clusters
    X = np.concatenate(
        [
            rng.normal(loc=-5, scale=1, size=(50, 1)),
            rng.normal(loc=5, scale=1, size=(50, 1)),
        ]
    )

    # Test K=1 vs K=2 vs K=3
    # Use few particles/steps for speed in test
    results = ais_select_K(
        X, K_list=[1, 2, 3], n_particles=20, n_intermediate=20, random_state=42
    )

    assert len(results) == 3
    # Results are sorted by logZ desc
    # top result should ideally be K=2 given the data,
    # but strictly checking sort order is enough for API test.
    assert results[0][1] >= results[1][1]

    # Check structure
    best_K, best_score, best_ess, info = results[0]
    assert isinstance(best_K, int) or isinstance(best_K, np.integer)
    assert isinstance(best_score, float)
    assert isinstance(best_ess, float)
    assert isinstance(info, dict)

def test_ais_select_K_recovery():
    # Test that AIS correctly identifies K=1 vs K=2 with more data
    rng = np.random.default_rng(42)
    
    # Case 1: K=1
    X1 = rng.normal(loc=0, scale=1, size=(500, 1))
    
    # Run with small n_particles/intermediate for speed, but enough for separation
    # 500 points is enough evidence.
    res1 = ais_select_K(
        X1, K_list=[1, 2],
        n_particles=20, 
        n_intermediate=20,
        random_state=42
    )
    # accurately should prefer K=1
    assert res1[0][0] == 1
    
    # Case 2: K=2 (well separated)
    X2 = np.concatenate([
        rng.normal(loc=-3, scale=1, size=(250, 1)),
        rng.normal(loc=3, scale=1, size=(250, 1))
    ])
    
    res2 = ais_select_K(
        X2, K_list=[1, 2],
        n_particles=20,
        n_intermediate=20,
        random_state=42
    )
    # accurately should prefer K=2
    assert res2[0][0] == 2

