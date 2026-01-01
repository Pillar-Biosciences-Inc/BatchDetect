import inspect

import numpy as np
import pytest
from scipy.stats import kstest

import batchdetect.lmr as lmr_mod
from batchdetect.mixture import HeavyMixture


def _get_lmr_callable():
    """
    Prefer the paper-consistent implementation if present.
    Adjust the name list if your public API differs.
    """
    for name in (
        "lmr_test_heavymixture",  # recommended
        "lmr_test_num_components",  # acceptable fallback
        "lmr_test",  # generic fallback
    ):
        fn = getattr(lmr_mod, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(
        "No LMR test function found in batchdetect.lmr. "
        "Expected one of: lmr_test_heavymixture, lmr_test_num_components, lmr_test."
    )


def _call_lmr(fn, X, L, K, dist, *, correction=True, fit_kwargs=None):
    """
    Call LMR function with a flexible signature (only pass kwargs it accepts).
    """
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)

    sig = inspect.signature(fn)
    kwargs = {}

    if "component_distribution" in sig.parameters:
        kwargs["component_distribution"] = dist
    if "fit_kwargs" in sig.parameters:
        kwargs["fit_kwargs"] = fit_kwargs
    if "correction" in sig.parameters:
        kwargs["correction"] = correction

    # Speed/stability knobs if supported by your implementation
    if "fd_eps" in sig.parameters:
        kwargs["fd_eps"] = 1e-4
    if "ridge_A" in sig.parameters:
        kwargs["ridge_A"] = 1e-8
    if "imhof_eps" in sig.parameters:
        kwargs["imhof_eps"] = 1e-6

    return fn(X, L, K, **kwargs)


# -----------------------------
# Data generation using HeavyMixture.sample (exact model match)
# -----------------------------
def _oracle_mixture(
    dist: str,
    weights,
    means,
    scales,
    *,
    t_df=7.0,
    gennorm_beta=1.5,
    random_state=0,
):
    m = HeavyMixture(
        n_components=len(weights),
        component_distribution=dist,
        t_df=t_df,
        gennorm_beta=gennorm_beta,
        random_state=random_state,
    )
    # Mark as "fitted" by setting required attributes
    m.weights_ = np.asarray(weights, dtype=float)
    m.weights_ = m.weights_ / m.weights_.sum()
    m.means_ = np.asarray(means, dtype=float)
    m.scales_ = np.asarray(scales, dtype=float)
    m.n_features_in_ = int(m.means_.shape[1])
    return m


def _sample_null(rng, n, d, dist, *, loc=0.0, scale=1.0):
    oracle = _oracle_mixture(
        dist=dist,
        weights=[1.0],
        means=np.full((1, d), float(loc)),
        scales=[float(scale)],
        random_state=rng.integers(0, 2**31 - 1),
    )
    X, _ = oracle.sample(n, random_state=rng.integers(0, 2**31 - 1))
    return X


def _sample_alt_two_comp(
    rng, n, d, dist, *, w0=0.5, m0=-3.0, m1=3.0, s0=1.0, s1=1.0
):
    oracle = _oracle_mixture(
        dist=dist,
        weights=[float(w0), 1.0 - float(w0)],
        means=np.vstack(
            [np.full((1, d), float(m0)), np.full((1, d), float(m1))]
        ),
        scales=[float(s0), float(s1)],
        random_state=rng.integers(0, 2**31 - 1),
    )
    X, _ = oracle.sample(n, random_state=rng.integers(0, 2**31 - 1))
    return X


# -----------------------------
# Shared fit kwargs
# -----------------------------
@pytest.fixture(scope="session")
def fit_kwargs():
    # These are exactly the HeavyMixture __init__ options in mixture.py
    return dict(
        n_init=3,
        max_iter=150,
        tol=1e-4,
        init_params="kmeans",
        reg_b=1e-6,
        random_state=0,
        verbose=0,
    )


# -----------------------------
# 12 Tests
# -----------------------------
def test_api_returns_lmrresult_fields(fit_kwargs):
    fn = _get_lmr_callable()
    rng = np.random.default_rng(1)
    X = _sample_null(rng, n=400, d=1, dist="laplace")

    res = _call_lmr(
        fn, X, L=1, K=2, dist="laplace", correction=True, fit_kwargs=fit_kwargs
    )

    assert hasattr(res, "lr")
    assert hasattr(res, "lmr_lr")
    assert hasattr(res, "df")
    assert hasattr(res, "p_value")
    assert np.isfinite(res.lr)
    assert np.isfinite(res.lmr_lr)
    assert isinstance(res.df, int)
    assert np.isfinite(res.p_value)


def test_rejects_invalid_component_order(fit_kwargs):
    fn = _get_lmr_callable()
    rng = np.random.default_rng(2)
    X = _sample_null(rng, n=200, d=1, dist="gaussian")

    with pytest.raises(Exception):
        _call_lmr(fn, X, L=2, K=2, dist="gaussian", fit_kwargs=fit_kwargs)
    with pytest.raises(Exception):
        _call_lmr(fn, X, L=2, K=1, dist="gaussian", fit_kwargs=fit_kwargs)


def test_p_value_in_unit_interval(fit_kwargs):
    fn = _get_lmr_callable()
    rng = np.random.default_rng(3)
    X = _sample_null(rng, n=300, d=1, dist="laplace")

    res = _call_lmr(fn, X, L=1, K=2, dist="laplace", fit_kwargs=fit_kwargs)
    assert 0.0 <= res.p_value <= 1.0


def test_deterministic_with_fixed_seed_and_data(fit_kwargs):
    """
    With deterministic initialization (random_state fixed) and fixed X, repeated calls
    should be identical or extremely close.
    """
    fn = _get_lmr_callable()
    rng = np.random.default_rng(4)
    X = _sample_alt_two_comp(rng, n=400, d=1, dist="gaussian", m0=-3.0, m1=3.0)

    res1 = _call_lmr(fn, X, L=1, K=2, dist="gaussian", fit_kwargs=fit_kwargs)
    res2 = _call_lmr(fn, X, L=1, K=2, dist="gaussian", fit_kwargs=fit_kwargs)

    assert abs(res1.p_value - res2.p_value) < 1e-10
    assert abs(res1.lmr_lr - res2.lmr_lr) < 1e-8


def test_correction_reduces_statistic(fit_kwargs):
    fn = _get_lmr_callable()
    rng = np.random.default_rng(5)
    X = _sample_alt_two_comp(rng, n=500, d=1, dist="gaussian", m0=-2.0, m1=2.0)

    res_corr = _call_lmr(
        fn, X, L=1, K=2, dist="gaussian", correction=True, fit_kwargs=fit_kwargs
    )
    res_raw = _call_lmr(
        fn,
        X,
        L=1,
        K=2,
        dist="gaussian",
        correction=False,
        fit_kwargs=fit_kwargs,
    )

    assert res_corr.lmr_lr <= res_raw.lmr_lr + 1e-12
    # If correction=False, lmr_lr should typically equal lr (depends on your API contract)
    assert abs(res_raw.lmr_lr - res_raw.lr) < 1e-8


def test_permutation_invariance(fit_kwargs):
    fn = _get_lmr_callable()
    rng = np.random.default_rng(6)
    X = _sample_alt_two_comp(rng, n=450, d=1, dist="laplace", m0=-3.0, m1=3.0)
    perm = rng.permutation(X.shape[0])

    res1 = _call_lmr(fn, X, L=1, K=2, dist="laplace", fit_kwargs=fit_kwargs)
    res2 = _call_lmr(
        fn, X[perm], L=1, K=2, dist="laplace", fit_kwargs=fit_kwargs
    )

    # Allow small numeric variation due to EM and finite differences
    assert abs(res1.p_value - res2.p_value) < 5e-3


def test_null_pvalues_approximately_uniform_laplace_large_n(fit_kwargs):
    fn = _get_lmr_callable()
    rng = np.random.default_rng(10)

    n_rep = 10
    n = 1200
    pvals = []
    for _ in range(n_rep):
        X = _sample_null(rng, n=n, d=1, dist="laplace")
        res = _call_lmr(fn, X, L=1, K=2, dist="laplace", fit_kwargs=fit_kwargs)
        pvals.append(res.p_value)

    pvals = np.asarray(pvals)
    assert np.all(np.isfinite(pvals))
    assert 0.30 <= float(pvals.mean()) <= 0.80

    D, p_ks = kstest(pvals, "uniform")
    assert D < 0.70
    assert p_ks > 1e-4


def test_null_pvalues_approximately_uniform_gaussian_large_n(fit_kwargs):
    fn = _get_lmr_callable()
    rng = np.random.default_rng(11)

    n_rep = 10
    n = 1200
    pvals = []
    for _ in range(n_rep):
        X = _sample_null(rng, n=n, d=1, dist="gaussian")
        res = _call_lmr(fn, X, L=1, K=2, dist="gaussian", fit_kwargs=fit_kwargs)
        pvals.append(res.p_value)

    pvals = np.asarray(pvals)
    assert np.all(np.isfinite(pvals))
    assert 0.30 <= float(pvals.mean()) <= 0.70

    D, p_ks = kstest(pvals, "uniform")
    assert D < 0.30
    assert p_ks > 1e-4


def test_alternative_gives_small_pvalues_laplace(fit_kwargs):
    fn = _get_lmr_callable()
    rng = np.random.default_rng(20)

    n_rep = 8
    n = 500
    pvals = []
    for _ in range(n_rep):
        X = _sample_alt_two_comp(
            rng, n=n, d=1, dist="laplace", m0=-4.0, m1=4.0, s0=1.0, s1=1.0
        )
        res = _call_lmr(fn, X, L=1, K=2, dist="laplace", fit_kwargs=fit_kwargs)
        pvals.append(res.p_value)

    pvals = np.asarray(pvals)
    assert np.median(pvals) < 0.01
    assert float(np.mean(pvals < 0.05)) >= 0.75


def test_alternative_gives_small_pvalues_gaussian(fit_kwargs):
    fn = _get_lmr_callable()
    rng = np.random.default_rng(21)

    n_rep = 8
    n = 500
    pvals = []
    for _ in range(n_rep):
        X = _sample_alt_two_comp(
            rng, n=n, d=1, dist="gaussian", m0=-3.5, m1=3.5, s0=1.0, s1=1.0
        )
        res = _call_lmr(fn, X, L=1, K=2, dist="gaussian", fit_kwargs=fit_kwargs)
        pvals.append(res.p_value)

    pvals = np.asarray(pvals)
    assert np.median(pvals) < 0.01
    assert float(np.mean(pvals < 0.05)) >= 0.75


def test_power_increases_with_separation(fit_kwargs):
    fn = _get_lmr_callable()
    rng = np.random.default_rng(30)

    n = 500
    X_small = _sample_alt_two_comp(
        rng, n=n, d=1, dist="gaussian", m0=-1.5, m1=1.5, s0=1.0, s1=1.0
    )
    X_large = _sample_alt_two_comp(
        rng, n=n, d=1, dist="gaussian", m0=-4.0, m1=4.0, s0=1.0, s1=1.0
    )

    res_small = _call_lmr(
        fn, X_small, L=1, K=2, dist="gaussian", fit_kwargs=fit_kwargs
    )
    res_large = _call_lmr(
        fn, X_large, L=1, K=2, dist="gaussian", fit_kwargs=fit_kwargs
    )

    assert res_large.p_value <= res_small.p_value + 1e-6


def test_power_increases_with_sample_size(fit_kwargs):
    fn = _get_lmr_callable()
    rng = np.random.default_rng(31)

    X_n200 = _sample_alt_two_comp(
        rng, n=200, d=1, dist="laplace", m0=-3.0, m1=3.0, s0=1.0, s1=1.0
    )
    X_n900 = _sample_alt_two_comp(
        rng, n=900, d=1, dist="laplace", m0=-3.0, m1=3.0, s0=1.0, s1=1.0
    )

    res_n200 = _call_lmr(
        fn, X_n200, L=1, K=2, dist="laplace", fit_kwargs=fit_kwargs
    )
    res_n900 = _call_lmr(
        fn, X_n900, L=1, K=2, dist="laplace", fit_kwargs=fit_kwargs
    )

    assert res_n900.p_value <= res_n200.p_value + 1e-6
