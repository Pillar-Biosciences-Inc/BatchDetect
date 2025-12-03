import numpy as np
import pytest
from scipy.stats import chi2

from batchdetect.lmr import (
    davies_pvalue_weighted_chisq,
    lo_mendell_rubin_lrt,
    weighted_chisq_lrt_num_components,
)


def test_lo_mendell_rubin_lrt_calculation():
    # Manual check
    n = 100
    ll_null = -100.0
    ll_alt = -90.0
    n_params_null = 5
    n_params_alt = 10
    k_null = 1
    k_alt = 2

    res = lo_mendell_rubin_lrt(
        n=n,
        ll_null=ll_null,
        ll_alt=ll_alt,
        n_params_null=n_params_null,
        n_params_alt=n_params_alt,
        k_null=k_null,
        k_alt=k_alt,
    )

    lr = 2.0 * (ll_alt - ll_null)  # 20.0
    delta_k = k_alt - k_null  # 1
    denom = 1.0 + 3.0 * delta_k / np.log(n)
    expected_lmr = lr / denom

    assert np.isclose(res.lr, lr)
    assert np.isclose(res.lmr_lr, expected_lmr)
    assert res.df == 5
    assert 0 <= res.p_value <= 1.0


def test_lo_mendell_rubin_lrt_errors():
    with pytest.raises(
        ValueError, match="Alternative model must have more classes"
    ):
        lo_mendell_rubin_lrt(100, -10, -5, 2, 4, 2, 2)

    with pytest.raises(
        ValueError, match="Alternative model must have more parameters"
    ):
        lo_mendell_rubin_lrt(100, -10, -5, 10, 5, 1, 2)


def test_davies_pvalue_weighted_chisq_simple():
    # If lambdas=[1.0], then Q ~ Chi2(1).
    # P(Q >= x) should match chi2.sf(x, 1)
    lambdas = [1.0]
    x = 3.84  # approx 95th percentile

    p_val = davies_pvalue_weighted_chisq(lambdas, x)
    expected = chi2.sf(x, df=1)

    assert np.isclose(p_val, expected, atol=1e-4)


def test_davies_pvalue_weighted_chisq_errors():
    with pytest.raises(ValueError, match="lambdas must be non-empty"):
        davies_pvalue_weighted_chisq([], 1.0)

    with pytest.raises(ValueError, match="x must be finite"):
        davies_pvalue_weighted_chisq([1.0], np.inf)


def test_weighted_chisq_lrt_num_components_integration():
    # Generate simple 1D Gaussian data
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 1))

    # Test 1 vs 2 components
    # We use "gaussian" distribution for speed/simplicity
    res = weighted_chisq_lrt_num_components(
        X,
        L=1,
        K=2,
        component_distribution="gaussian",
        pvalue_method="davies",
        random_state=42,
        fit_kwargs={"max_iter": 10, "n_init": 1},  # speed up
    )

    assert "lr" in res
    assert "p_value" in res
    assert "lambdas" in res
    assert res["pvalue_method"] == "davies"
    assert 0 <= res["p_value"] <= 1.0

    # Test MC method
    res_mc = weighted_chisq_lrt_num_components(
        X,
        L=1,
        K=2,
        component_distribution="gaussian",
        pvalue_method="mc",
        n_sim=100,  # small for speed
        random_state=42,
        fit_kwargs={"max_iter": 10, "n_init": 1},
    )
    assert res_mc["pvalue_method"] == "mc"
    assert 0 <= res_mc["p_value"] <= 1.0


def test_weighted_chisq_lrt_invalid_L_K():
    X = np.zeros((10, 1))
    with pytest.raises(ValueError, match="Require 1 <= L < K"):
        weighted_chisq_lrt_num_components(X, L=2, K=1)
