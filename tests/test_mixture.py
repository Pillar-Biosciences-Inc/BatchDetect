import numpy as np
import pytest

from batchdetect.mixture import HeavyMixture, parametric_bootstrap_lrt


def _make_synthetic_data(
    n_samples=1000,
    weights=(0.3, 0.7),
    means=(-2.0, 3.0),
    scales=(0.5, 0.7),
    distribution="laplace",
    random_state=42,
    **kwargs,
):
    rng = np.random.RandomState(random_state)
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    scales = np.asarray(scales, dtype=float)

    n_components = len(weights)
    z = rng.choice(n_components, size=n_samples, p=weights)
    X = np.empty((n_samples, 1), dtype=float)

    for k in range(n_components):
        mask = z == k
        nk = mask.sum()
        if nk == 0:
            continue

        loc = means[k]
        scale = scales[k]

        if distribution == "laplace":
            X[mask, 0] = rng.laplace(loc=loc, scale=scale, size=nk)
        elif distribution == "gaussian":
            X[mask, 0] = rng.normal(loc=loc, scale=scale, size=nk)
        elif distribution == "student_t":
            df = kwargs.get("t_df", 7.0)
            # t-distribution scaled
            X[mask, 0] = loc + scale * rng.standard_t(df, size=nk)
        elif distribution == "gennorm":
            beta = kwargs.get("gennorm_beta", 1.5)
            # generalized normal
            from scipy.stats import gennorm

            X[mask, 0] = gennorm.rvs(
                beta, loc=loc, scale=scale, size=nk, random_state=rng
            )
        elif distribution == "hypsecant":
            from scipy.stats import hypsecant

            X[mask, 0] = hypsecant.rvs(
                loc=loc, scale=scale, size=nk, random_state=rng
            )

    return X, z, weights, means, scales


@pytest.mark.parametrize(
    "distribution", ["laplace", "gaussian", "student_t", "gennorm", "hypsecant"]
)
def test_heavy_mixture_fit_predict(distribution):
    X, z, true_weights, true_means, true_scales = _make_synthetic_data(
        distribution=distribution, n_samples=500, random_state=42
    )

    model = HeavyMixture(
        n_components=2,
        component_distribution=distribution,
        n_init=3,
        max_iter=50,
        random_state=42,
    )
    model.fit(X)

    assert hasattr(model, "weights_")
    assert hasattr(model, "means_")
    assert hasattr(model, "scales_")

    # Check shapes
    assert model.weights_.shape == (2,)
    assert model.means_.shape == (2, 1)
    assert model.scales_.shape == (2,)

    # Check prediction shapes
    proba = model.predict_proba(X)
    assert proba.shape == (500, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    labels = model.predict(X)
    assert labels.shape == (500,)

    # Check score
    score = model.score(X)
    assert np.isfinite(score)


def test_heavy_mixture_parameter_recovery_laplace():
    # Test specifically for Laplace as it's the default and we want to ensure good recovery
    X, z, true_weights, true_means, true_scales = _make_synthetic_data(
        distribution="laplace", n_samples=2000, random_state=10
    )

    model = HeavyMixture(
        n_components=2,
        component_distribution="laplace",
        n_init=5,
        max_iter=100,
        random_state=10,
    )
    model.fit(X)

    # Sort by means to compare
    order = np.argsort(model.means_.ravel())
    est_weights = model.weights_[order]
    est_means = model.means_[order, 0]
    est_scales = model.scales_[order]

    # Sort true parameters
    true_order = np.argsort(true_means)
    true_weights = true_weights[true_order]
    true_means = true_means[true_order]
    true_scales = true_scales[true_order]

    # Tolerances
    assert np.allclose(est_weights, true_weights, atol=0.1)
    assert np.allclose(est_means, true_means, atol=0.2)
    assert np.allclose(est_scales, true_scales, atol=0.2)


def test_heavy_mixture_sample():
    model = HeavyMixture(n_components=2, random_state=42)
    # fake fit
    model.weights_ = np.array([0.5, 0.5])
    model.means_ = np.array([[0.0], [5.0]])
    model.scales_ = np.array([1.0, 1.0])
    model.n_features_in_ = 1
    model.component_distribution = "laplace"

    X_sample, y_sample = model.sample(n_samples=100, random_state=42)

    assert X_sample.shape == (100, 1)
    assert y_sample.shape == (100,)
    assert len(np.unique(y_sample)) <= 2


def test_parametric_bootstrap_lrt():
    # Simple test with Gaussian data
    X = np.random.normal(size=(100, 1))

    def null_factory():
        return HeavyMixture(
            n_components=1, component_distribution="gaussian", random_state=42
        )

    def alt_factory():
        return HeavyMixture(
            n_components=2, component_distribution="gaussian", random_state=42
        )

    results = parametric_bootstrap_lrt(
        X,
        null_model_factory=null_factory,
        alt_model_factory=alt_factory,
        n_bootstrap=10,  # small for speed
        random_state=42,
    )

    assert "statistic" in results
    assert "p_value" in results
    assert "lr_bootstrap" in results
    assert len(results["lr_bootstrap"]) == 10
    assert 0 <= results["p_value"] <= 1


def test_invalid_distribution():
    with pytest.raises(ValueError, match="Unsupported component_distribution"):
        model = HeavyMixture(component_distribution="invalid_dist")
        model.fit(np.random.randn(10, 1))


def test_check_is_fitted():
    model = HeavyMixture()
    with pytest.raises(RuntimeError, match="is not fitted yet"):
        model.predict(np.random.randn(10, 1))
