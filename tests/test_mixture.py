# test_laplace_mixture.py

import numpy as np

from batchdetect.laplace_mixture import LaplaceMixture


def _make_synthetic_laplace_1d(
    n_samples=4000,
    weights=(0.3, 0.7),
    means=(-2.0, 3.0),
    scales=(0.5, 0.7),
    random_state=42,
):
    rng = np.random.RandomState(random_state)
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    scales = np.asarray(scales, dtype=float)

    # Sample mixture assignments
    z = rng.choice(len(weights), size=n_samples, p=weights)
    X = np.empty((n_samples, 1), dtype=float)
    for k in range(len(weights)):
        mask = z == k
        X[mask, 0] = rng.laplace(
            loc=means[k],
            scale=scales[k],
            size=mask.sum(),
        )
    return X, z, weights, means, scales


def test_fit_recovers_parameters_1d():
    X, z, true_weights, true_means, true_scales = _make_synthetic_laplace_1d()

    model = LaplaceMixture(
        n_components=2,
        n_init=5,
        max_iter=200,
        random_state=0,
        verbose=0,
    )
    model.fit(X)

    # Sort components by mean so that we can compare to ground truth
    order = np.argsort(model.means_.ravel())
    est_weights = model.weights_[order]
    est_means = model.means_[order, 0]
    est_scales = model.scales_[order]

    # Parameter recovery tests with relaxed tolerances
    assert np.allclose(est_weights, true_weights, atol=0.1)
    assert np.allclose(est_means, true_means, atol=0.3)
    assert np.allclose(est_scales, true_scales, atol=0.3)

    # Log likelihood should be finite
    avg_log_like = model.score(X)
    assert np.isfinite(avg_log_like)


def test_predict_proba_and_predict_shapes_and_ranges():
    X, _, _, _, _ = _make_synthetic_laplace_1d(n_samples=500)

    model = LaplaceMixture(
        n_components=3,
        n_init=3,
        max_iter=100,
        random_state=123,
    )
    model.fit(X)

    proba = model.predict_proba(X)
    labels = model.predict(X)

    n_samples, n_components = X.shape[0], model.n_components

    # Shape checks
    assert proba.shape == (n_samples, n_components)
    assert labels.shape == (n_samples,)

    # Rows of predict_proba should sum to 1
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)

    # Labels should be in the allowed range
    assert labels.min() >= 0
    assert labels.max() < n_components

    # predict should be argmax of predict_proba
    assert np.array_equal(labels, np.argmax(proba, axis=1))


def test_score_and_score_samples_are_finite():
    X, _, _, _, _ = _make_synthetic_laplace_1d(n_samples=300)

    model = LaplaceMixture(
        n_components=2,
        n_init=2,
        max_iter=100,
        random_state=7,
    )
    model.fit(X)

    logp = model.score_samples(X)
    assert logp.shape == (X.shape[0],)
    assert np.all(np.isfinite(logp))

    avg_logp = model.score(X)
    assert np.isfinite(avg_logp)


def test_sample_shape_and_labels():
    X, _, _, _, _ = _make_synthetic_laplace_1d(n_samples=500)

    model = LaplaceMixture(
        n_components=2,
        n_init=3,
        max_iter=100,
        random_state=99,
    )
    model.fit(X)

    n_samples = 250
    X_samp, labels = model.sample(n_samples=n_samples, random_state=1234)

    assert X_samp.shape == (n_samples, X.shape[1])
    assert labels.shape == (n_samples,)
    assert labels.min() >= 0
    assert labels.max() < model.n_components


def test_deterministic_fit_with_same_random_state():
    X, _, _, _, _ = _make_synthetic_laplace_1d(n_samples=800)

    model1 = LaplaceMixture(
        n_components=2,
        n_init=3,
        max_iter=100,
        random_state=2025,
    )
    model1.fit(X)

    model2 = LaplaceMixture(
        n_components=2,
        n_init=3,
        max_iter=100,
        random_state=2025,
    )
    model2.fit(X)

    # With same random_state and same data we expect identical parameters
    assert np.allclose(model1.weights_, model2.weights_)
    assert np.allclose(model1.means_, model2.means_)
    assert np.allclose(model1.scales_, model2.scales_)
