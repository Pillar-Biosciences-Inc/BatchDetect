import numpy as np
import pytest

from batchdetect.clustering import (
    _check_corr_matrix,
    cluster_hierarchical_corr,
    cluster_leiden_corr,
    cluster_pca_kmeans_corr,
    cluster_spectral_corr,
)


def test_check_corr_matrix_valid():
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    checked = _check_corr_matrix(corr)
    assert np.allclose(checked, corr)
    assert checked.shape == (2, 2)


def test_check_corr_matrix_invalid_shape():
    with pytest.raises(ValueError, match="must be a square 2D array"):
        _check_corr_matrix(np.array([1.0, 0.5]))

    with pytest.raises(ValueError, match="must be a square 2D array"):
        _check_corr_matrix(np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0]]))


def test_check_corr_matrix_asymmetric():
    corr = np.array([[1.0, 0.5], [0.4, 1.0]])
    with pytest.raises(ValueError, match="must be symmetric"):
        _check_corr_matrix(corr)


def test_cluster_hierarchical_corr():
    # 5 samples: {0,1,2} correlated, {3,4} correlated
    corr = np.array(
        [
            [1.0, 0.9, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.9, 0.1, 0.1],
            [0.9, 0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.1, 0.9, 1.0],
        ]
    )

    labels = cluster_hierarchical_corr(corr, n_clusters=2)
    assert len(labels) == 5

    # {0,1,2} should be in one cluster
    assert labels[0] == labels[1]
    assert labels[0] == labels[2]

    # {3,4} should be in another
    assert labels[3] == labels[4]

    # The two groups should be distinct
    assert labels[0] != labels[3]

    # Check range
    assert np.all(labels >= 0)
    assert len(np.unique(labels)) == 2


def test_cluster_spectral_corr():
    try:
        import sklearn.cluster
    except ImportError:
        pytest.skip("sklearn not installed")
    except ValueError:
        pytest.skip("sklearn binary incompatibility")

    corr = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # We might need to mock sklearn if the environment is broken,
    # but let's try running it first.
    try:
        labels = cluster_spectral_corr(corr, n_clusters=2, random_state=42)
    except ValueError as e:
        if "numpy.dtype size changed" in str(e):
            pytest.skip("sklearn binary incompatibility")
        raise e

    assert len(labels) == 3
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_cluster_leiden_corr():
    try:
        import igraph
        import leidenalg
    except ImportError:
        pytest.skip("igraph or leidenalg not installed")

    corr = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])

    labels = cluster_leiden_corr(corr, resolution=1.0)
    assert len(labels) == 3
    # Ideally 0 and 1 are together
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_cluster_pca_kmeans_corr():
    try:
        import sklearn.cluster
    except ImportError:
        pytest.skip("sklearn not installed")
    except ValueError:
        pytest.skip("sklearn binary incompatibility")

    corr = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])

    try:
        labels = cluster_pca_kmeans_corr(
            corr, n_components=2, n_clusters=2, random_state=42
        )
    except ValueError as e:
        if "numpy.dtype size changed" in str(e):
            pytest.skip("sklearn binary incompatibility")
        raise e

    assert len(labels) == 3
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_cluster_pca_kmeans_corr_invalid_components():
    corr = np.eye(3)
    with pytest.raises(ValueError, match="n_components cannot exceed"):
        cluster_pca_kmeans_corr(corr, n_components=4, n_clusters=2)
