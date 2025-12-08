import pytest
import numpy as np
from sklearn.decomposition import PCA
from batchdetect.ward import pca_ward_clustering

def test_pca_ward_clustering_valid_inputs():
    """Test pca_ward_clustering with valid synthetic data."""
    # Create synthetic data: 2 clusters in 5D space
    np.random.seed(42)
    X1 = np.random.normal(loc=0, scale=0.1, size=(20, 5))
    X2 = np.random.normal(loc=5, scale=0.1, size=(20, 5))
    X = np.vstack([X1, X2])
    
    labels = pca_ward_clustering(X, n_components=2, n_clusters=2)
    
    assert len(labels) == 40
    assert len(np.unique(labels)) == 2
    # Check if it separated the two clear clusters
    # Since labels are arbitrary (0,1 or 1,0), we check if first 20 are same and last 20 are same
    assert np.all(labels[:20] == labels[0])
    assert np.all(labels[20:] == labels[20])
    assert labels[0] != labels[20]

def test_pca_ward_clustering_return_options():
    """Test return_pca and return_embedding options."""
    X = np.random.rand(20, 5)
    
    # Case 1: return_pca=True
    res = pca_ward_clustering(X, n_components=2, return_pca=True)
    assert isinstance(res, tuple)
    assert len(res) == 2
    labels, pca = res
    assert isinstance(pca, PCA)
    
    # Case 2: return_embedding=True
    res = pca_ward_clustering(X, n_components=2, return_embedding=True)
    assert isinstance(res, tuple)
    assert len(res) == 2
    labels, X_pca = res
    assert X_pca.shape == (20, 2)
    
    # Case 3: Both True
    res = pca_ward_clustering(X, n_components=2, return_pca=True, return_embedding=True)
    assert isinstance(res, tuple)
    assert len(res) == 3
    labels, pca, X_pca = res
    assert isinstance(pca, PCA)
    assert X_pca.shape == (20, 2)

def test_pca_ward_clustering_no_scale():
    """Test with scale=False."""
    X = np.random.rand(20, 5)
    labels = pca_ward_clustering(X, n_components=2, scale=False)
    assert len(labels) == 20

def test_pca_ward_clustering_invalid_n_components():
    """Test error when n_components is too large."""
    X = np.random.rand(10, 5)
    # n_samples = 10. Max n_components = 9.
    with pytest.raises(ValueError, match="n_components cannot exceed n_samples - 1"):
        pca_ward_clustering(X, n_components=10)

def test_pca_ward_clustering_output_shapes():
    """Verify shapes of returned labels and embedding."""
    n_samples = 15
    n_features = 4
    n_components = 3
    X = np.random.rand(n_samples, n_features)
    
    labels, X_pca = pca_ward_clustering(
        X, n_components=n_components, return_embedding=True
    )
    
    assert labels.shape == (n_samples,)
    assert X_pca.shape == (n_samples, n_components)
