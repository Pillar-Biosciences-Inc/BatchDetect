import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_ward_clustering(
    X,
    n_components=3,
    n_clusters=2,
    scale=True,
    return_pca=False,
    return_embedding=False,
):
    """
    Perform PCA on samples and then cluster in PC space using Ward linkage.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix: rows are samples, columns are features.
    n_components : int, optional (default=3)
        Number of principal components to keep (must be <= n_samples - 1).
    n_clusters : int, optional (default=2)
        Desired number of clusters for Ward hierarchical clustering.
    scale : bool, optional (default=True)
        If True, standardize features to mean 0 and variance 1 before PCA.
    return_pca : bool, optional (default=False)
        If True, also return the fitted PCA object.
    return_embedding : bool, optional (default=False)
        If True, also return the low-dimensional PC scores (X_pca).

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels (0, 1, ..., n_clusters - 1).
    pca : PCA object, optional
        Only returned if return_pca=True.
    X_pca : ndarray of shape (n_samples, n_components), optional
        PC scores for each sample, returned if return_embedding=True.
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape

    if n_components > n_samples - 1:
        raise ValueError(
            "n_components cannot exceed n_samples - 1. "
            "Got n_components=%d, n_samples=%d." % (n_components, n_samples)
        )

    # 1. Optional feature scaling
    if scale:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)
    else:
        X_proc = X

    # 2. PCA on samples
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_proc)

    # 3. Ward hierarchical clustering in PC space (Euclidean)
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = ward.fit_predict(X_pca)

    # 4. Flexible return API
    outputs = [labels]
    if return_pca:
        outputs.append(pca)
    if return_embedding:
        outputs.append(X_pca)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)
