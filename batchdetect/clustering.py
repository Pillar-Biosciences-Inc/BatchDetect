import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _check_corr_matrix(corr):
    """
    Basic sanity checks for a correlation matrix.
    """
    corr = np.asarray(corr)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be a square 2D array.")
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("corr must be symmetric.")
    np.fill_diagonal(corr, 1.0)
    return corr


# 1. Hierarchical clustering with correlation distance
def cluster_hierarchical_corr(corr, n_clusters, linkage_method="average"):
    """
    Cluster observations using hierarchical clustering on correlation distance.

    Parameters
    ----------
    corr : array_like, shape (n_samples, n_samples)
        Correlation matrix between observations.
    n_clusters : int
        Desired number of clusters.
    linkage_method : str
        Linkage method for scipy.cluster.hierarchy.linkage
        ("single", "complete", "average", "ward", etc.).

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Cluster labels in {1, 2, ..., n_clusters}.
    """
    corr = _check_corr_matrix(corr)
    dist = 1.0 - corr
    # Convert to condensed form for linkage
    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    labels = labels - 1
    return labels


# 2. Spectral clustering on correlation-based similarity
def cluster_spectral_corr(
    corr,
    n_clusters,
    gamma=1.0,
    zero_negative=True,
    random_state=0,
):
    """
    Spectral clustering using a similarity derived from the correlation matrix.

    Similarity S is constructed as:
        S_ij = exp(gamma * r_ij)
    optionally with negative correlations zeroed out before exponentiation.

    Parameters
    ----------
    corr : array_like, shape (n_samples, n_samples)
        Correlation matrix between observations.
    n_clusters : int
        Number of clusters.
    gamma : float
        Scale factor for correlations inside the exponential.
    zero_negative : bool
        If True, clip negative correlations to zero before applying exp.
    random_state : int
        Random state for SpectralClustering.

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Cluster labels in {0, 1, ..., n_clusters-1}.
    """
    corr = _check_corr_matrix(corr)

    if zero_negative:
        base = np.clip(corr, 0.0, None)
    else:
        base = corr

    S = np.exp(gamma * base)

    S = np.exp(gamma * base)

    from sklearn.cluster import SpectralClustering

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
    )
    labels = sc.fit_predict(S)
    return labels


# 3. Leiden community detection on correlation graph
def cluster_leiden_corr(
    corr,
    resolution=1.0,
    n_iterations=-1,
    min_weight=1e-6,
):
    """
    Leiden community detection on a graph built from the correlation matrix.

    We construct a weighted undirected graph with edges:
        w_ij = max(corr_ij, 0)
    and optionally drop near-zero weights.

    Requires:
        pip install igraph leidenalg

    Parameters
    ----------
    corr : array_like, shape (n_samples, n_samples)
        Correlation matrix between observations.
    resolution : float
        Resolution parameter for Leiden (higher -> more/smaller communities).
    n_iterations : int
        Number of iterations for Leiden. -1 means until convergence.
    min_weight : float
        Minimum edge weight; edges with w < min_weight are dropped.

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Community labels in {0, 1, ..., n_communities-1}.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        raise ImportError(
            "cluster_leiden_corr requires igraph and leidenalg. "
            "Install with: pip install igraph leidenalg"
        ) from e

    corr = _check_corr_matrix(corr)
    W = np.clip(corr, 0.0, None)
    n = W.shape[0]

    # Build adjacency list for igraph
    edges = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            w = W[i, j]
            if w >= min_weight:
                edges.append((i, j))
                weights.append(float(w))

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights

    partition_type = leidenalg.RBConfigurationVertexPartition
    partition = leidenalg.find_partition(
        g,
        partition_type,
        weights=g.es["weight"],
        resolution_parameter=resolution,
        n_iterations=n_iterations,
    )

    labels = np.array(partition.membership, dtype=int)
    return labels


def cluster_pca_kmeans_corr(
    corr,
    n_components,
    n_clusters,
    random_state=0,
    n_init=10,
    max_iter=300,
):
    """
    Cluster observations by applying PCA to the correlation matrix
    and running k-means on the top principal component scores.

    For an observation-by-observation correlation matrix C, its eigenvectors
    give a low-dimensional embedding of observations. We keep the top
    n_components eigenvectors and cluster their rows.

    Parameters
    ----------
    corr : array_like, shape (n_samples, n_samples)
        Correlation matrix between observations.
    n_components : int
        Number of principal components to use.
    n_clusters : int
        Number of clusters for k-means.
    random_state : int
        Random state for k-means.
    n_init : int
        Number of k-means restarts.
    max_iter : int
        Maximum iterations for k-means.

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Cluster labels in {0, 1, ..., n_clusters-1}.
    """
    corr = _check_corr_matrix(corr)
    n = corr.shape[0]

    if n_components > n:
        raise ValueError("n_components cannot exceed number of observations.")

    # Eigen-decomposition of symmetric matrix
    vals, vecs = np.linalg.eigh(corr)
    # Sort by descending eigenvalue
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    components = vecs[:, :n_components]

    components = vecs[:, :n_components]

    from sklearn.cluster import KMeans

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels = km.fit_predict(components)
    return labels
