import numpy as np
from scipy.stats import pearsonr, spearmanr


def normalize_mat(X):
    """
    Normalize the input matrix X by dividing each sample by the mean of (sample + 1).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data matrix.

    Returns
    -------
    X_norm : ndarray, shape (n_samples, n_features)
        Normalized data matrix.
    """
    n_samples, n_amplicons = X.shape
    X_norm = np.zeros(X.shape)
    for i in range(n_samples):
        a = X[i] / np.mean(X[i] + 1)
        X_norm[i] = a
    return X_norm


def get_correlations(X_norm):
    """
    Compute Pearson and Spearman correlation matrices between samples.

    Parameters
    ----------
    X_norm : array-like, shape (n_samples, n_features)
        Normalized input data.

    Returns
    -------
    corrs_pearson : ndarray, shape (n_samples, n_samples)
        Pearson correlation matrix with diagonal elements set to 0.
    corrs_spearman : ndarray, shape (n_samples, n_samples)
        Spearman correlation matrix with diagonal elements set to 0.
    """
    n_samples, n_amplicons = X_norm.shape
    corrs_spearman = np.zeros((n_samples, n_samples))
    corrs_pearson = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            stat, pval = pearsonr(X_norm[i], X_norm[j])
            corrs_pearson[i, j] = stat
            stat, pval = spearmanr(X_norm[i], X_norm[j])
            corrs_spearman[i, j] = stat
    corrs_spearman = corrs_spearman - np.diag(np.diag(corrs_spearman))
    corrs_pearson = corrs_pearson - np.diag(np.diag(corrs_pearson))
    return corrs_pearson, corrs_spearman
