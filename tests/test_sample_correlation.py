import numpy as np

from batchdetect.sample_correlation import get_correlations, normalize_mat


def test_normalize_mat():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Row 0 mean(X[0]+1) = mean(2, 3) = 2.5. Row 0 -> [1/2.5, 2/2.5] = [0.4, 0.8]
    # Row 1 mean(X[1]+1) = mean(4, 5) = 4.5. Row 1 -> [3/4.5, 4/4.5] = [0.666..., 0.888...]

    X_norm = normalize_mat(X)

    assert X_norm.shape == X.shape
    expected_row0 = np.array([0.4, 0.8])
    assert np.allclose(X_norm[0], expected_row0)

    expected_row1 = np.array([3.0 / 4.5, 4.0 / 4.5])
    assert np.allclose(X_norm[1], expected_row1)


def test_get_correlations():
    # Create 3 samples. 0 and 1 perfectly correlated, 2 uncorrelated/negative
    # We use normalized data logic or just pass simple data since get_correlations
    # just computes correlations on rows.
    X_norm = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],  # 2x sample 0, so correlation 1
            [3.0, 2.0, 1.0],  # Reverse of sample 0, so correlation -1
        ]
    )

    pearson, spearman = get_correlations(X_norm)

    assert pearson.shape == (3, 3)
    assert spearman.shape == (3, 3)

    # Diagonals should be 0 (as per implementation)
    assert np.all(np.diag(pearson) == 0)
    assert np.all(np.diag(spearman) == 0)

    # Check values
    # 0 vs 1: Pearson 1.0
    assert np.isclose(pearson[0, 1], 1.0)
    assert np.isclose(pearson[1, 0], 1.0)

    # 0 vs 2: Pearson -1.0
    assert np.isclose(pearson[0, 2], -1.0)
    assert np.isclose(pearson[2, 0], -1.0)

    # Spearman should also be 1 and -1 for these monotonic relationships
    assert np.isclose(spearman[0, 1], 1.0)
    assert np.isclose(spearman[0, 2], -1.0)


def test_get_correlations_symmetry():
    rng = np.random.RandomState(42)
    X_norm = rng.randn(5, 10)

    pearson, spearman = get_correlations(X_norm)

    assert np.allclose(pearson, pearson.T)
    assert np.allclose(spearman, spearman.T)
