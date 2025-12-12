import numpy as np
import pytest

from batchdetect.quantro import quantro


def test_quantro_input_validation():
    # 1. 1D array -> expect ValueError
    x_1d = np.array([1, 2, 3])
    group_1d = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="object must be 2D"):
        quantro(x_1d, group_1d)

    # 2. Mismatched group factor length -> expect ValueError
    x_2d = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2) shaped

    # So if X is (3, 2), n_samples is 2. group should be length 2.
    # Let's make a mismatch.
    group_mismatch_len = np.array([0, 1, 0])
    with pytest.raises(
        ValueError, match="Number of columns in object does not match"
    ):
        quantro(x_2d, group_mismatch_len)

    # 3. B < 0 -> expect ValueError
    group_match = np.array([0, 1])
    with pytest.raises(
        ValueError, match="Must pick B greater than or equal to 0"
    ):
        quantro(x_2d, group_match, B=-1)


def test_quantro_basic_no_perm():
    # Setup synthetic data
    # 2 groups, 5 samples each, 10 features
    np.random.seed(42)
    n_features = 10
    n_samples = 10
    X = np.random.randn(n_features, n_samples)
    # Introduce some group effect
    group = np.array([0] * 5 + [1] * 5)
    X[:, 5:] += 2.0  # shift group 1

    res = quantro(X, group, B=0, verbose=False)

    assert isinstance(res, dict)
    keys = [
        "anova",
        "MSbetween",
        "MSwithin",
        "quantroStat",
        "quantroStatPerm",
        "quantroPvalPerm",
    ]
    for k in keys:
        assert k in res

    # Check values
    assert res["quantroStatPerm"] is None
    assert res["quantroPvalPerm"] is None
    assert isinstance(res["anova"], dict)
    assert "F" in res["anova"]
    assert "pvalue" in res["anova"]
    assert isinstance(res["quantroStat"], float)
    assert res["quantroStat"] > 0


def test_quantro_permutation():
    # Setup synthetic data
    np.random.seed(123)
    n_features = 5
    n_samples = 6
    X = np.random.randn(n_features, n_samples)
    group = np.array([0, 0, 0, 1, 1, 1])

    B = 10
    res = quantro(X, group, B=B, verbose=False, seed=42)

    assert res["quantroStatPerm"] is not None
    assert len(res["quantroStatPerm"]) == B
    assert isinstance(res["quantroPvalPerm"], float)
    assert 0.0 <= res["quantroPvalPerm"] <= 1.0


def test_quantro_options():
    np.random.seed(1)
    X = np.random.randn(5, 6)
    group = np.array([0, 0, 0, 1, 1, 1])

    # Test useMedianNormalized=False
    res_no_norm = quantro(
        X, group, B=0, useMedianNormalized=False, verbose=False
    )
    # Just check it returns valid result
    assert res_no_norm["quantroStat"] > 0

    # Test qRange
    q_range = [0.25, 0.5, 0.75]
    res_q = quantro(X, group, B=0, qRange=q_range, verbose=False)
    assert res_q["quantroStat"] > 0


def test_quantro_identical_medians():
    # Create data where every column has the same median value,
    # so object_medians (median of X, axis=0) are all identical.
    # Actually, object_medians is `np.median(X, axis=0)`.
    # That is the median of each column (sample).
    # "All median values equal" checks `np.unique(object_medians).size == 1`.

    # Let's make every sample identical
    col = np.array([1, 2, 3])
    X = np.column_stack([col, col, col, col])  # (3, 4)
    group = np.array([0, 0, 1, 1])

    # Medians of each column will be 2.0
    # So unique medians size is 1.

    res = quantro(X, group, B=0, verbose=True)

    # Expect anova to be None because "All median values equal. No ANOVA performed."
    assert res["anova"] is None
    # Calculation should proceed with X_norm (if useMedianNormalized)
    # or X (if not, but here we expect no median normalization if size==1?
    # Code says:
    # if unique...size == 1: print message
    # else: do anova
    # Indepdendent of that:
    # if useMedianNormalized: X_norm = X - medians
    # else: X_norm = X
    # So it just skips ANOVA.

    assert res["quantroStat"] is not None
