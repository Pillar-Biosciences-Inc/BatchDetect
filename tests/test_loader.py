import numpy as np
import pytest

from batchdetect.loader import (
    load_br283_cross_lot_covs,
    load_covariates,
    load_essential_ffpe,
    load_multimode,
    load_thal_cross_lot_covs,
)


def test_load_multimode_tuple_return():
    """load_multimode should return three nonempty ndarrays by default."""
    acidity, enzyme, stamps = load_multimode()

    # Types
    assert isinstance(acidity, np.ndarray)
    assert isinstance(enzyme, np.ndarray)
    assert isinstance(stamps, np.ndarray)

    # Basic shape checks (these are intentionally weak so they do not depend
    # on exact dataset sizes)
    assert acidity.ndim == 1
    assert enzyme.ndim == 1
    assert stamps.ndim == 1

    assert acidity.size > 0
    assert enzyme.size > 0
    assert stamps.shape[0] > 0


def test_load_multimode_return_dict():
    """return_dict=True should return the same arrays, keyed by name."""
    data_dict = load_multimode(return_dict=True)

    assert isinstance(data_dict, dict)

    for key in ("acidity", "enzyme", "stamps"):
        assert key in data_dict
        assert isinstance(data_dict[key], np.ndarray)
        assert data_dict[key].size > 0

    # The dict path and tuple path should be consistent
    acidity_t, enzyme_t, stamps_t = load_multimode()

    assert np.array_equal(acidity_t, data_dict["acidity"])
    assert np.array_equal(enzyme_t, data_dict["enzyme"])
    assert np.array_equal(stamps_t, np.squeeze(data_dict["stamps"]))


def test_load_br283_cross_lot_covs():
    # Test default return
    X, y, sample_id, features, target_names = load_br283_cross_lot_covs()

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(sample_id, np.ndarray)
    assert isinstance(features, np.ndarray)
    assert isinstance(target_names, np.ndarray)

    assert X.ndim == 2
    assert y.ndim == 2
    assert X.shape[0] == y.shape[0] == sample_id.shape[0]
    assert X.shape[1] == features.shape[0]
    assert y.shape[1] == target_names.shape[0]


def test_load_thal_cross_lot_covs():
    # Test default return
    X, y, sample_id, features, target_names = load_thal_cross_lot_covs()

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(sample_id, np.ndarray)
    assert isinstance(features, np.ndarray)
    assert isinstance(target_names, np.ndarray)

    assert X.ndim == 2
    assert y.ndim == 2
    assert X.shape[0] == y.shape[0] == sample_id.shape[0]
    assert X.shape[1] == features.shape[0]
    assert y.shape[1] == target_names.shape[0]


def test_load_essential_ffpe():
    # Test default return
    X, y, sample_id, features, target_names = load_essential_ffpe()

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(sample_id, np.ndarray)
    assert isinstance(features, np.ndarray)
    assert isinstance(target_names, np.ndarray)

    assert X.ndim == 2
    assert y.ndim == 2
    assert X.shape[0] == y.shape[0] == sample_id.shape[0]
    assert X.shape[1] == features.shape[0]
    assert y.shape[1] == target_names.shape[0]


def test_load_covariates():
    # Test loading by name
    for name in ["br283", "thal", "essential"]:
        data = load_covariates(name)
        assert len(data) == 5  # Default returns 5 items

    # Test case insensitivity and variations
    data = load_covariates("BR283_Cross_Lot_Covs")
    assert len(data) == 5

    # Test return_X_y=False
    data_dict = load_covariates("br283", return_X_y=False)
    assert isinstance(data_dict, dict)
    assert "X" in data_dict
    assert "y" in data_dict

    # Test invalid name
    with pytest.raises(ValueError, match="Unknown dataset name"):
        load_covariates("invalid_name")
