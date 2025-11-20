import numpy as np

from batchdetect.loader import load_multimode


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
    assert stamps.ndim == 2

    assert acidity.size > 0
    assert enzyme.size > 0
    assert stamps.shape[0] > 0
    assert stamps.shape[1] > 0


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
    assert np.array_equal(stamps_t, data_dict["stamps"])
