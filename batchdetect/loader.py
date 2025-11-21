import importlib.resources as pkg_resources
import io

import numpy as np


def load_multimode(return_dict: bool = False):
    """
    Load multimode example datasets from a packaged NPZ file.

    Parameters
    ----------
    return_dict : bool, default=False
        If True, return a dict of all arrays in the NPZ file.
        If False, return a tuple with selected arrays.

    Returns
    -------
    dict or tuple
        - If return_dict is True: {name: ndarray, ...}
        - Else: (acidity, enzyme, stamps)
    """
    data_path = pkg_resources.files(__package__).joinpath(
        "data/multimode_datasets.npz"
    )

    with data_path.open("rb") as f:
        buffer = io.BytesIO(f.read())
    data = np.load(buffer, allow_pickle=False)

    if return_dict:
        # Return all arrays in a dict
        return {name: data[name] for name in data.files}

    # Or explicitly pick out what you want
    acidity = np.squeeze(data["acidity"])
    enzyme = np.squeeze(data["enzyme"])
    stamps = np.squeeze(data["stamps"])

    return acidity, enzyme, stamps
