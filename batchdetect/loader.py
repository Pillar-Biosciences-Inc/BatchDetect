import importlib.resources as pkg_resources
import io
from typing import Dict

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


def _load_npz_from_package(filename: str) -> Dict[str, np.ndarray]:
    """Load a packaged .npz file into a dict of arrays."""
    data_path = pkg_resources.files(__package__).joinpath(f"data/{filename}")
    with data_path.open("rb") as f:
        buffer = io.BytesIO(f.read())
    # features/target_names are stored as object arrays; allow_pickle=True is required
    # (we're only unpickling Python strings saved by us).
    return dict(np.load(buffer, allow_pickle=True))


def _format_return(
    data: Dict[str, np.ndarray],
    *,
    return_X_y: bool,
    return_feature_names: bool,
    return_target_names: bool,
):
    if not return_X_y:
        return data

    X = data["X"]
    y = data["y"]
    sample_id = data["sample_id"]
    if return_feature_names and return_target_names:
        return X, y, sample_id, data["features"], data["target_names"]
    if return_feature_names and not return_target_names:
        return X, y, sample_id, data["features"]
    if (not return_feature_names) and return_target_names:
        return X, y, sample_id, data["target_names"]
    return X, y, sample_id


def load_br283_cross_lot_covs(
    return_X_y: bool = True,
    return_feature_names: bool = True,
    return_target_names: bool = True,
):
    """Load BR283 cross-lot covariates + labels.

    Packaged file contains:
    - X: (n_samples, n_features)
    - y: (n_samples, n_targets)
    - sample_id: (n_samples,)
    - features: (n_features,)
    - target_names: (n_targets,)
    """
    data = _load_npz_from_package("BR283_cross_lot_covs.npz")
    return _format_return(
        data,
        return_X_y=return_X_y,
        return_feature_names=return_feature_names,
        return_target_names=return_target_names,
    )


def load_thal_cross_lot_covs(
    return_X_y: bool = True,
    return_feature_names: bool = True,
    return_target_names: bool = True,
):
    """Load Thal cross-lot covariates + labels."""
    data = _load_npz_from_package("Thal_cross_lot_covs.npz")
    return _format_return(
        data,
        return_X_y=return_X_y,
        return_feature_names=return_feature_names,
        return_target_names=return_target_names,
    )


def load_essential_ffpe(
    return_X_y: bool = True,
    return_feature_names: bool = True,
    return_target_names: bool = True,
):
    """Load Essential FFPE covariates + labels."""
    data = _load_npz_from_package("Essential_ffpe.npz")
    return _format_return(
        data,
        return_X_y=return_X_y,
        return_feature_names=return_feature_names,
        return_target_names=return_target_names,
    )


def load_covariates(
    name: str,
    return_X_y: bool = True,
    return_feature_names: bool = True,
    return_target_names: bool = True,
):
    """Generic loader.

    Parameters
    ----------
    name : {"br283_cross_lot_covs", "thal_cross_lot_covs", "essential_ffpe"}
        Dataset identifier (case-insensitive).
    """
    key = name.strip().lower()
    if key in {"br283", "br283_cross_lot_covs", "br283_cross_lot"}:
        return load_br283_cross_lot_covs(
            return_X_y=return_X_y,
            return_feature_names=return_feature_names,
            return_target_names=return_target_names,
        )
    if key in {"thal", "thal_cross_lot_covs", "thal_cross_lot"}:
        return load_thal_cross_lot_covs(
            return_X_y=return_X_y,
            return_feature_names=return_feature_names,
            return_target_names=return_target_names,
        )
    if key in {"essential_ffpe", "essential"}:
        return load_essential_ffpe(
            return_X_y=return_X_y,
            return_feature_names=return_feature_names,
            return_target_names=return_target_names,
        )
    raise ValueError(f"Unknown dataset name: {name!r}")
