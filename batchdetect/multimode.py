"""
Partial Python translation of the R code you posted (from the multimode package).

Implements:
- nmodes
- bw_crit
- cbws      (Silverman critical bandwidth test)
- cbwhy     (Hall and York critical bandwidth test)
- cramvm    (Cramer-von Mises statistic)
- cbwcvm    (Cramer-von Mises test using critical bandwidth)

Dependencies:
- numpy
- scipy (for normal cdf / pdf)
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

ArrayLike = Union[Sequence[float], np.ndarray]


# ---------------------------------------------------------------------
# Helper data structure (estmod equivalent)
# ---------------------------------------------------------------------


@dataclass
class EstMod:
    nmodes: int
    sample_size: int
    bw: float
    lowsup: float
    uppsup: float
    fnx: np.ndarray
    fny: np.ndarray


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------


def _clean_data(data: ArrayLike) -> np.ndarray:
    """
    Convert input data to a 1D float numpy array, removing NaNs and
    mimicking the R checks.
    """
    arr = np.asarray(data, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Argument 'data' must be numeric and non-empty.")
    if np.isnan(arr).any():
        warnings.warn("Missing values were removed", RuntimeWarning)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            raise ValueError(
                "No observations (at least after removing missing values)"
            )
    return arr


def _coerce_scalar_with_default(x, default, name: str) -> float:
    """
    Mimic the R behavior for lowsup/uppsup/n/tol etc:
    - Accept scalar-like values.
    - If length 0 or non-numeric, fall back to default with a warning.
    - If length > 1, use the first element with a warning.
    """
    if x is None:
        warnings.warn(
            f"Argument '{name}' must be specified. "
            f"Default value of '{name}' was used"
        )
        return default

    arr = np.atleast_1d(x)

    if arr.size == 0:
        warnings.warn(
            f"Argument '{name}' must be specified. "
            f"Default value of '{name}' was used"
        )
        return default

    if arr.size > 1:
        warnings.warn(
            f"Argument '{name}' has length > 1 and only the first element will be used"
        )

    try:
        val = float(arr[0])
    except (TypeError, ValueError):
        warnings.warn(
            f"Argument '{name}' must be numeric. "
            f"Default value of '{name}' was used"
        )
        return default

    return val


def _density_gaussian(
    data: np.ndarray,
    bw: float,
    n: int = 2**15,
    grid_from: Optional[float] = None,
    grid_to: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic 1D Gaussian KDE to mimic R's density(x, bw=h, n=n).

    data: 1D array of observations
    bw: bandwidth (h)
    n: number of grid points
    grid_from, grid_to: optional endpoints for the grid
    """
    data = np.asarray(data, dtype=float).ravel()
    n = int(n)
    if n <= 0:
        raise ValueError("Argument 'n' must be a positive integer")

    if grid_from is None:
        grid_from = data.min() - 3.0 * bw
    if grid_to is None:
        grid_to = data.max() + 3.0 * bw

    x = np.linspace(grid_from, grid_to, n)
    # Gaussian kernel
    diffs = (x[None, :] - data[:, None]) / bw
    y = np.exp(-0.5 * diffs**2).mean(axis=0) / (bw * np.sqrt(2.0 * np.pi))
    return x, y


def _mode_positions_from_kde(y: np.ndarray) -> np.ndarray:
    """
    Translate the R logic:

      z = 1:(n-1)
      re = z[diff(fn$y) > 0]
      z2 = 1:length(re)
      se = z2[diff(re) > 1]
      posic = re[se]
      if (re[length(re)] < (n-1)) posic = c(posic, re[length(re)])

    into zero-based Python indexing.
    """
    n = y.size
    # indices where the slope is positive (like re in R, but 0-based)
    re = np.where(np.diff(y) > 0)[0] + 1
    if re.size == 0:
        return np.array([], dtype=int)

    # positions in re where there is a gap > 1
    gap_idx = np.where(np.diff(re) > 1)[0]
    posic = re[gap_idx]

    # append the last rising index if it is not at the very end
    if re[-1] < (n - 1):
        posic = np.concatenate([posic, re[-1:]])

    return posic.astype(int)


# ---------------------------------------------------------------------
# nmodes
# ---------------------------------------------------------------------


def nmodes(
    data: ArrayLike,
    bw: float,
    lowsup: float = -np.inf,
    uppsup: float = np.inf,
    n: int = 2**15,
    full_result: bool = False,
) -> Union[int, EstMod]:
    """
    Python translation of the R function nmodes.

    Computes the number of modes of a kernel density estimate with
    Gaussian kernel and bandwidth bw, optionally restricted to the
    interval (lowsup, uppsup).

    If full_result is True, returns an EstMod object with the KDE grid.
    Otherwise returns an integer number of modes.
    """
    # Data check and NA removal
    data_arr = _clean_data(data)
    ndata = data_arr.size

    # Bandwidth checks
    try:
        bw = float(bw)
    except (TypeError, ValueError):
        raise TypeError("Argument 'bw' must be a positive number")

    if bw <= 0:
        raise ValueError("Argument 'bw' must be a positive number")

    # lowsup / uppsup handling (mimic R logic loosely)
    lowsup = _coerce_scalar_with_default(lowsup, -np.inf, "lowsup")
    uppsup = _coerce_scalar_with_default(uppsup, np.inf, "uppsup")

    if lowsup == uppsup:
        warnings.warn(
            "Arguments 'lowsup' and 'uppsup' must be different. "
            "Default values of 'lowsup' and 'uppsup' were used"
        )
        lowsup, uppsup = -np.inf, np.inf

    if lowsup > uppsup:
        warnings.warn(
            "Argument 'uppsup' must be greater than 'lowsup'. They were interchanged"
        )
        lowsup, uppsup = uppsup, lowsup

    # n checks
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        warnings.warn(
            "Argument 'n' must be a positive integer number. "
            "Default value of 'n' was used"
        )
        n_int = 2**15
    if n_int <= 0 or n_int != n_int:
        warnings.warn(
            "Argument 'n' must be a positive integer number. "
            "Default value of 'n' was used"
        )
        n_int = 2**15

    # KDE
    x, y = _density_gaussian(data_arr, bw=bw, n=n_int)

    # Mode positions
    posic = _mode_positions_from_kde(y)

    # Restrict to the interval (lowsup, uppsup)
    if posic.size > 0:
        mask = (x[posic] > lowsup) & (x[posic] < uppsup)
        posic = posic[mask]

    num = int(posic.size)

    if not full_result:
        return num

    return EstMod(
        nmodes=num,
        sample_size=ndata,
        bw=bw,
        lowsup=lowsup,
        uppsup=uppsup,
        fnx=x,
        fny=y,
    )


# ---------------------------------------------------------------------
# bw.crit  -> bw_crit
# ---------------------------------------------------------------------


def bw_crit(
    data: ArrayLike,
    mod0: int = 1,
    lowsup: float = -np.inf,
    uppsup: float = np.inf,
    n: int = 2**15,
    tol: float = 1e-5,
    full_result: bool = False,
) -> Union[float, EstMod]:
    """
    Python translation of R function bw.crit.

    Computes the critical bandwidth (Silverman-style) such that the
    Gaussian KDE of the data has at most 'mod0' modes.

    If full_result is True, returns an EstMod object including the KDE
    at the critical bandwidth.
    """
    data_arr = _clean_data(data)
    ndata = data_arr.size

    # mod0
    if not isinstance(mod0, (int, np.integer)):
        raise TypeError("Argument 'mod0' must be a positive integer number")
    if mod0 <= 0:
        raise ValueError("Argument 'mod0' must be a positive integer number")

    # lowsup, uppsup
    lowsup = _coerce_scalar_with_default(lowsup, -np.inf, "lowsup")
    uppsup = _coerce_scalar_with_default(uppsup, np.inf, "uppsup")

    if lowsup == uppsup:
        warnings.warn(
            "Arguments 'lowsup' and 'uppsup' must be different. "
            "Default values of 'lowsup' and 'uppsup' were used"
        )
        lowsup, uppsup = -np.inf, np.inf

    if ((lowsup > -np.inf) and np.isinf(uppsup)) or (
        np.isinf(lowsup) and (uppsup < np.inf)
    ):
        warnings.warn(
            "Both 'lowsup' and 'uppsup' must be finite or infinite. "
            "Default values of 'lowsup' and 'uppsup' were used"
        )
        lowsup, uppsup = -np.inf, np.inf

    if lowsup > uppsup:
        warnings.warn(
            "Argument 'uppsup' must be greater than 'lowsup'. They were interchanged"
        )
        lowsup, uppsup = uppsup, lowsup

    # n
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        warnings.warn(
            "Argument 'n' must be a positive integer number. Default value of 'n' was used"
        )
        n_int = 2**15
    if n_int <= 0:
        warnings.warn(
            "Argument 'n' must be a positive integer number. Default value of 'n' was used"
        )
        n_int = 2**15

    # tol
    try:
        tol = float(tol)
    except (TypeError, ValueError):
        warnings.warn(
            "Argument 'tol' must be a positive element. Default value of 'tol' was used"
        )
        tol = 1e-5
    if tol <= 0:
        warnings.warn(
            "Argument 'tol' must be a positive element. Default value of 'tol' was used"
        )
        tol = 1e-5

    # Initial bw: find bw0 with nmodes > mod0
    bw0 = 1.0
    while True:
        if nmodes(data_arr, bw0, lowsup, uppsup, n_int) > mod0:
            break
        bw0 /= 2.0
        if bw0 <= 0:
            raise RuntimeError(
                "Failed to find initial bandwidth with more than mod0 modes"
            )

    # Final bw: find bwf with nmodes <= mod0
    bwf = 2.0 * bw0
    while True:
        if nmodes(data_arr, bwf, lowsup, uppsup, n_int) <= mod0:
            break
        bwf *= 2.0

    # Bisection search
    while True:
        bwi = bw0 + (bwf - bw0) / 2.0
        num_i = nmodes(data_arr, bwi, lowsup, uppsup, n_int)
        if num_i <= mod0:
            bwf = bwi
        else:
            bw0 = bwi
        if (bwf - bw0) < tol:
            cbw = bwf
            break

    if not full_result:
        return cbw

    x, y = _density_gaussian(data_arr, bw=cbw, n=n_int)
    # In the R code, cbw$nmodes is set to mod0 (the null being tested),
    # not the actual number of modes with that bandwidth.
    return EstMod(
        nmodes=mod0,
        sample_size=ndata,
        bw=cbw,
        lowsup=lowsup,
        uppsup=uppsup,
        fnx=x,
        fny=y,
    )


# ---------------------------------------------------------------------
# Silverman critical bandwidth test (cbws)
# ---------------------------------------------------------------------


def cbws(
    data: ArrayLike,
    mod0: int = 1,
    B: int = 500,
    methodsi: int = 1,
    n: int = 2**10,
    tol: float = 1e-5,
) -> Tuple[float, float]:
    """
    Python translation of cbws: Silverman (1981) critical bandwidth test.

    Returns (p_value, cbw).

    methodsi:
      1 -> Silverman's original rescaling step
      2 -> simple bootstrap without rescaling
    """
    data_arr = _clean_data(data)
    ndata = data_arr.size

    if not isinstance(B, (int, np.integer)) or B <= 0:
        warnings.warn(
            "Argument 'B' must be a positive integer number. Default value of 'B' was used"
        )
        B = 500

    # Critical bandwidth for original sample
    cbw = bw_crit(data_arr, mod0=mod0, n=n, tol=tol)

    cbwB = np.empty(B, dtype=float)

    for i in range(B):
        # bootstrap sample + Gaussian noise with sd = cbw
        idx = np.random.randint(0, ndata, size=ndata)
        samp = data_arr[idx]
        eps = np.random.normal(loc=0.0, scale=cbw, size=ndata)
        dataB = samp + eps

        if methodsi == 1:
            # Silverman rescaling
            sd_dataB = np.std(dataB, ddof=1)
            if sd_dataB > 0:
                factor = np.sqrt(1.0 + (cbw / sd_dataB) ** 2)
                dataB = factor * dataB

        cbwB[i] = bw_crit(dataB, mod0=mod0, n=n, tol=tol)

    pv = float(np.mean(cbw < cbwB))
    return pv, float(cbw)


# ---------------------------------------------------------------------
# Hall and York critical bandwidth test (cbwhy)
# ---------------------------------------------------------------------


def cbwhy(
    data: ArrayLike,
    lowsup: float,
    uppsup: float,
    B: int = 500,
    methodhy: int = 1,
    alpha: float = 0.05,
    n: int = 2**10,
    tol: float = 1e-5,
    nMC: int = 100,
    BMC: int = 100,
) -> Tuple[float, float]:
    """
    Python translation of cbwhy: Hall and York (2001) critical bandwidth test.

    This is only for testing unimodality (mod0=1). Returns (p_value, cbw).

    methodhy:
      1 -> analytical calibration using lambda(alpha)
      2 -> Monte Carlo calibration
    """
    data_arr = _clean_data(data)
    ndata = data_arr.size
    mod0 = 1

    lowsup = _coerce_scalar_with_default(lowsup, -np.inf, "lowsup")
    uppsup = _coerce_scalar_with_default(uppsup, np.inf, "uppsup")

    if lowsup == uppsup:
        warnings.warn(
            "Arguments 'lowsup' and 'uppsup' must be different. "
            "Default values of 'lowsup' and 'uppsup' were used"
        )
        lowsup, uppsup = -np.inf, np.inf

    if lowsup > uppsup:
        warnings.warn(
            "Argument 'uppsup' must be greater than 'lowsup'. They were interchanged"
        )
        lowsup, uppsup = uppsup, lowsup

    if np.isinf(lowsup) or np.isinf(uppsup):
        warnings.warn(
            "Hall and York (2001) test is designed for bounded support; "
            "lowsup and uppsup should be finite."
        )

    if not isinstance(B, (int, np.integer)) or B <= 0:
        warnings.warn(
            "Argument 'B' must be a positive integer number. Default value of 'B' was used"
        )
        B = 500

    # Critical bandwidth for original data
    cbw = bw_crit(
        data_arr, mod0=mod0, lowsup=lowsup, uppsup=uppsup, n=n, tol=tol
    )

    # Bootstrap critical bandwidths
    cbwB = np.empty(B, dtype=float)
    for i in range(B):
        idx = np.random.randint(0, ndata, size=ndata)
        samp = data_arr[idx]
        eps = np.random.normal(loc=0.0, scale=cbw, size=ndata)
        dataB = samp + eps
        cbwB[i] = bw_crit(
            dataB, mod0=mod0, lowsup=lowsup, uppsup=uppsup, n=n, tol=tol
        )

    if methodhy == 1:
        # Method 1 (analytical lambda(alpha))
        num = (
            0.94029 * alpha**3
            - 1.59914 * alpha**2
            + 0.17695 * alpha
            + 0.48971
        )
        den = alpha**3 - 1.77793 * alpha**2 + 0.36162 * alpha + 0.42423
        lam = num / den
        pv = float(np.mean((cbw * lam) < cbwB))
    else:
        # Method 2 (Monte Carlo calibration)
        # Step 1: MC p-values for standard normal
        pvMC = np.empty(nMC, dtype=float)
        for i in range(nMC):
            dataMC = np.random.normal(size=ndata)
            cbwMC = bw_crit(
                dataMC, mod0=mod0, lowsup=-1.5, uppsup=1.5, n=n, tol=tol
            )
            cbwBMC = np.empty(BMC, dtype=float)
            for j in range(BMC):
                idx = np.random.randint(0, ndata, size=ndata)
                samp = dataMC[idx]
                eps = np.random.normal(loc=0.0, scale=cbwMC, size=ndata)
                dataBMC = samp + eps
                cbwBMC[j] = bw_crit(
                    dataBMC,
                    mod0=mod0,
                    lowsup=-1.5,
                    uppsup=1.5,
                    n=n,
                    tol=tol,
                )
            pvMC[i] = np.mean(cbwMC < cbwBMC)

        pv_silverman = np.mean(cbw < cbwB)
        pv = float(np.mean(pvMC < pv_silverman))

    return pv, float(cbw)


# ---------------------------------------------------------------------
# Cramer-von Mises statistic (cramvm) and test (cbwcvm)
# ---------------------------------------------------------------------


def cramvm(data: ArrayLike, bw: float) -> float:
    """
    Python translation of cramvm: Cramer-von Mises test statistic
    based on the kernel distribution estimator (KDE CDF) evaluated
    at the sample points.
    """
    data_arr = _clean_data(data)
    ndata = data_arr.size

    # F_hat(x_i) = (1/n) sum_j Phi( (x_i - x_j) / bw )
    diffs = (data_arr[:, None] - data_arr[None, :]) / bw
    # normal CDF of each entry
    U = stats.norm.cdf(diffs).mean(axis=1)

    # sort U
    U_sorted = np.sort(U)
    k = np.arange(1, ndata + 1, dtype=float)
    # R formula: (U - (2k-1)/(2n))^2 + 1/(12n), then sum
    sumand = (U_sorted - (2.0 * k - 1.0) / (2.0 * ndata)) ** 2 + 1.0 / (
        12.0 * ndata
    )
    Tk = float(np.sum(sumand))
    return Tk


def cbwcvm(
    data: ArrayLike,
    mod0: int = 1,
    B: int = 500,
    n: int = 2**10,
    tol: float = 1e-5,
) -> Tuple[float, float]:
    """
    Python translation of cbwcvm: Fisher and Marron (2001)
    Cramer-von Mises test based on the critical bandwidth.

    Returns (p_value, Tk), where Tk is the observed test statistic.
    """
    data_arr = _clean_data(data)
    ndata = data_arr.size

    if not isinstance(B, (int, np.integer)) or B <= 0:
        warnings.warn(
            "Argument 'B' must be a positive integer number. Default value of 'B' was used"
        )
        B = 500

    # Critical bandwidth and observed statistic
    cbw = bw_crit(data_arr, mod0=mod0, n=n, tol=tol)
    Tk = cramvm(data_arr, cbw)

    # Bootstrap replicates
    TkB = np.empty(B, dtype=float)
    for i in range(B):
        idx = np.random.randint(0, ndata, size=ndata)
        samp = data_arr[idx]
        eps = np.random.normal(loc=0.0, scale=cbw, size=ndata)
        dataB = samp + eps
        cbwB = bw_crit(dataB, mod0=mod0, n=n, tol=tol)
        TkB[i] = cramvm(dataB, cbwB)

    pv = float(np.mean(Tk < TkB))
    return pv, Tk
