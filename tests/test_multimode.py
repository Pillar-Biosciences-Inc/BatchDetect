# test_multimode_py.py

import numpy as np
import pytest

from batchdetect.loader import load_multimode
from batchdetect.multimode import (
    EstMod,
    bw_crit,
    cbwcvm,
    cbwhy,
    cbws,
    cramvm,
    nmodes,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def silverman_bandwidth(x: np.ndarray) -> float:
    """Standard Silverman rule of thumb bandwidth."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 points for bandwidth.")
    std = x.std(ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    return 0.9 * sigma * n ** (-1.0 / 5.0)


# ---------------------------------------------------------------------
# nmodes tests
# ---------------------------------------------------------------------


def test_nmodes_unimodal_gaussian():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.0, scale=1.0, size=500)
    h = silverman_bandwidth(data)

    m = nmodes(data, bw=h)
    assert isinstance(m, int)
    # For a reasonably smooth unimodal Gaussian, we expect 1 mode.
    # assert m == 1


def test_nmodes_bimodal_mixture():
    rng = np.random.default_rng(1)
    n = 500
    data1 = rng.normal(loc=-3.0, scale=1.0, size=n)
    data2 = rng.normal(loc=3.0, scale=1.0, size=n)
    data = np.concatenate([data1, data2])

    h = silverman_bandwidth(data)
    m = nmodes(data, bw=h)
    # Two well separated normals -> expect two modes
    assert m == 2


def test_nmodes_full_result_returns_estmod():
    rng = np.random.default_rng(2)
    data = rng.normal(size=200)
    h = silverman_bandwidth(data)

    result = nmodes(data, bw=h, full_result=True)
    assert isinstance(result, EstMod)
    assert result.sample_size == len(data)
    assert result.bw == pytest.approx(h)
    assert result.fnx.shape == result.fny.shape
    assert result.nmodes >= 1


# ---------------------------------------------------------------------
# bw_crit tests
# ---------------------------------------------------------------------


def test_bw_crit_unimodal_has_at_most_one_mode():
    rng = np.random.default_rng(3)
    data = rng.normal(size=300)

    cbw = bw_crit(data, mod0=1, n=2**9, tol=1e-4)
    assert cbw > 0.0

    # By construction, nmodes at the critical bandwidth should be <= mod0
    m = nmodes(data, bw=cbw, n=2**9)
    assert m <= 1


def test_bw_crit_full_result_estmod():
    rng = np.random.default_rng(4)
    data = rng.normal(size=100)

    res = bw_crit(data, mod0=1, n=2**9, tol=1e-4, full_result=True)
    assert isinstance(res, EstMod)
    assert res.sample_size == len(data)
    assert res.bw > 0.0
    assert res.lowsup <= res.uppsup


def test_bw_comparison_multimode():
    acidity, enzyme, stamps = load_multimode()

    assert np.abs(bw_crit(acidity) - 0.6350098) < 1e-4
    assert np.abs(bw_crit(enzyme) - 0.2993164) < 1e-4
    assert np.abs(bw_crit(stamps) - 0.006729126) < 1e-4

    assert np.abs(bw_crit(acidity, mod0=2) - 0.2605515) < 1e-4
    assert np.abs(bw_crit(enzyme, mod0=2) - 0.161293) < 1e-4
    assert np.abs(bw_crit(stamps, mod0=2) - 0.003234863) < 1e-4

    assert np.abs(bw_crit(acidity, n=10) - 0.5775833) < 1e-2
    assert np.abs(bw_crit(enzyme, n=10) - 0.2588425) < 1e-2
    assert np.abs(bw_crit(stamps, n=10) - 0.005226135) < 1e-2

    assert np.abs(bw_crit(acidity, lowsup=5, uppsup=6) - 0.08103943) < 1e-2
    assert np.abs(bw_crit(enzyme, lowsup=1, uppsup=2) - 0.0874176) < 1e-2
    assert np.abs(bw_crit(stamps, lowsup=0.1, uppsup=1) - 0.003013611) < 1e-2


# ---------------------------------------------------------------------
# cbws: Silverman critical bandwidth test
# ---------------------------------------------------------------------


def test_cbws_pvalue_range_and_cbw_positive():
    rng = np.random.default_rng(5)
    data = rng.normal(size=200)

    # Use smaller B and n for speed in unit tests
    np.random.seed(123)  # make bootstrap deterministic
    pv, cbw = cbws(data, mod0=1, B=30, methodsi=1, n=2**8, tol=1e-4)

    assert 0.0 <= pv <= 1.0
    assert cbw > 0.0


# ---------------------------------------------------------------------
# cbwhy: Hall and York critical bandwidth test
# ---------------------------------------------------------------------


def test_cbwhy_bounded_support():
    rng = np.random.default_rng(6)
    data = rng.normal(size=150)

    lowsup, uppsup = -3.0, 3.0
    np.random.seed(456)
    pv, cbw = cbwhy(
        data,
        lowsup=lowsup,
        uppsup=uppsup,
        B=20,
        methodhy=1,
        alpha=0.05,
        n=2**8,
        tol=1e-4,
        nMC=20,  # unused for method 1
        BMC=20,  # unused for method 1
    )

    assert 0.0 <= pv <= 1.0
    assert cbw > 0.0


def test_cbwhy():
    acidity, enzyme, stamps = load_multimode()

    pv, cbw = cbwhy(stamps, B=2000, lowsup=0, uppsup=1)
    print(pv)
    assert np.abs(cbw - 0.0067291) < 1e-4

    pv, cbw = cbwhy(enzyme, B=2000, lowsup=0, uppsup=1)
    print(pv)
    assert np.abs(cbw - 0.11434) < 1e-2
    assert np.abs(pv - 0.2255) < 1e-2


# ---------------------------------------------------------------------
# cramvm and cbwcvm tests
# ---------------------------------------------------------------------


def test_cramvm_returns_positive_value():
    rng = np.random.default_rng(7)
    data = rng.normal(size=80)
    h = silverman_bandwidth(data)

    stat = cramvm(data, bw=h)
    assert np.isfinite(stat)
    assert stat > 0.0


def test_cbwcvm():
    acidity, enzyme, stamps = load_multimode()

    pv, Tk = cbwcvm(stamps, B=2000)
    assert np.abs(Tk - 1.2302) < 1e-2

    pv, Tk = cbwcvm(enzyme, B=2000)
    assert np.abs(Tk - 2.1794) < 1e-2


def test_cbwcvm_pvalue_range():
    rng = np.random.default_rng(8)
    data = rng.normal(size=120)

    np.random.seed(789)
    pv, stat = cbwcvm(data, mod0=1, B=20, n=2**8, tol=1e-4)

    assert 0.0 <= pv <= 1.0
    assert np.isfinite(stat)
    assert stat > 0.0
