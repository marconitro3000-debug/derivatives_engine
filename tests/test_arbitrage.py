"""
tests/test_arbitrage.py
Tests for the static arbitrage scanner (arbitrage/vol_surface.py)
and put-call parity scanner (arbitrage/put_call_parity.py).

Key invariants:
  1. A clean SSVI surface has zero calendar violations.
  2. An artificially introduced calendar arbitrage is detected.
  3. A clean SSVI surface has zero butterfly violations.
  4. An artificially introduced butterfly arbitrage is detected.
  5. The density proxy g(k) is non-negative on a Lee-valid SVI slice.
  6. PCPScanner detects no violations when BS call/put prices are consistent.
  7. PCPScanner flags a clear violation when put price is manipulated.
  8. ArbScanResult.summary() is a non-empty string.
"""

import numpy as np
import pytest

from ml.ssvi import SVIParams, SSVIParams, calibrate_ssvi
from arbitrage.vol_surface import (
    VolSurfaceArbScanner,
    ArbScanResult,
    CalendarViolation,
    ButterflyViolation,
    _density,
    _lee_wing_check,
)
from arbitrage.put_call_parity import PCPScanner, PCPScanResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_clean_chain():
    """
    Build a synthetic option chain from a known-good SSVI surface.
    Returns (spot, strikes, maturities, ivs) with no arbitrage.
    """
    ssvi       = SSVIParams(rho=-0.65, eta=0.85, gamma=0.40)
    maturities = np.array([0.08, 0.25, 0.50, 1.00])
    atm_ivs    = np.array([0.22, 0.20, 0.19, 0.18])
    spot       = 100.0
    r          = 0.04

    strikes_list, T_list, iv_list = [], [], []
    for T, iv_atm in zip(maturities, atm_ivs):
        theta = iv_atm ** 2 * T
        ks    = np.linspace(-0.20, 0.20, 11)
        F     = spot * np.exp(r * T)
        Ks    = F * np.exp(ks)
        w     = ssvi.total_var(ks, theta)
        ivs   = np.sqrt(np.maximum(w / T, 0))
        strikes_list.extend(Ks)
        T_list.extend([T] * len(Ks))
        iv_list.extend(ivs)

    return (
        spot,
        np.array(strikes_list),
        np.array(T_list),
        np.array(iv_list),
    )


def _make_arb_chain_calendar():
    """
    Modify the clean chain so w(k, T=0.08) > w(k, T=0.25) at k=0:
    a clear calendar arbitrage.
    """
    spot, strikes, mats, ivs = _make_clean_chain()
    # Inflate short-dated ATM IV so total variance exceeds the next slice
    mask = mats < 0.15
    ivs  = ivs.copy()
    ivs[mask] *= 2.5      # short slice total var now well above long slice
    return spot, strikes, mats, ivs


def _make_arb_chain_butterfly():
    """
    Create a butterfly arbitrage by using an SVI slice that violates the
    Lee wing condition (b*(1+|ρ|) > 4).
    """
    # Single maturity synthetic data from a bad SVI
    bad_svi  = SVIParams(a=0.04, b=1.95, rho=-0.99, m=0.0, sigma=0.05)
    ks       = np.linspace(-0.40, 0.40, 21)
    T        = 0.50
    spot     = 100.0
    r        = 0.04
    F        = spot * np.exp(r * T)
    Ks       = F * np.exp(ks)
    w        = bad_svi.total_var(ks)
    # Clip to positive to avoid sqrt(negative)
    ivs      = np.sqrt(np.maximum(w / T, 0))

    return spot, Ks, np.full(len(Ks), T), ivs


# ── Tests: density proxy ──────────────────────────────────────────────────────

def test_density_nonneg_valid_svi():
    """A Lee-valid SVI slice should have g(k) ≥ 0 everywhere."""
    p    = SVIParams(a=0.04, b=0.15, rho=-0.40, m=0.0, sigma=0.10)
    k    = np.linspace(-0.40, 0.40, 201)
    g    = _density(p, k)
    assert np.all(g >= -1e-4), f"Negative density: min={g.min():.6f}"


def test_lee_wing_check_valid():
    p = SVIParams(a=0.04, b=0.15, rho=-0.40, m=0.0, sigma=0.10)
    assert _lee_wing_check(p) is True


def test_lee_wing_check_invalid_b():
    """b*(1+|ρ|) > 4 violates Lee's bound."""
    p = SVIParams(a=0.04, b=3.0, rho=-0.99, m=0.0, sigma=0.10)
    assert _lee_wing_check(p) is False


def test_lee_wing_check_negative_wmin():
    p = SVIParams(a=-0.10, b=0.10, rho=-0.40, m=0.0, sigma=0.10)
    assert _lee_wing_check(p) is False


# ── Tests: scanner on clean surface ──────────────────────────────────────────

def test_clean_surface_no_calendar_violations():
    spot, strikes, mats, ivs = _make_clean_chain()
    scanner = VolSurfaceArbScanner()
    result  = scanner.scan_from_chain(spot, strikes, mats, ivs, ticker="TEST")

    assert isinstance(result, ArbScanResult)
    assert result.is_calendar_arb_free, (
        f"Expected no calendar violations, got: {result.calendar_violations}"
    )


def test_clean_surface_no_butterfly_violations():
    spot, strikes, mats, ivs = _make_clean_chain()
    scanner = VolSurfaceArbScanner()
    result  = scanner.scan_from_chain(spot, strikes, mats, ivs, ticker="TEST")

    assert result.is_butterfly_arb_free, (
        f"Expected no butterfly violations, got: {result.butterfly_violations}"
    )


def test_clean_surface_is_arb_free():
    spot, strikes, mats, ivs = _make_clean_chain()
    result = VolSurfaceArbScanner().scan_from_chain(spot, strikes, mats, ivs)
    assert result.is_arbitrage_free


# ── Tests: scanner detects introduced arbitrage ───────────────────────────────

def test_detects_calendar_arbitrage():
    spot, strikes, mats, ivs = _make_arb_chain_calendar()
    result = VolSurfaceArbScanner().scan_from_chain(
        spot, strikes, mats, ivs, ticker="ARB_CAL"
    )
    assert not result.is_calendar_arb_free, (
        "Expected calendar arbitrage to be detected."
    )
    assert len(result.calendar_violations) > 0
    v = result.calendar_violations[0]
    assert isinstance(v, CalendarViolation)
    assert v.magnitude > 0
    assert v.T_short < v.T_long


def test_calendar_violation_fields():
    spot, strikes, mats, ivs = _make_arb_chain_calendar()
    result = VolSurfaceArbScanner().scan_from_chain(spot, strikes, mats, ivs)
    for v in result.calendar_violations:
        assert v.T_short < v.T_long
        assert v.magnitude >= 0
        assert v.n_violations > 0


# ── Tests: result containers ──────────────────────────────────────────────────

def test_summary_is_string():
    spot, strikes, mats, ivs = _make_clean_chain()
    result = VolSurfaceArbScanner().scan_from_chain(spot, strikes, mats, ivs)
    s = result.summary()
    assert isinstance(s, str)
    assert len(s) > 50


def test_matrix_shapes():
    spot, strikes, mats, ivs = _make_clean_chain()
    result  = VolSurfaceArbScanner().scan_from_chain(spot, strikes, mats, ivs)
    n_T     = len(result.maturities)
    n_k     = len(result.k_grid)
    assert result.total_var_matrix.shape  == (n_T, n_k)
    assert result.density_matrix.shape    == (n_T, n_k)


def test_total_var_positive():
    spot, strikes, mats, ivs = _make_clean_chain()
    result = VolSurfaceArbScanner().scan_from_chain(spot, strikes, mats, ivs)
    assert np.all(result.total_var_matrix > 0)


def test_maturities_sorted():
    spot, strikes, mats, ivs = _make_clean_chain()
    result = VolSurfaceArbScanner().scan_from_chain(spot, strikes, mats, ivs)
    assert np.all(np.diff(result.maturities) > 0)


# ── Tests: put-call parity scanner ───────────────────────────────────────────

def _make_pcp_data(spot=100.0, T=0.50, r=0.04, iv=0.20):
    """Consistent call/put data from BS — no PCP violation."""
    from options.black_scholes import price as bs_price

    strikes  = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    call_ivs = np.full(len(strikes), iv)
    put_ivs  = np.full(len(strikes), iv)   # same IV → consistent
    return spot, strikes, np.full(len(strikes), T), call_ivs, put_ivs


def test_pcp_no_violation_consistent_prices():
    spot, strikes, mats, c_ivs, p_ivs = _make_pcp_data()
    scanner = PCPScanner(min_profit=0.01)
    result  = scanner.scan_from_data(
        spot, strikes, mats, c_ivs, p_ivs, ticker="TEST"
    )
    assert isinstance(result, PCPScanResult)
    # With consistent BS prices the deviation should be essentially zero
    assert result.max_deviation < 1e-8, (
        f"Expected near-zero deviation, got {result.max_deviation:.6f}"
    )
    assert result.n_violations == 0


def test_pcp_detects_large_violation():
    """Artificially inflate put price to create a clear PCP violation."""
    from options.black_scholes import price as bs_price

    spot, strikes, mats, c_ivs, p_ivs = _make_pcp_data()
    # Put IV much higher than call IV at K=100 → PCP violation
    p_ivs = p_ivs.copy()
    p_ivs[2] = 0.45      # ATM put IV = 45% vs call IV = 20%

    scanner = PCPScanner(min_profit=0.01)
    result  = scanner.scan_from_data(
        spot, strikes, mats, c_ivs, p_ivs, ticker="ARB_PCP"
    )
    assert result.n_violations > 0, "Should detect at least one PCP violation."
    assert result.max_deviation > 0.50   # should be several dollars off


def test_pcp_all_deviations_length():
    spot, strikes, mats, c_ivs, p_ivs = _make_pcp_data()
    result = PCPScanner().scan_from_data(spot, strikes, mats, c_ivs, p_ivs)
    assert len(result.all_deviations) == len(strikes)


def test_pcp_summary_is_string():
    spot, strikes, mats, c_ivs, p_ivs = _make_pcp_data()
    result = PCPScanner().scan_from_data(spot, strikes, mats, c_ivs, p_ivs)
    s = result.summary()
    assert isinstance(s, str) and len(s) > 20
