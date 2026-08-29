"""SVI / SSVI calibration and the arbitrage properties each one does or does not have."""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.svi import SSVIParams, SSVISurface, SVIParams, SVISliceSurface, calibrate_svi
from volsurface.diagnostics import scan_arbitrage


# -- raw SVI ------------------------------------------------------------------

def test_svi_recovers_its_own_parameters():
    truth = SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.02, sigma=0.12)
    k = np.linspace(-0.5, 0.4, 40)
    fitted = calibrate_svi(k, truth.total_var(k))
    assert np.allclose(fitted.total_var(k), truth.total_var(k), atol=1e-6)


def test_svi_total_variance_is_convex_in_log_moneyness():
    p = SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.12)
    k = np.linspace(-0.6, 0.6, 200)
    assert np.all(np.diff(p.total_var(k), 2) > -1e-12)


def test_lee_wing_bound_rejects_an_exploding_smile():
    assert not SVIParams(a=0.02, b=5.0, rho=-0.4, m=0.0, sigma=0.12).is_butterfly_free()
    assert SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.12).is_butterfly_free()


# -- SSVI ---------------------------------------------------------------------

def test_ssvi_theta_is_monotone_even_from_non_monotone_nodes():
    """The calendar-arbitrage proof needs a non-decreasing theta; enforce it."""
    surf = SSVISurface(
        SSVIParams(rho=-0.5, eta=1.0, gamma=0.4),
        T_nodes=np.array([0.1, 0.5, 1.0]),
        theta_nodes=np.array([0.004, 0.003, 0.020]),   # dips in the middle
    )
    T = np.linspace(0.01, 2.0, 200)
    assert np.all(np.diff(surf.theta(T)) >= -1e-12)


def test_ssvi_extrapolates_beyond_the_last_expiry_without_turning_over():
    surf = SSVISurface(
        SSVIParams(rho=-0.5, eta=1.0, gamma=0.4),
        T_nodes=np.array([0.1, 0.5, 1.0]),
        theta_nodes=np.array([0.004, 0.010, 0.020]),
    )
    T = np.linspace(1.0, 5.0, 100)
    assert np.all(np.diff(surf.theta(T)) >= 0)


def test_ssvi_fit_is_arbitrage_free(clean_chain):
    """The whole point of SSVI: no violations anywhere, by construction."""
    report = scan_arbitrage(SSVISurface.fit(clean_chain))
    assert report.is_arbitrage_free, str(report)


def test_ssvi_needs_two_expiries(clean_chain):
    one = clean_chain.slice_at(clean_chain.maturities[0])
    with pytest.raises(ValueError, match="at least two"):
        SSVISurface.fit(one)


# -- the trade-off the project is about ---------------------------------------

def test_per_slice_svi_fits_better_but_admits_arbitrage(noisy_chain):
    """The motivating result: independent slices win on RMSE and lose on soundness.

    Fitting each smile on its own has more parameters and no cross-maturity
    constraint, so it tracks the quotes more closely than SSVI -- and the
    interpolation between those independently fitted slices is exactly where
    calendar arbitrage appears.
    """
    from volsurface.diagnostics import fit_report

    svi = SVISliceSurface.fit(noisy_chain)
    ssvi = SSVISurface.fit(noisy_chain)

    assert fit_report(svi, noisy_chain).rmse_vol_bps < fit_report(ssvi, noisy_chain).rmse_vol_bps

    svi_arb = scan_arbitrage(svi)
    ssvi_arb = scan_arbitrage(ssvi)
    assert not svi_arb.is_arbitrage_free, "expected the naive construction to break"
    assert ssvi_arb.is_arbitrage_free, str(ssvi_arb)


def test_single_slice_surface_scales_total_variance_to_zero_at_expiry():
    """One slice is still a surface: w must vanish as T -> 0, not stay flat."""
    surf = SVISliceSurface({0.5: SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.12)})
    k = np.zeros(3)
    w = surf.total_variance(k, np.array([0.001, 0.25, 0.5]))
    assert w[0] < w[1] < w[2]
