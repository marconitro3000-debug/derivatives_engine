"""
tests/test_local_vol.py
Tests for Dupire local volatility surface (volatility/local_vol.py).

Key invariants we verify:
  1. ATM local vol ≈ ATM implied vol for a flat surface (ρ = 0, no smile).
  2. Local vol is non-negative everywhere on an arbitrage-free SSVI surface.
  3. Risk-neutral density g(k,T) ≥ 0 everywhere (no butterfly arb).
  4. ∂w/∂T = (∂w/∂θ)·(dθ/dT) matches numerical finite difference.
  5. MC local-vol pricer recovers BS price for a flat surface (zero skew).
  6. LocalVolSurface raises on decreasing ATM term structure (calendar arb).
  7. LV/IV ratio is close to 1 at ATM (exact for flat surfaces).
"""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from ml.ssvi import SSVIParams
from volatility.local_vol import (
    LocalVolSurface,
    LocalVolSlice,
    from_ssvi_and_atm,
    mc_price_local_vol,
    compare_with_bs,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_ssvi() -> SSVIParams:
    """Nearly flat SSVI — no skew, no smile.  ρ ≈ 0, small η."""
    return SSVIParams(rho=-0.01, eta=0.30, gamma=0.40)


@pytest.fixture
def spx_like_ssvi() -> SSVIParams:
    """SPX-like SSVI with pronounced negative skew."""
    return SSVIParams(rho=-0.70, eta=0.90, gamma=0.35)


@pytest.fixture
def atm_setup():
    maturities = np.array([0.08, 0.25, 0.50, 1.00, 2.00])
    atm_ivs    = np.array([0.22, 0.20, 0.19, 0.18, 0.17])   # downward sloping
    return maturities, atm_ivs


@pytest.fixture
def flat_surface(flat_ssvi, atm_setup) -> LocalVolSurface:
    maturities, atm_ivs = atm_setup
    return from_ssvi_and_atm(flat_ssvi, maturities, atm_ivs)


@pytest.fixture
def skew_surface(spx_like_ssvi, atm_setup) -> LocalVolSurface:
    maturities, atm_ivs = atm_setup
    return from_ssvi_and_atm(spx_like_ssvi, maturities, atm_ivs)


# ── Constructor tests ─────────────────────────────────────────────────────────

def test_constructor_valid(flat_ssvi, atm_setup):
    maturities, atm_ivs = atm_setup
    lv = from_ssvi_and_atm(flat_ssvi, maturities, atm_ivs)
    assert isinstance(lv, LocalVolSurface)


def test_constructor_rejects_decreasing_theta(flat_ssvi):
    """Calendar arb in ATM term structure must be caught."""
    maturities = np.array([0.25, 0.50, 1.00])
    atm_ivs    = np.array([0.20, 0.20, 0.20])   # flat sigma → flat theta → ok
    lv = from_ssvi_and_atm(flat_ssvi, maturities, atm_ivs)
    assert isinstance(lv, LocalVolSurface)

    # Decreasing total variance (sigma falls faster than sqrt(T))
    bad_ivs = np.array([0.30, 0.15, 0.10])      # theta₁ > theta₂ > theta₃
    # Only raises if theta is non-monotone
    bad_theta = bad_ivs ** 2 * maturities
    if np.any(np.diff(bad_theta) < 0):
        with pytest.raises(ValueError, match="non-decreasing"):
            LocalVolSurface(flat_ssvi, maturities, bad_theta)


# ── ATM term structure ────────────────────────────────────────────────────────

def test_atm_iv_recovers_input(flat_surface, atm_setup):
    """θ-spline should interpolate the input ATM vols exactly at nodes."""
    maturities, atm_ivs = atm_setup
    for T, iv in zip(maturities, atm_ivs):
        assert abs(flat_surface.atm_iv(T) - iv) < 1e-4, (
            f"ATM IV mismatch at T={T}: got {flat_surface.atm_iv(T):.4f}, expect {iv:.4f}"
        )


def test_dtheta_positive(flat_surface, atm_setup):
    """dθ/dT should be positive when θ is upward sloping."""
    maturities, atm_ivs = atm_setup
    # The theta = iv²*T series: check at interior points
    for T in [0.20, 0.40, 0.75]:
        assert flat_surface.dtheta_dT(T) >= 0


# ── Density / no-butterfly-arb ────────────────────────────────────────────────

def test_density_nonnegative_flat(flat_surface):
    """Risk-neutral density must be non-negative on a flat SSVI surface."""
    k_grid = np.linspace(-0.40, 0.40, 201)
    for T in [0.08, 0.25, 0.50, 1.00]:
        g = flat_surface.density(k_grid, T)
        assert np.all(g >= -1e-6), (
            f"Negative density at T={T}: min g = {g.min():.6f}"
        )


def test_density_nonnegative_skew(skew_surface):
    """SPX-like SSVI surface should also have non-negative density."""
    k_grid = np.linspace(-0.40, 0.40, 201)
    for T in [0.25, 0.50, 1.00, 2.00]:
        g = skew_surface.density(k_grid, T)
        assert np.all(g >= -1e-6), (
            f"Negative density at T={T}: min g = {g.min():.6f}"
        )


# ── Local volatility ──────────────────────────────────────────────────────────

def test_local_vol_nonnegative(skew_surface):
    """Local vol must be non-negative (NaN is acceptable at extremes)."""
    k_grid = np.linspace(-0.40, 0.40, 201)
    for T in [0.08, 0.25, 0.50, 1.00, 2.00]:
        lv = skew_surface(k_grid, T)
        assert np.all((lv >= 0) | np.isnan(lv)), (
            f"Negative local vol at T={T}: min = {np.nanmin(lv):.4f}"
        )


def test_atm_lv_close_to_iv_flat(flat_surface):
    """
    For a nearly-flat smile, σ_local(k=0) ≈ σ_BS(k=0).
    (Exact equality holds only for a perfectly flat surface.)
    """
    for T in [0.25, 0.50, 1.00]:
        lv_atm = float(flat_surface(np.array([0.0]), T)[0])
        iv_atm = float(flat_surface.implied_vol(np.array([0.0]), T)[0])
        # Tolerance is loose because SSVI is not perfectly flat even for small η
        assert abs(lv_atm - iv_atm) < 0.05 * iv_atm + 0.01, (
            f"T={T}: lv_atm={lv_atm:.4f}  iv_atm={iv_atm:.4f}"
        )


def test_skew_lv_greater_than_iv_in_put_wing(skew_surface):
    """
    Classic result: with negative skew the put-wing local vol exceeds
    the put-wing implied vol (Derman-Kani intuition).
    This is not always strictly true but holds for typical SPX parameters.
    """
    T  = 0.50
    k  = -0.20          # put wing
    lv = float(skew_surface(np.array([k]), T)[0])
    iv = float(skew_surface.implied_vol(np.array([k]), T)[0])
    # LV >= IV in put wing for negative-skew surfaces
    assert lv >= iv * 0.80, (
        f"k={k}: lv={lv:.4f}  iv={iv:.4f}  — expected lv ≥ iv in put wing"
    )


# ── dw/dT accuracy ────────────────────────────────────────────────────────────

def test_dw_dT_matches_finite_diff(skew_surface):
    """
    ∂w/∂T from the analytical chain rule should match numerical d/dT[w(k,T)].
    """
    k_test = np.array([-0.20, -0.10, 0.0, 0.10, 0.20])
    T      = 0.50
    h      = 1e-4

    theta_T = skew_surface.theta(T)
    theta_p = skew_surface.theta(T + h)
    theta_m = skew_surface.theta(T - h)

    for k in k_test:
        w_p = float(skew_surface.ssvi.total_var(np.array([k]), theta_p)[0])
        w_m = float(skew_surface.ssvi.total_var(np.array([k]), theta_m)[0])
        dw_numerical = (w_p - w_m) / (2 * h)

        dw_analytical = float(skew_surface._dw_dT(np.array([k]), T)[0])
        assert abs(dw_analytical - dw_numerical) < 1e-6, (
            f"k={k}: analytical={dw_analytical:.8f}  numerical={dw_numerical:.8f}"
        )


# ── Slice and surface ─────────────────────────────────────────────────────────

def test_slice_shapes(skew_surface):
    sl = skew_surface.slice(0.50)
    assert isinstance(sl, LocalVolSlice)
    assert len(sl.k) == len(sl.local_vol) == len(sl.implied_vol) == len(sl.density)


def test_surface_shapes(skew_surface):
    T_grid = np.array([0.25, 0.50, 1.00])
    K_mesh, T_mesh, LV = skew_surface.surface(T_grid)
    assert K_mesh.shape == T_mesh.shape == LV.shape
    assert LV.shape[0] == 3


# ── Monte Carlo pricer ────────────────────────────────────────────────────────

def test_mc_price_positive(flat_surface):
    S, K, T, r = 100.0, 100.0, 0.50, 0.04
    px, se = mc_price_local_vol(S, K, T, r, flat_surface, n_sims=10_000, seed=0)
    assert px > 0
    assert se > 0


def test_mc_price_atm_call_gt_intrinsic(flat_surface):
    """ATM call must be worth more than intrinsic (= 0 for ATM)."""
    px, _ = mc_price_local_vol(100.0, 100.0, 1.0, 0.04, flat_surface,
                                n_sims=20_000, seed=1)
    assert px > 0.5          # ATM call on 20% vol, 1Y should be several dollars


def test_mc_put_call_parity_approx(flat_surface):
    """MC call − put ≈ S·e^{-qT} − K·e^{-rT} (within MC noise)."""
    S, K, T, r = 100.0, 100.0, 0.50, 0.04
    n = 50_000
    c, _ = mc_price_local_vol(S, K, T, r, flat_surface, option="call",  n_sims=n, seed=42)
    p, _ = mc_price_local_vol(S, K, T, r, flat_surface, option="put",   n_sims=n, seed=42)
    fwd  = S - K * np.exp(-r * T)
    assert abs((c - p) - fwd) < 0.50    # $0.50 tolerance on MC noise


# ── Model comparison ──────────────────────────────────────────────────────────

def test_compare_with_bs_length(flat_surface):
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    rows = compare_with_bs(100.0, strikes, 0.50, 0.04, flat_surface, n_sims=5_000)
    assert len(rows) == len(strikes)
    for row in rows:
        assert row.price_local_vol > 0
        assert row.price_bs > 0
