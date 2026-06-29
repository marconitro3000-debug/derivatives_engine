"""
tests/test_rates_advanced.py
Unit tests for rates/swaption.py, rates/capfloor.py, rates/sabr.py,
and rates/nelson_siegel.py.

Coverage:
  TestSwaption    (12 tests)
  TestCapFloor    (13 tests)
  TestSABR        (10 tests)
  TestNelsonSiegel(10 tests)
"""

import numpy as np
import pytest

from rates import DiscountCurve
from rates.swaption import (
    forward_swap_rate, annuity,
    price_swaption_black, price_swaption_bachelier, implied_black_vol,
)
from rates.capfloor import (
    forward_libor, caplet, floorlet, cap, floor, collar,
    cap_floor_parity_check, implied_cap_vol,
)
from rates.sabr import (
    SABRParams, implied_vol_sabr, implied_vol_grid, calibrate, atm_vol,
)
from rates.nelson_siegel import (
    NSParams, SvenssonParams,
    ns_yield, svensson_yield,
    ns_discount_factor, svensson_discount_factor,
    fit_ns, fit_svensson, fit_summary,
)


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def flat_dc():
    return DiscountCurve.flat(0.05, 20.0)


@pytest.fixture(scope="module")
def upward_dc():
    # Simple upward-sloping curve
    times = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]
    rates = [0.040, 0.042, 0.045, 0.048, 0.050, 0.053, 0.055, 0.057, 0.059, 0.060]
    return DiscountCurve.from_zero_rates(times, rates)


# ── TestSwaption ──────────────────────────────────────────────────────────────

class TestSwaption:

    def test_forward_swap_rate_positive(self, flat_dc):
        F, A = forward_swap_rate(flat_dc, T_exp=1.0, tenor=5.0)
        assert F > 0
        assert A > 0

    def test_annuity_positive(self, flat_dc):
        A = annuity(flat_dc, T_exp=1.0, tenor=5.0)
        assert A > 0

    def test_payer_positive(self, flat_dc):
        res = price_swaption_black(flat_dc, T_exp=1.0, tenor=5.0,
                                    strike=0.05, sigma=0.20)
        assert res["price"] > 0

    def test_receiver_positive(self, flat_dc):
        res = price_swaption_black(flat_dc, T_exp=1.0, tenor=5.0,
                                    strike=0.05, sigma=0.20,
                                    swaption_type="receiver")
        assert res["price"] > 0

    def test_put_call_parity(self, flat_dc):
        """Payer - Receiver = Fwd swap value = A × (F - K)"""
        T_exp, tenor, sigma = 1.0, 5.0, 0.20
        K = 0.05
        F, A = forward_swap_rate(flat_dc, T_exp, tenor)
        payer    = price_swaption_black(flat_dc, T_exp, tenor, K, sigma,
                                         "payer")["price"]
        receiver = price_swaption_black(flat_dc, T_exp, tenor, K, sigma,
                                         "receiver")["price"]
        notional = 1_000_000.0
        swap_fv  = notional * A * (F - K)
        assert abs((payer - receiver) - swap_fv) < 1.0

    def test_atm_payer_receiver_equal(self, flat_dc):
        """At K=F (ATM), payer and receiver have the same price."""
        F, _ = forward_swap_rate(flat_dc, 1.0, 5.0)
        p = price_swaption_black(flat_dc, 1.0, 5.0, F, 0.20, "payer")["price"]
        r = price_swaption_black(flat_dc, 1.0, 5.0, F, 0.20, "receiver")["price"]
        assert abs(p - r) < 1.0

    def test_higher_vol_higher_price(self, flat_dc):
        lo = price_swaption_black(flat_dc, 1.0, 5.0, 0.05, 0.10)["price"]
        hi = price_swaption_black(flat_dc, 1.0, 5.0, 0.05, 0.30)["price"]
        assert hi > lo

    def test_result_keys(self, flat_dc):
        res = price_swaption_black(flat_dc, 1.0, 5.0, 0.05, 0.20)
        for k in ("price", "forward_swap_rate", "annuity", "d1", "d2"):
            assert k in res

    def test_bachelier_positive(self, flat_dc):
        res = price_swaption_bachelier(flat_dc, 1.0, 5.0,
                                        strike=0.05, sigma_n=0.005)
        assert res["price"] > 0

    def test_bachelier_payer_receiver_parity(self, flat_dc):
        F, A = forward_swap_rate(flat_dc, 1.0, 5.0)
        K    = 0.05
        p    = price_swaption_bachelier(flat_dc, 1.0, 5.0, K, 0.005, "payer")["price"]
        r    = price_swaption_bachelier(flat_dc, 1.0, 5.0, K, 0.005, "receiver")["price"]
        swap = 1_000_000 * A * (F - K)
        assert abs((p - r) - swap) < 1.0

    def test_implied_vol_round_trip(self, flat_dc):
        sigma = 0.20
        px    = price_swaption_black(flat_dc, 1.0, 5.0, 0.05, sigma)["price"]
        iv    = implied_black_vol(px, flat_dc, 1.0, 5.0, 0.05)
        assert abs(iv - sigma) < 1e-5

    def test_deep_itm_payer_close_to_swap(self, flat_dc):
        """Deep ITM payer swaption ≈ swap value (low vol → intrinsic)."""
        F, A = forward_swap_rate(flat_dc, 1.0, 5.0)
        K    = F * 0.5        # very low strike (ITM payer)
        p    = price_swaption_black(flat_dc, 1.0, 5.0, K, 0.01)["price"]
        swap = 1_000_000 * A * (F - K)
        assert abs(p - swap) / swap < 0.01


# ── TestCapFloor ──────────────────────────────────────────────────────────────

class TestCapFloor:

    def test_forward_libor_positive(self, flat_dc):
        F = forward_libor(flat_dc, 1.0, 1.25)
        assert F > 0

    def test_caplet_positive(self, flat_dc):
        c = caplet(flat_dc, 1.0, 1.25, strike=0.05, sigma=0.20)
        assert c > 0

    def test_floorlet_positive(self, flat_dc):
        f = floorlet(flat_dc, 1.0, 1.25, strike=0.05, sigma=0.20)
        assert f > 0

    def test_caplet_floorlet_parity(self, flat_dc):
        """Caplet − Floorlet = Forward rate leg − Fixed leg (discounted)"""
        t_fix, t_pay = 1.0, 1.25
        K, sigma = 0.05, 0.20
        dt = t_pay - t_fix
        F  = forward_libor(flat_dc, t_fix, t_pay)
        P  = flat_dc.discount_factor(t_pay)
        N  = 1_000_000.0
        c  = caplet(flat_dc, t_fix, t_pay, K, sigma, N)
        fl = floorlet(flat_dc, t_fix, t_pay, K, sigma, N)
        diff = c - fl
        fwd_leg = N * P * dt * (F - K)
        assert abs(diff - fwd_leg) < 1.0

    def test_cap_positive(self, flat_dc):
        res = cap(flat_dc, maturity=3.0, strike=0.05, sigma=0.20)
        assert res["price"] > 0

    def test_floor_positive(self, flat_dc):
        res = floor(flat_dc, maturity=3.0, strike=0.05, sigma=0.20)
        assert res["price"] > 0

    def test_cap_floor_parity(self, flat_dc):
        """Cap − Floor = floating − fixed = swap PV."""
        res = cap_floor_parity_check(flat_dc, maturity=3.0, strike=0.05, sigma=0.20)
        assert res["parity_error"] < 5.0   # within $5 on $1M notional

    def test_cap_increases_with_vol(self, flat_dc):
        lo = cap(flat_dc, 3.0, 0.05, 0.10)["price"]
        hi = cap(flat_dc, 3.0, 0.05, 0.30)["price"]
        assert hi > lo

    def test_collar_net_cost(self, flat_dc):
        res = collar(flat_dc, maturity=3.0,
                     cap_strike=0.06, floor_strike=0.04,
                     sigma_cap=0.20, sigma_floor=0.20)
        # Long cap, short floor at lower strike → net cost typically positive
        assert "net_cost" in res
        assert "cap_price" in res

    def test_cap_schedule_length(self, flat_dc):
        res = cap(flat_dc, maturity=2.0, strike=0.05, sigma=0.20, pay_freq=4)
        # 2y × 4/year = 8 caplets
        assert res["n_caplets"] == 8

    def test_implied_cap_vol_round_trip(self, flat_dc):
        sigma  = 0.20
        market = cap(flat_dc, 3.0, 0.05, sigma)["price"]
        iv     = implied_cap_vol(market, flat_dc, 3.0, 0.05)
        assert abs(iv - sigma) < 1e-5

    def test_atm_caplet_vs_floorlet(self, flat_dc):
        """At ATM strike (=forward), caplet and floorlet are approx equal."""
        t_fix, t_pay = 2.0, 2.25
        F = forward_libor(flat_dc, t_fix, t_pay)
        c = caplet(flat_dc, t_fix, t_pay, F, 0.20)
        f = floorlet(flat_dc, t_fix, t_pay, F, 0.20)
        # Parity: c - f = P*dt*(F-F) = 0, so c ≈ f
        assert abs(c - f) < 1.0


# ── TestSABR ─────────────────────────────────────────────────────────────────

class TestSABR:

    @pytest.fixture
    def params(self):
        return SABRParams(alpha=0.04, beta=0.5, rho=-0.30, nu=0.40)

    def test_atm_vol_positive(self, params):
        v = atm_vol(F=0.05, T=1.0, params=params)
        assert v > 0

    def test_implied_vol_positive(self, params):
        v = implied_vol_sabr(F=0.05, K=0.05, T=1.0, params=params)
        assert v > 0

    def test_vol_near_atm_continuous(self, params):
        """Vol should be nearly continuous across K=F."""
        F = 0.05
        v_atm  = implied_vol_sabr(F, F, 1.0, params)
        v_near = implied_vol_sabr(F, F + 1e-6, 1.0, params)
        assert abs(v_atm - v_near) < 1e-4

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            SABRParams(alpha=0.04, beta=1.5, rho=-0.30, nu=0.40)

    def test_invalid_rho_raises(self):
        with pytest.raises(ValueError):
            SABRParams(alpha=0.04, beta=0.5, rho=1.5, nu=0.40)

    def test_smile_shape_negative_rho(self, params):
        """Negative rho → downward-sloping smile (OTM puts more expensive)."""
        F  = 0.05
        v_otm_put  = implied_vol_sabr(F, 0.03, 1.0, params)   # OTM put
        v_otm_call = implied_vol_sabr(F, 0.07, 1.0, params)   # OTM call
        assert v_otm_put > v_otm_call

    def test_vol_grid_length(self, params):
        strikes = np.linspace(0.02, 0.09, 10)
        vols    = implied_vol_grid(0.05, strikes, 1.0, params)
        assert len(vols) == 10

    def test_calibration_reduces_error(self):
        """Calibrated SABR should fit market vols better than random params."""
        F       = 0.05
        T       = 1.0
        true_p  = SABRParams(alpha=0.04, beta=0.5, rho=-0.30, nu=0.40)
        strikes = np.linspace(0.02, 0.09, 8)
        mkt_v   = implied_vol_grid(F, strikes, T, true_p)

        # Calibrate back
        fitted  = calibrate(F, T, strikes, mkt_v, beta=0.5)
        fitted_v= implied_vol_grid(F, strikes, T, fitted)
        rmse    = np.sqrt(np.mean((fitted_v - mkt_v) ** 2))
        assert rmse < 1e-4   # sub-bp accuracy

    def test_vol_increases_with_nu(self):
        """Higher vol-of-vol → wider smile (more curvature)."""
        p_lo = SABRParams(alpha=0.04, beta=0.5, rho=0.0, nu=0.10)
        p_hi = SABRParams(alpha=0.04, beta=0.5, rho=0.0, nu=0.80)
        # Wing vols should be higher for high ν
        F = 0.05
        v_wing_lo = implied_vol_sabr(F, 0.03, 1.0, p_lo)
        v_wing_hi = implied_vol_sabr(F, 0.03, 1.0, p_hi)
        assert v_wing_hi > v_wing_lo

    def test_beta_one_log_normal(self):
        """β=1 recovers log-normal (flat smile for ρ=0, ν=0)."""
        p = SABRParams(alpha=0.04, beta=1.0, rho=0.0, nu=1e-8)
        F = 0.05
        for K in [0.03, 0.04, 0.05, 0.06, 0.07]:
            v = implied_vol_sabr(F, K, 1.0, p)
            # Should be approximately alpha / F^{1-beta=0} = alpha / 1 = alpha
            assert abs(v - p.alpha) < 0.005


# ── TestNelsonSiegel ──────────────────────────────────────────────────────────

class TestNelsonSiegel:

    @pytest.fixture
    def ns_params(self):
        return NSParams(beta0=0.060, beta1=-0.020, beta2=0.010, lam=2.5)

    @pytest.fixture
    def market_data(self):
        maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
        zero_rates = np.array([0.042, 0.043, 0.045, 0.048, 0.050,
                                0.053, 0.055, 0.057])
        return maturities, zero_rates

    def test_ns_yield_shape(self, ns_params):
        T = np.array([1.0, 2.0, 5.0, 10.0])
        r = ns_yield(T, ns_params)
        assert r.shape == (4,)

    def test_ns_long_rate(self, ns_params):
        """NS yield approaches β₀ as t→∞"""
        r_long = ns_yield(np.array([100.0]), ns_params)[0]
        assert abs(r_long - ns_params.beta0) < 0.001

    def test_ns_short_rate(self, ns_params):
        """NS yield approaches β₀+β₁ as t→0"""
        r_short = ns_yield(np.array([0.001]), ns_params)[0]
        assert abs(r_short - ns_params.short_rate) < 0.001

    def test_ns_discount_factor_lt_1(self, ns_params):
        df = ns_discount_factor(np.array([1.0, 5.0, 10.0]), ns_params)
        assert (df < 1.0).all()
        assert (df > 0.0).all()

    def test_fit_ns_converges(self, market_data):
        T, r = market_data
        fitted = fit_ns(T, r)
        # Long rate should be in ballpark of highest observed rate
        assert 0.0 < fitted.beta0 < 0.15

    def test_fit_ns_accuracy(self, market_data):
        """NS fit should have RMSE < 5 bps."""
        T, r = market_data
        fitted    = fit_ns(T, r)
        r_fit     = ns_yield(T, fitted)
        rmse_bps  = np.sqrt(np.mean((r_fit - r) ** 2)) * 10_000
        assert rmse_bps < 5.0

    def test_fit_svensson_accuracy(self, market_data):
        """Svensson has more dof → better fit than NS."""
        T, r         = market_data
        ns_fitted    = fit_ns(T, r)
        sv_fitted    = fit_svensson(T, r)
        ns_rmse      = np.sqrt(np.mean((ns_yield(T, ns_fitted) - r) ** 2))
        sv_rmse      = np.sqrt(np.mean((svensson_yield(T, sv_fitted) - r) ** 2))
        assert sv_rmse <= ns_rmse + 1e-5   # Svensson should do at least as well

    def test_svensson_long_rate(self, market_data):
        T, r   = market_data
        fitted = fit_svensson(T, r)
        # At t=1000 all exponential loadings → 0, so yield → β₀
        r_long = svensson_yield(np.array([1000.0]), fitted)[0]
        assert abs(r_long - fitted.beta0) < 0.001

    def test_fit_summary_string(self, market_data):
        T, r   = market_data
        fitted = fit_ns(T, r)
        s      = fit_summary(T, r, fitted, model="NS")
        assert "Nelson-Siegel" in s
        assert "RMSE" in s

    def test_ns_on_flat_curve(self):
        """NS fitted to a flat curve should return β₁≈0, β₂≈0."""
        T = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
        r = np.full_like(T, 0.05)
        fitted = fit_ns(T, r)
        assert abs(fitted.beta0 - 0.05) < 0.001
        assert abs(fitted.beta1) < 0.001
        assert abs(fitted.beta2) < 0.005
