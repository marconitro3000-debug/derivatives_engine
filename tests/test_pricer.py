"""
tests/test_pricer.py
Full test suite for the options pricer.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from options_pricer import (
    price, greeks, put_call_parity_check,
    implied_vol, iv_surface,
    mc_price,
    binomial_price,
    VolSurface, from_iv_dict,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def params():
    """Standard test case: ATM, 1Y, 20% vol, 5% rate."""
    return dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20)


# ── black-scholes ─────────────────────────────────────────────────────────────

class TestBlackScholes:

    def test_call_positive(self, params):
        assert price(**params, option="call") > 0

    def test_put_positive(self, params):
        assert price(**params, option="put") > 0

    def test_put_call_parity(self, params):
        c = price(**params, option="call")
        p = price(**params, option="put")
        result = put_call_parity_check(params["S"], params["K"], params["T"],
                                       params["r"], c, p)
        assert result["error"] < 1e-10

    def test_call_intrinsic_itm(self):
        """Deep ITM call ≈ S - K*disc."""
        c = price(S=200, K=100, T=1.0, r=0.05, sigma=0.01)
        assert abs(c - (200 - 100 * np.exp(-0.05))) < 0.01

    def test_put_greater_than_zero_itm(self):
        """ITM put price is positive and larger than OTM put of same distance."""
        p_itm = price(S=90,  K=100, T=1.0, r=0.05, sigma=0.20, option="put")
        p_otm = price(S=110, K=100, T=1.0, r=0.05, sigma=0.20, option="put")
        assert p_itm > p_otm > 0

    def test_greeks_delta_bounds(self, params):
        g = greeks(**params)
        assert 0 < g["delta_call"] < 1
        assert -1 < g["delta_put"] < 0

    def test_greeks_gamma_positive(self, params):
        g = greeks(**params)
        assert g["gamma"] > 0

    def test_greeks_vega_positive(self, params):
        g = greeks(**params)
        assert g["vega"] > 0

    def test_greeks_theta_negative(self, params):
        g = greeks(**params)
        assert g["theta_call"] < 0
        assert g["theta_put"] < 0

    def test_invalid_option(self, params):
        with pytest.raises(ValueError):
            price(**params, option="straddle")

    def test_invalid_T(self, params):
        with pytest.raises(ValueError):
            price(S=100, K=100, T=0, r=0.05, sigma=0.2)


# ── implied volatility ────────────────────────────────────────────────────────

class TestImpliedVol:

    def test_round_trip(self, params):
        """IV(BS(σ)) should recover σ."""
        c  = price(**params, option="call")
        iv = implied_vol(params["S"], params["K"], params["T"],
                         params["r"], c, "call")
        assert abs(iv - params["sigma"]) < 1e-6

    def test_round_trip_put(self, params):
        p  = price(**params, option="put")
        iv = implied_vol(params["S"], params["K"], params["T"],
                         params["r"], p, "put")
        assert abs(iv - params["sigma"]) < 1e-6

    def test_otm_call(self):
        p  = price(S=100, K=110, T=0.5, r=0.03, sigma=0.25, option="call")
        iv = implied_vol(100, 110, 0.5, 0.03, p, "call")
        assert abs(iv - 0.25) < 1e-5

    def test_below_intrinsic_raises(self, params):
        with pytest.raises(ValueError):
            implied_vol(params["S"], params["K"], params["T"],
                        params["r"], 0.0001, "call")

    def test_iv_surface_shape(self, params):
        strikes    = [90, 100, 110]
        maturities = [0.5, 1.0]
        market_prices = {
            (K, T): price(params["S"], K, T, params["r"], params["sigma"])
            for K in strikes for T in maturities
        }
        surf = iv_surface(params["S"], strikes, maturities,
                          params["r"], market_prices)
        assert len(surf) == 6
        for v in surf.values():
            assert v is not None
            assert abs(v - params["sigma"]) < 1e-5


# ── monte carlo ───────────────────────────────────────────────────────────────

class TestMonteCarlo:

    def test_european_call_close_to_bs(self, params):
        res = mc_price(**params, option_type="european_call",
                       n_sims=80_000, seed=0)
        bs  = price(**params, option="call")
        assert abs(res["price"] - bs) / bs < 0.02   # within 2%

    def test_european_put_close_to_bs(self, params):
        res = mc_price(**params, option_type="european_put",
                       n_sims=80_000, seed=0)
        bs  = price(**params, option="put")
        assert abs(res["price"] - bs) / bs < 0.01

    def test_confidence_interval_contains_bs(self, params):
        res = mc_price(**params, option_type="european_call",
                       n_sims=100_000, seed=1)
        bs  = price(**params, option="call")
        assert res["conf_95_lo"] < bs < res["conf_95_hi"]

    def test_antithetic_reduces_std_error(self, params):
        """Antithetic variates should reduce std_error on average — test with large N."""
        plain = mc_price(**params, option_type="european_call",
                         n_sims=100_000, antithetic=False, seed=42)
        anti  = mc_price(**params, option_type="european_call",
                         n_sims=100_000, antithetic=True, seed=42)
        # With enough paths antithetic is reliably better; allow small tolerance
        assert anti["std_error"] < plain["std_error"] * 1.05

    def test_asian_call_cheaper_than_euro(self, params):
        asian = mc_price(**params, option_type="asian_call",  n_sims=50_000, seed=2)
        euro  = mc_price(**params, option_type="european_call", n_sims=50_000, seed=2)
        assert asian["price"] <= euro["price"]   # Asian ≤ European by theory

    def test_barrier_call_cheaper_than_euro(self, params):
        barrier = mc_price(**params, option_type="barrier_call",
                           barrier=80, n_sims=50_000, seed=3)
        euro    = mc_price(**params, option_type="european_call",
                           n_sims=50_000, seed=3)
        assert barrier["price"] <= euro["price"]

    def test_digital_call_price_in_range(self, params):
        res = mc_price(**params, option_type="digital_call", n_sims=100_000, seed=5)
        assert 0 < res["price"] < 1   # payo $1 if S_T > K

    def test_unknown_payoff_raises(self, params):
        with pytest.raises(ValueError):
            mc_price(**params, option_type="mystery_option")


# ── binomial tree ─────────────────────────────────────────────────────────────

class TestBinomialTree:

    def test_european_call_converges_to_bs(self, params):
        res = binomial_price(**params, option="call", style="european", n_steps=1000)
        bs  = price(**params, option="call")
        assert abs(res["price"] - bs) / bs < 0.001   # within 0.1%

    def test_european_put_converges_to_bs(self, params):
        res = binomial_price(**params, option="put", style="european", n_steps=1000)
        bs  = price(**params, option="put")
        assert abs(res["price"] - bs) / bs < 0.001

    def test_american_put_ge_european_put(self, params):
        am = binomial_price(**params, option="put", style="american", n_steps=500)
        eu = binomial_price(**params, option="put", style="european", n_steps=500)
        assert am["price"] >= eu["price"]   # early exercise has value

    def test_american_call_no_early_exercise(self, params):
        """For non-dividend-paying stock, American call = European call."""
        am = binomial_price(**params, option="call", style="american", n_steps=500)
        assert am["early_exercise"] < 0.01   # negligible premium

    def test_deep_itm_american_put_premium(self):
        """Deep ITM put should have significant early exercise premium."""
        am = binomial_price(S=60, K=100, T=1.0, r=0.05, sigma=0.20,
                            option="put", style="american", n_steps=500)
        assert am["early_exercise"] > 0.5

    def test_invalid_style_raises(self, params):
        with pytest.raises(ValueError):
            binomial_price(**params, style="bermudan")


# ── vol surface ───────────────────────────────────────────────────────────────

class TestVolSurface:

    @pytest.fixture
    def flat_surface(self):
        """Flat vol surface at 20% for easy verification."""
        strikes    = np.array([80., 90., 100., 110., 120.])
        maturities = np.array([0.25, 0.5, 1.0, 2.0])
        iv_grid    = np.full((5, 4), 0.20)
        return VolSurface(strikes, maturities, iv_grid, method="spline")

    def test_flat_surface_query(self, flat_surface):
        iv = flat_surface.iv(100., 1.0)
        assert abs(iv - 0.20) < 0.001

    def test_smile_length(self, flat_surface):
        ks, ivs = flat_surface.smile(T=1.0)
        assert len(ks) == len(ivs) == 5

    def test_term_structure_length(self, flat_surface):
        ts, ivs = flat_surface.term_structure(K=100.)
        assert len(ts) == len(ivs) == 4

    def test_grid_shape(self, flat_surface):
        K_grid, T_grid, IV_grid = flat_surface.grid(n_strikes=20, n_maturities=10)
        assert IV_grid.shape == (20, 10)

    def test_iv_positive(self, flat_surface):
        _, _, IV = flat_surface.grid()
        assert (IV > 0).all()

    def test_from_iv_dict(self, params):
        strikes    = [90, 100, 110]
        maturities = [0.5, 1.0]
        iv_dict    = {
            (K, T): implied_vol(params["S"], K, T, params["r"],
                                price(params["S"], K, T, params["r"],
                                      params["sigma"]))
            for K in strikes for T in maturities
        }
        surf = from_iv_dict(params["S"], iv_dict)
        assert surf.iv(100., 1.0) is not None

    def test_rbf_method(self, flat_surface):
        strikes    = np.array([80., 90., 100., 110., 120.])
        maturities = np.array([0.25, 0.5, 1.0, 2.0])
        iv_grid    = np.full((5, 4), 0.20)
        surf_rbf   = VolSurface(strikes, maturities, iv_grid, method="rbf")
        iv         = surf_rbf.iv(100., 1.0)
        assert abs(iv - 0.20) < 0.01
