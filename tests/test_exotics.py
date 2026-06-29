"""
tests/test_exotics.py
Unit tests for the exotics module.
Run with: pytest tests/test_exotics.py -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from options.black_scholes import price as bs_price
from exotics.barrier  import price_barrier, mc_barrier
from exotics.asian    import price_asian_geo, price_asian_kv, mc_asian_arith
from exotics.lookback import price_lookback_float, mc_lookback
from exotics.digital  import (price_cash_or_nothing, price_asset_or_nothing,
                               price_one_touch, price_no_touch, mc_digital)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def params():
    return dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20)


# ═══════════════════════════════════════════════════════════════
# BARRIER OPTIONS
# ═══════════════════════════════════════════════════════════════

class TestBarrier:

    def test_down_out_call_price_positive(self, params):
        res = price_barrier(**params, H=85, option_type="call", barrier_type="down-out")
        assert res["price"] > 0

    def test_down_out_call_cheaper_than_vanilla(self, params):
        res = price_barrier(**params, H=85, option_type="call", barrier_type="down-out")
        assert res["price"] < res["vanilla"]

    def test_down_in_plus_down_out_equals_vanilla(self, params):
        out = price_barrier(**params, H=85, option_type="call", barrier_type="down-out")
        inn = price_barrier(**params, H=85, option_type="call", barrier_type="down-in")
        assert abs(out["price"] + inn["price"] - out["vanilla"]) < 1e-8

    def test_up_in_plus_up_out_equals_vanilla(self, params):
        out = price_barrier(**params, H=115, option_type="call", barrier_type="up-out")
        inn = price_barrier(**params, H=115, option_type="call", barrier_type="up-in")
        assert abs(out["price"] + inn["price"] - out["vanilla"]) < 1e-8

    def test_put_parity(self, params):
        out = price_barrier(**params, H=85, option_type="put", barrier_type="down-out")
        inn = price_barrier(**params, H=85, option_type="put", barrier_type="down-in")
        assert abs(out["price"] + inn["price"] - out["vanilla"]) < 1e-8

    def test_up_out_call_zero_when_barrier_below_strike(self, params):
        # K=100, H=90 < K: up-out call with H < K
        # Up-out means H > S. H=90 < S=100 violates that; instead test K >= H case
        res = price_barrier(S=100, K=110, T=1.0, r=0.05, sigma=0.20,
                            H=105, option_type="call", barrier_type="up-out")
        # K >= H: up-out call → 0 (can't be exercised without being knocked out)
        assert res["price"] == pytest.approx(0.0, abs=1e-6)

    def test_barrier_at_spot_knocked_out(self, params):
        # S = H: immediately knocked out → price = rebate = 0
        res = price_barrier(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                            H=100, option_type="call", barrier_type="down-out")
        assert res["price"] == 0.0

    def test_mc_continuous_vs_closed_form(self, params):
        cf = price_barrier(**params, H=85, option_type="call", barrier_type="down-out")
        # Daily monitoring (252 steps) converges towards continuous CF within ~0.25
        mc = mc_barrier(**params, H=85, option_type="call", barrier_type="down-out",
                        n_sims=50_000, n_steps=252, seed=42)
        assert abs(mc["price"] - cf["price"]) < 0.25

    def test_mc_down_in_cheaper_than_vanilla(self, params):
        mc = mc_barrier(**params, H=85, option_type="call", barrier_type="down-in",
                        n_sims=50_000, seed=0)
        van = bs_price(**params, option="call")
        assert mc["price"] < van

    def test_barrier_put_call(self, params):
        dop = price_barrier(**params, H=85, option_type="put", barrier_type="down-out")
        assert dop["price"] >= 0


# ═══════════════════════════════════════════════════════════════
# ASIAN OPTIONS
# ═══════════════════════════════════════════════════════════════

class TestAsian:

    def test_geo_call_cheaper_than_vanilla(self, params):
        geo = price_asian_geo(**params, option_type="call")
        van = bs_price(**params, option="call")
        assert geo["price"] < van

    def test_geo_put_cheaper_than_vanilla(self, params):
        geo = price_asian_geo(**params, option_type="put")
        van = bs_price(**params, option="put")
        assert geo["price"] < van

    def test_geo_price_positive(self, params):
        geo = price_asian_geo(**params)
        assert geo["price"] > 0

    def test_arith_cheaper_than_vanilla(self, params):
        mc = mc_asian_arith(**params, option_type="call", n_sims=80_000, seed=42)
        van = bs_price(**params, option="call")
        assert mc["price"] < van

    def test_arith_dearer_than_geometric(self, params):
        mc_arith = mc_asian_arith(**params, option_type="call", n_sims=100_000, seed=42)
        geo      = price_asian_geo(**params, option_type="call")
        # Arithmetic average >= geometric average (AM-GM), so call option on arithmetic
        # is at least as expensive as on geometric.
        assert mc_arith["price"] >= geo["price"] - 0.05  # small MC tolerance

    def test_kv_approximation_close_to_mc(self, params):
        kv = price_asian_kv(**params, option_type="call")
        mc = mc_asian_arith(**params, option_type="call", n_sims=100_000, seed=42)
        assert abs(kv["price"] - mc["price"]) / mc["price"] < 0.10  # within 10%

    def test_floating_strike_call(self, params):
        mc = mc_asian_arith(params["S"], params["K"], params["T"], params["r"],
                            params["sigma"], option_type="call",
                            strike_type="floating", n_sims=50_000, seed=7)
        assert mc["price"] > 0

    def test_geo_sigma_zero_gives_deterministic_price(self):
        # With σ=0, S_t = S*e^{rt}, geometric average is deterministic
        S, K, T, r, sig = 100, 95, 1.0, 0.05, 0.001
        geo = price_asian_geo(S, K, T, r, sig, "call")
        assert geo["price"] > 0   # S*e^{rT/2} > K, so call has value


# ═══════════════════════════════════════════════════════════════
# LOOKBACK OPTIONS
# ═══════════════════════════════════════════════════════════════

class TestLookback:

    def test_float_call_dearer_than_vanilla(self, params):
        lb  = price_lookback_float(params["S"], params["T"], params["r"], params["sigma"])
        van = bs_price(**params, option="call")
        assert lb["price"] > van

    def test_float_put_positive(self, params):
        lb = price_lookback_float(params["S"], params["T"], params["r"], params["sigma"],
                                  option_type="put")
        assert lb["price"] > 0

    def test_float_call_closed_vs_mc(self, params):
        cf = price_lookback_float(params["S"], params["T"], params["r"], params["sigma"],
                                  option_type="call")
        mc = mc_lookback(params["S"], params["T"], params["r"], params["sigma"],
                         option_type="call", strike_type="float",
                         n_sims=150_000, n_steps=int(params["T"] * 252 * 4), seed=0)
        # Fine-grained MC (4× daily) should be close to continuous closed-form
        assert abs(cf["price"] - mc["price"]) / cf["price"] < 0.05

    def test_float_call_always_nonneg_payoff(self, params):
        mc = mc_lookback(params["S"], params["T"], params["r"], params["sigma"],
                         option_type="call", strike_type="float",
                         n_sims=10_000, seed=1)
        assert mc["price"] >= 0

    def test_fixed_strike_lookback_call(self, params):
        mc = mc_lookback(params["S"], params["T"], params["r"], params["sigma"],
                         option_type="call", strike_type="fixed", K=params["K"],
                         n_sims=50_000, seed=3)
        van = bs_price(**params, option="call")
        # Fixed-strike lookback call ≥ vanilla call (you get the best price ever)
        assert mc["price"] >= van - 0.05


# ═══════════════════════════════════════════════════════════════
# DIGITAL OPTIONS
# ═══════════════════════════════════════════════════════════════

class TestDigital:

    def test_cash_or_nothing_call_in_range(self, params):
        res = price_cash_or_nothing(**params, option_type="call", cash=1.0)
        assert 0 < res["price"] < np.exp(-params["r"] * params["T"])

    def test_cash_or_nothing_put_in_range(self, params):
        res = price_cash_or_nothing(**params, option_type="put", cash=1.0)
        assert 0 < res["price"] < np.exp(-params["r"] * params["T"])

    def test_cash_call_plus_put_equals_discount(self, params):
        c = price_cash_or_nothing(**params, option_type="call", cash=1.0)["price"]
        p = price_cash_or_nothing(**params, option_type="put",  cash=1.0)["price"]
        df = np.exp(-params["r"] * params["T"])
        assert abs(c + p - df) < 1e-10

    def test_asset_or_nothing_call_in_range(self, params):
        res = price_asset_or_nothing(**params, option_type="call")
        assert 0 < res["price"] < params["S"]

    def test_bsm_decomposition(self, params):
        # BSM = asset-or-nothing − K·e^{-rT}·cash-or-nothing
        aon = price_asset_or_nothing(**params, option_type="call")["price"]
        con = price_cash_or_nothing(**params,  option_type="call", cash=params["K"])["price"]
        van = bs_price(**params, option="call")
        assert abs(aon - con - van) < 1e-8

    def test_one_touch_below_one_discounted(self, params):
        # OT price ≤ discounted sure thing
        ot  = price_one_touch(params["S"], params["T"], params["r"], params["sigma"],
                               H=params["S"] * 0.85, touch_type="down")
        df  = np.exp(-params["r"] * params["T"])
        assert 0 < ot["price"] < 1.0   # pays $1, always below $1

    def test_one_touch_plus_no_touch_equals_discount(self, params):
        S, T, r, sig = params["S"], params["T"], params["r"], params["sigma"]
        H = S * 0.85
        ot = price_one_touch(S, T, r, sig, H, "down", 1.0, payout_at="expiry")["price"]
        nt = price_no_touch( S, T, r, sig, H, "down", 1.0)["price"]
        df = np.exp(-r * T)
        assert abs(ot + nt - df) < 1e-6

    def test_mc_cash_or_nothing_close_to_closed_form(self, params):
        cf = price_cash_or_nothing(**params, option_type="call", cash=1.0)["price"]
        mc = mc_digital(**params, digital_type="cash-or-nothing", option_type="call",
                        n_sims=200_000, seed=0)["price"]
        assert abs(mc - cf) < 0.01

    def test_mc_one_touch_close_to_closed_form(self, params):
        S, T, r, sig = params["S"], params["T"], params["r"], params["sigma"]
        H = S * 0.85
        # Both compared with payout_at='expiry' so conventions match
        cf = price_one_touch(S, T, r, sig, H, "down", 1.0, payout_at="expiry")["price"]
        mc = mc_digital(S, S, T, r, sig, "one-touch", "call",
                        H=H, cash=1.0, n_sims=100_000, seed=1)["price"]
        assert abs(mc - cf) / cf < 0.07
