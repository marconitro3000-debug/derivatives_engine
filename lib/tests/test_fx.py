"""
tests/test_fx.py
Unit tests for fx/ — Garman-Kohlhagen and FX smile.
"""

import numpy as np
import pytest

from fx import (
    fx_forward, price_gk, implied_vol_gk, put_call_parity_check,
    delta_to_strike, strike_to_delta,
    atm_dns_strike, atm_forward_strike,
    FXSmileQuotes, build_smile,
    vanna_volga_price,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def eurusd():
    return dict(S=1.08, r_d=0.05, r_f=0.03, T=1.0, sigma=0.08)


@pytest.fixture
def smile_quotes():
    return FXSmileQuotes(
        S=1.08, T=1.0, r_d=0.05, r_f=0.03,
        atm=0.080, rr25=0.010, bf25=0.003,
    )


# ── Garman-Kohlhagen ──────────────────────────────────────────────────────────

class TestGarmanKohlhagen:

    def test_forward_price(self, eurusd):
        F = fx_forward(**{k: eurusd[k] for k in ['S','r_d','r_f','T']})
        # F = 1.08 * exp((0.05-0.03)*1) ≈ 1.1017
        assert abs(F - 1.08 * np.exp(0.02)) < 1e-8

    def test_call_positive(self, eurusd):
        res = price_gk(**eurusd, K=1.08, option_type="call")
        assert res["price"] > 0

    def test_put_positive(self, eurusd):
        res = price_gk(**eurusd, K=1.08, option_type="put")
        assert res["price"] > 0

    def test_put_call_parity(self, eurusd):
        res = put_call_parity_check(**eurusd, K=1.08)
        assert res["error"] < 1e-10

    def test_call_delta_in_01(self, eurusd):
        res = price_gk(**eurusd, K=1.08, option_type="call")
        assert 0 < res["delta"] < 1

    def test_put_delta_in_m1_0(self, eurusd):
        res = price_gk(**eurusd, K=1.08, option_type="put")
        assert -1 < res["delta"] < 0

    def test_gamma_positive(self, eurusd):
        res = price_gk(**eurusd, K=1.08, option_type="call")
        assert res["gamma"] > 0

    def test_vega_positive(self, eurusd):
        res = price_gk(**eurusd, K=1.08, option_type="call")
        assert res["vega"] > 0

    def test_call_monotone_in_spot(self, eurusd):
        lo = price_gk(**{**eurusd, 'S': 1.00}, K=1.08, option_type="call")["price"]
        hi = price_gk(**{**eurusd, 'S': 1.20}, K=1.08, option_type="call")["price"]
        assert hi > lo

    def test_higher_vol_higher_price(self, eurusd):
        lo = price_gk(**{**eurusd, 'sigma': 0.05}, K=1.08, option_type="call")["price"]
        hi = price_gk(**{**eurusd, 'sigma': 0.20}, K=1.08, option_type="call")["price"]
        assert hi > lo

    def test_implied_vol_round_trip(self, eurusd):
        sigma = eurusd["sigma"]
        px = price_gk(**eurusd, K=1.08, option_type="call")["price"]
        iv = implied_vol_gk(px, eurusd["S"], 1.08, eurusd["T"],
                             eurusd["r_d"], eurusd["r_f"], "call")
        assert abs(iv - sigma) < 1e-5

    def test_result_has_vanna_volga(self, eurusd):
        res = price_gk(**eurusd, K=1.08, option_type="call")
        assert "vanna" in res and "volga" in res

    def test_deep_itm_call_near_forward(self, eurusd):
        """Deep ITM call ≈ (F - K) × e^{-r_d T}."""
        F  = fx_forward(**{k: eurusd[k] for k in ['S','r_d','r_f','T']})
        K  = 0.80   # very low strike
        px = price_gk(**eurusd, K=K, option_type="call", notional=1.0)["unit_px"]
        intrinsic = (F - K) * np.exp(-eurusd["r_d"] * eurusd["T"])
        assert abs(px - intrinsic) / intrinsic < 0.02


# ── Delta / Strike ────────────────────────────────────────────────────────────

class TestDeltaStrike:

    def test_delta_to_strike_call(self, eurusd):
        K = delta_to_strike(0.25, eurusd["S"], eurusd["T"],
                             eurusd["r_d"], eurusd["r_f"], eurusd["sigma"])
        assert K > 0

    def test_delta_to_strike_put(self, eurusd):
        K = delta_to_strike(0.25, eurusd["S"], eurusd["T"],
                             eurusd["r_d"], eurusd["r_f"], eurusd["sigma"],
                             option_type="put")
        assert K > 0

    def test_strike_to_delta_call_positive(self, eurusd):
        d = strike_to_delta(1.08, eurusd["S"], eurusd["T"],
                             eurusd["r_d"], eurusd["r_f"], eurusd["sigma"])
        assert 0 < d < 1

    def test_round_trip_25delta(self, eurusd):
        """delta_to_strike(0.25) → K → strike_to_delta(K) ≈ 0.25"""
        K = delta_to_strike(0.25, eurusd["S"], eurusd["T"],
                             eurusd["r_d"], eurusd["r_f"], eurusd["sigma"])
        d = strike_to_delta(K, eurusd["S"], eurusd["T"],
                             eurusd["r_d"], eurusd["r_f"], eurusd["sigma"])
        assert abs(d - 0.25 * np.exp(-eurusd["r_f"] * eurusd["T"])) < 0.02

    def test_atm_dns_above_atm_forward(self, eurusd):
        K_dns  = atm_dns_strike(eurusd["S"], eurusd["T"], eurusd["r_d"], eurusd["r_f"], eurusd["sigma"])
        K_fwd  = atm_forward_strike(eurusd["S"], eurusd["T"], eurusd["r_d"], eurusd["r_f"])
        # K_DNS = F × exp(½σ²T) ≥ F
        assert K_dns >= K_fwd


# ── FX Smile ──────────────────────────────────────────────────────────────────

class TestFXSmile:

    def test_build_smile_3points(self, smile_quotes):
        smile = build_smile(smile_quotes)
        assert len(smile.strikes) == 3
        assert len(smile.vols) == 3

    def test_strikes_sorted(self, smile_quotes):
        smile = build_smile(smile_quotes)
        assert np.all(np.diff(smile.strikes) > 0)

    def test_vols_positive(self, smile_quotes):
        smile = build_smile(smile_quotes)
        assert np.all(smile.vols > 0)

    def test_atm_in_range(self, smile_quotes):
        smile = build_smile(smile_quotes)
        S = smile_quotes.S
        # ATM strike near spot
        assert S * 0.85 < smile.strikes[1] < S * 1.20

    def test_25call_vol_higher_than_atm_positive_rr(self, smile_quotes):
        """RR > 0 means 25C vol > ATM (positive skew)."""
        smile = build_smile(smile_quotes)
        # smile vols: [25P, ATM, 25C]
        assert smile.vols[-1] > smile.vols[1]   # 25C > ATM

    def test_vol_at_strike_interpolation(self, smile_quotes):
        smile = build_smile(smile_quotes)
        K_mid = (smile.strikes[0] + smile.strikes[-1]) / 2
        v = smile.vol_at_strike(K_mid)
        assert 0.04 < v < 0.20

    def test_build_smile_with_10delta(self):
        q = FXSmileQuotes(
            S=1.08, T=1.0, r_d=0.05, r_f=0.03,
            atm=0.080, rr25=0.010, bf25=0.003,
            rr10=0.020, bf10=0.006,
        )
        smile = build_smile(q)
        assert len(smile.strikes) == 5

    def test_higher_rr_tilts_smile(self):
        """Higher RR → 25C vol increases relative to 25P."""
        q_lo = FXSmileQuotes(S=1.08, T=1.0, r_d=0.05, r_f=0.03,
                              atm=0.08, rr25=0.000, bf25=0.003)
        q_hi = FXSmileQuotes(S=1.08, T=1.0, r_d=0.05, r_f=0.03,
                              atm=0.08, rr25=0.020, bf25=0.003)
        sm_lo = build_smile(q_lo)
        sm_hi = build_smile(q_hi)
        # 25C vol should be higher with higher RR
        idx_25C = -1   # last element
        assert sm_hi.vols[idx_25C] > sm_lo.vols[idx_25C]
