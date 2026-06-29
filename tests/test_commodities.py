"""
tests/test_commodities.py
Unit tests for commodities/ — futures curve, convenience yield, Schwartz model.
"""

import numpy as np
import pytest

from commodities import (
    FuturesCurve, crack_spread, spark_spread,
    implied_convenience_yield, futures_fair_price, implied_spot,
    ConvenienceYieldCurve, convenience_yield_curve,
    spread_option_margrabe, spread_option_kirk,
    SchwartzParams, futures_price, calibrate_schwartz,
    simulate_schwartz, price_commodity_option, schwartz_fit_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def oil_curve():
    """WTI-like contango futures curve."""
    mats = np.array([1/12, 3/12, 6/12, 1.0, 1.5, 2.0])
    prices = np.array([80.0, 80.5, 81.2, 82.0, 82.5, 83.0])
    return FuturesCurve(maturities=mats, futures_prices=prices,
                         spot=79.5, commodity="WTI Crude")


@pytest.fixture(scope="module")
def backwardation_curve():
    """Gold-like backwardation curve."""
    mats   = np.array([1/12, 3/12, 6/12, 1.0, 2.0])
    prices = np.array([1900, 1895, 1888, 1880, 1875], dtype=float)
    return FuturesCurve(maturities=mats, futures_prices=prices,
                         spot=1905, commodity="Gold")


@pytest.fixture(scope="module")
def schwartz_params():
    return SchwartzParams(kappa=0.8, mu_star=np.log(80), sigma=0.30)


# ── FuturesCurve ──────────────────────────────────────────────────────────────

class TestFuturesCurve:

    def test_price_interpolation(self, oil_curve):
        p = oil_curve.price(0.75)
        # Between 6m (81.2) and 1y (82.0)
        assert 81.2 < float(p) < 82.0

    def test_calendar_spread_positive_contango(self, oil_curve):
        cs = oil_curve.calendar_spread(0.5, 1.0)
        assert cs > 0   # contango

    def test_calendar_spread_negative_backwardation(self, backwardation_curve):
        cs = backwardation_curve.calendar_spread(0.25, 1.0)
        assert cs < 0   # backwardation

    def test_is_contango(self, oil_curve):
        assert oil_curve.is_contango() is True

    def test_is_backwardation(self, backwardation_curve):
        assert backwardation_curve.is_contango() is False

    def test_roll_yield_backwardation_positive(self, backwardation_curve):
        # In backwardation, rolling from long to shorter expiry is profitable
        ry = backwardation_curve.roll_yield(0.25, 1.0)
        assert ry > 0

    def test_annualized_basis_shape(self, oil_curve):
        cy = oil_curve.annualized_basis(r=0.05)
        assert len(cy) == len(oil_curve.maturities)

    def test_prices_sorted_by_maturity(self, oil_curve):
        mats = oil_curve.maturities
        assert np.all(np.diff(mats) > 0)

    def test_summary_contains_commodity_name(self, oil_curve):
        s = oil_curve.summary()
        assert "WTI Crude" in s

    def test_crack_spread(self):
        cs = crack_spread(crude_price=80, gasoline_price=100,
                           heating_oil_price=95)
        assert isinstance(cs, float)

    def test_spark_spread(self):
        ss = spark_spread(power_price=50, gas_price=4.0, heat_rate=7.5)
        assert ss == pytest.approx(50 - 7.5 * 4.0)


# ── Convenience Yield ─────────────────────────────────────────────────────────

class TestConvenienceYield:

    def test_implied_cy_contango_near_zero(self):
        """Contango: F > S*e^{rT} → cy < r + storage."""
        cy = implied_convenience_yield(82.0, 80.0, 1.0, r=0.05, storage_cost=0.02)
        assert cy < 0.07   # should be low (costs exceed convenience)

    def test_implied_cy_backwardation_high(self):
        """Backwardation: F < S → cy > r + storage."""
        cy = implied_convenience_yield(75.0, 80.0, 1.0, r=0.05, storage_cost=0.02)
        assert cy > 0.07

    def test_futures_fair_price_roundtrip(self):
        S, r, u, cy = 80.0, 0.05, 0.02, 0.04
        F  = futures_fair_price(S, 1.0, r, cy, u)
        cy2 = implied_convenience_yield(F, S, 1.0, r, u)
        assert abs(cy2 - cy) < 1e-10

    def test_implied_spot_roundtrip(self):
        F, cy = 82.0, 0.03
        S_back = implied_spot(F, 1.0, 0.05, cy, 0.02)
        F_back = futures_fair_price(S_back, 1.0, 0.05, cy, 0.02)
        assert abs(F_back - F) < 1e-8

    def test_convenience_yield_curve(self, oil_curve):
        cy_curve = convenience_yield_curve(
            oil_curve.futures_prices, oil_curve.maturities,
            oil_curve.spot, r=0.05, storage_cost=0.02,
        )
        assert len(cy_curve.yields) == len(oil_curve.maturities)

    def test_spread_option_margrabe_call_positive(self):
        res = spread_option_margrabe(F1=100, F2=90, T=1.0,
                                      sigma1=0.20, sigma2=0.25, rho=0.6, r=0.05)
        assert res["price"] > 0

    def test_margrabe_exchange_parity(self):
        """Call - Put = F1 - F2 (discounted)."""
        c = spread_option_margrabe(100, 90, 1.0, 0.20, 0.25, 0.6, 0.05, "call")
        p = spread_option_margrabe(100, 90, 1.0, 0.20, 0.25, 0.6, 0.05, "put")
        df = np.exp(-0.05 * 1.0)
        assert abs((c["price"] - p["price"]) - (100 - 90) * df) < 1e-6

    def test_kirk_approximation_positive(self):
        res = spread_option_kirk(F1=100, F2=90, K=5.0, T=1.0,
                                  sigma1=0.20, sigma2=0.25, rho=0.6, r=0.05)
        assert res["price"] > 0

    def test_kirk_le_margrabe(self):
        """With K>0, Kirk price < Margrabe price (K=0 is cheaper strike)."""
        m = spread_option_margrabe(100, 90, 1.0, 0.20, 0.25, 0.6, 0.05)
        k = spread_option_kirk(100, 90, 5.0, 1.0, 0.20, 0.25, 0.6, 0.05)
        assert k["price"] < m["price"]


# ── Schwartz 1F Model ─────────────────────────────────────────────────────────

class TestSchwartzModel:

    def test_futures_price_at_zero_is_spot(self, schwartz_params):
        """F(0, T→0) → S₀."""
        S0 = 80.0
        F  = futures_price(S0, np.array([0.001]), schwartz_params)[0]
        assert abs(F - S0) / S0 < 0.01

    def test_futures_price_positive(self, schwartz_params):
        T   = np.linspace(0.1, 5.0, 20)
        F   = futures_price(80.0, T, schwartz_params)
        assert (F > 0).all()

    def test_long_run_price(self, schwartz_params):
        """F(T→∞) → long-run price."""
        F_long  = float(futures_price(80.0, np.array([100.0]), schwartz_params)[0])
        F_model = schwartz_params.long_run_price()
        assert abs(F_long - F_model) < 0.1

    def test_half_life_positive(self, schwartz_params):
        assert schwartz_params.half_life() > 0

    def test_calibrate_recovers_params(self):
        """Calibrate model to its own forward curve → recover params."""
        true_p = SchwartzParams(kappa=0.8, mu_star=np.log(80), sigma=0.25)
        S0     = 80.0
        T_list = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
        F_mkt  = futures_price(S0, T_list, true_p)

        fitted, rmse = calibrate_schwartz(F_mkt, T_list, S0, method="de")
        assert rmse < 0.5
        # Long-run price should be close
        assert abs(fitted.long_run_price() - true_p.long_run_price()) < 2.0

    def test_calibrate_rmse_low(self):
        true_p = SchwartzParams(kappa=1.0, mu_star=np.log(50), sigma=0.35)
        T_list = np.linspace(0.5, 3.0, 8)
        F_mkt  = futures_price(50.0, T_list, true_p)
        _, rmse = calibrate_schwartz(F_mkt, T_list, 50.0)
        assert rmse < 0.1

    def test_simulate_shape(self, schwartz_params):
        paths = simulate_schwartz(80.0, 1.0, schwartz_params,
                                   n_sims=500, n_steps=50)
        assert paths.shape == (500, 51)

    def test_simulate_paths_positive(self, schwartz_params):
        paths = simulate_schwartz(80.0, 1.0, schwartz_params,
                                   n_sims=1000, n_steps=100)
        assert (paths > 0).all()

    def test_simulate_mean_vs_futures(self, schwartz_params):
        """E[S_T] ≈ F(0, T)."""
        S0, T = 80.0, 1.0
        paths = simulate_schwartz(S0, T, schwartz_params,
                                   n_sims=50_000, n_steps=100, seed=0)
        S_T   = paths[:, -1]
        F_model = float(futures_price(S0, np.array([T]), schwartz_params)[0])
        assert abs(np.mean(S_T) - F_model) / F_model < 0.03

    def test_option_price_positive(self, schwartz_params):
        res = price_commodity_option(80.0, 80.0, 1.0, 0.05, schwartz_params,
                                      n_sims=10_000, n_steps=50)
        assert res["price"] > 0

    def test_fit_summary_contains_rmse(self, schwartz_params):
        T  = np.array([0.5, 1.0, 2.0])
        Fm = futures_price(80.0, T, schwartz_params)
        s  = schwartz_fit_summary(80.0, T, Fm, schwartz_params)
        assert "RMSE" in s
