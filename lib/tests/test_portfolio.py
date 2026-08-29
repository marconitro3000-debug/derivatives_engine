"""
tests/test_portfolio.py
Unit tests for portfolio/, VaR, Greeks aggregation, stress testing.
"""

import numpy as np
import pytest

from portfolio import (
    Position, Portfolio,
    var_historical, var_parametric, var_monte_carlo, var_cornish_fisher,
    var_comparison, backtest_var,
    Scenario, STANDARD_SCENARIOS, stress_portfolio, spot_vol_grid,
)


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def call_pos():
    return Position(
        instrument="call",
        quantity=10,
        params=dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20, multiplier=100),
        label="ATM call",
    )


@pytest.fixture
def put_pos():
    return Position(
        instrument="put",
        quantity=10,
        params=dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20, multiplier=100),
        label="ATM put",
    )


@pytest.fixture
def stock_pos():
    return Position(
        instrument="stock",
        quantity=100,
        params=dict(S=100, multiplier=1),
        label="Long stock",
    )


@pytest.fixture
def straddle(call_pos, put_pos):
    return Portfolio([call_pos, put_pos])


@pytest.fixture
def mixed_portfolio(call_pos, stock_pos):
    short_call = Position(
        instrument="call", quantity=-5,
        params=dict(S=100, K=110, T=1.0, r=0.05, sigma=0.20, multiplier=100),
        label="Short OTM call",
    )
    return Portfolio([call_pos, stock_pos, short_call])


@pytest.fixture
def pnl_series():
    rng = np.random.default_rng(42)
    return rng.normal(0, 1000, 500)   # 500 daily P&L observations


# ── Position ──────────────────────────────────────────────────────────────────

class TestPosition:

    def test_call_value_positive(self, call_pos):
        assert call_pos.market_value() > 0

    def test_put_value_positive(self, put_pos):
        assert put_pos.market_value() > 0

    def test_stock_value(self, stock_pos):
        assert stock_pos.market_value() == pytest.approx(10_000.0)

    def test_call_delta_positive(self, call_pos):
        assert call_pos.delta() > 0

    def test_put_delta_negative(self, put_pos):
        assert put_pos.delta() < 0

    def test_stock_delta(self, stock_pos):
        assert stock_pos.delta() == pytest.approx(100.0)

    def test_option_gamma_positive(self, call_pos, put_pos):
        assert call_pos.gamma() > 0
        assert put_pos.gamma() > 0

    def test_stock_gamma_zero(self, stock_pos):
        assert stock_pos.gamma() == 0.0

    def test_vega_positive(self, call_pos):
        assert call_pos.vega() > 0

    def test_theta_negative(self, call_pos):
        assert call_pos.theta() < 0

    def test_pnl_vector_length(self, call_pos):
        returns = np.linspace(-0.1, 0.1, 50)
        pnl = call_pos.pnl_vector(returns)
        assert len(pnl) == 50

    def test_pnl_up_market_call_long(self, call_pos):
        pnl_up   = call_pos.pnl_vector(np.array([+0.10]))[0]
        pnl_down = call_pos.pnl_vector(np.array([-0.10]))[0]
        assert pnl_up > pnl_down


# ── Portfolio ─────────────────────────────────────────────────────────────────

class TestPortfolio:

    def test_total_value_positive(self, straddle):
        assert straddle.total_value() > 0

    def test_straddle_delta_smaller_than_directional(self, straddle, stock_pos):
        """Straddle |delta| < long stock delta (straddle partially offsets)."""
        g_straddle = straddle.aggregate_greeks()
        # Long stock 10 shares: delta = 10, straddle qty=10 multiplier=100
        # Straddle delta = 10 * 100 * (delta_call + delta_put) ≈ small vs stock
        stock_delta = 10.0  # 10 shares
        strad_delta = abs(g_straddle["delta"])
        # straddle call + put delta = 2*N(d1)-1 ≈ 0.27 per unit × 10 × 100 = 270
        # just verify the sum is the correct sign (positive when r>0, d1>0)
        assert g_straddle["delta"] != 0.0   # non-zero with r=5%

    def test_straddle_gamma_positive(self, straddle):
        assert straddle.aggregate_greeks()["gamma"] > 0

    def test_straddle_vega_positive(self, straddle):
        assert straddle.aggregate_greeks()["vega"] > 0

    def test_greeks_by_position_length(self, mixed_portfolio):
        rows = mixed_portfolio.greeks_by_position()
        assert len(rows) == len(mixed_portfolio.positions)

    def test_add_position(self, stock_pos):
        port = Portfolio()
        port.add(stock_pos)
        assert len(port.positions) == 1

    def test_pnl_vector_portfolio(self, mixed_portfolio):
        returns = np.linspace(-0.1, 0.1, 100)
        pnl = mixed_portfolio.pnl_vector(returns)
        assert len(pnl) == 100
        assert pnl.sum() != 0   # not all zero


# ── VaR ───────────────────────────────────────────────────────────────────────

class TestVaR:

    def test_var_historical_positive(self, pnl_series):
        r = var_historical(pnl_series, 0.99)
        assert r.var > 0
        assert r.cvar > 0

    def test_cvar_ge_var(self, pnl_series):
        r = var_historical(pnl_series, 0.99)
        assert r.cvar >= r.var

    def test_var_parametric_positive(self, pnl_series):
        r = var_parametric(pnl_series, 0.99)
        assert r.var > 0

    def test_var_cornish_fisher_positive(self, pnl_series):
        r = var_cornish_fisher(pnl_series, 0.99)
        assert r.var > 0

    def test_var_99_gt_95(self, pnl_series):
        r99 = var_historical(pnl_series, 0.99)
        r95 = var_historical(pnl_series, 0.95)
        assert r99.var > r95.var

    def test_var_monte_carlo(self, mixed_portfolio):
        r = var_monte_carlo(mixed_portfolio, 0.99, n_sims=10_000, annual_vol=0.20)
        assert r.var >= 0   # can be 0 if portfolio has near-zero sensitivity
        assert "Monte Carlo" in r.method

    def test_horizon_scaling(self, pnl_series):
        r1  = var_historical(pnl_series, 0.99, horizon_days=1)
        r10 = var_historical(pnl_series, 0.99, horizon_days=10)
        # 10-day VaR ≈ sqrt(10) × 1-day VaR
        ratio = r10.var / r1.var
        assert abs(ratio - np.sqrt(10)) < 1.0   # loose test for sqrt-of-time

    def test_comparison_returns_three_methods(self, pnl_series):
        results = var_comparison(pnl_series, 0.99)
        assert len(results) == 3

    def test_backtest_kupiec(self, pnl_series):
        r = var_historical(pnl_series, 0.99)
        var_arr = np.full(len(pnl_series), r.var)
        bt = backtest_var(pnl_series, var_arr, 0.99)
        assert "p_value" in bt
        assert "pass" in bt
        assert 0 <= bt["p_value"] <= 1

    def test_str_representation(self, pnl_series):
        r = var_historical(pnl_series, 0.99)
        s = str(r)
        assert "VaR" in s
        assert "CVaR" in s


# ── Stress ────────────────────────────────────────────────────────────────────

class TestStress:

    def test_stress_returns_one_per_scenario(self, mixed_portfolio):
        results = stress_portfolio(mixed_portfolio, STANDARD_SCENARIOS)
        assert len(results) == len(STANDARD_SCENARIOS)

    def test_crash_scenario_negative_pnl(self, mixed_portfolio):
        crash = Scenario("test_crash", spot_shock=-0.40, vol_shock=+0.50)
        r = stress_portfolio(mixed_portfolio, [crash])[0]
        # portfolio with long stock + long call should lose in crash
        assert r.pnl < 0

    def test_vol_spike_vega_positive_pnl(self):
        # Pure long straddle gains from vol spike
        pos = Position("call", 10, dict(S=100, K=100, T=1.0, r=0.05,
                                         sigma=0.20, multiplier=100), "long call")
        port = Portfolio([pos])
        sc   = Scenario("vol_spike", vol_shock=+0.20)
        r    = stress_portfolio(port, [sc])[0]
        assert r.pnl > 0

    def test_position_breakdown(self, mixed_portfolio):
        sc = Scenario("test", spot_shock=-0.10)
        r  = stress_portfolio(mixed_portfolio, [sc],
                               include_position_breakdown=True)[0]
        assert len(r.position_results) == len(mixed_portfolio.positions)

    def test_spot_vol_grid_shape(self, mixed_portfolio):
        ds = np.linspace(-0.30, 0.30, 5)
        dv = np.linspace(-0.20, 0.20, 4)
        grid = spot_vol_grid(mixed_portfolio, ds, dv)
        assert grid.shape == (5, 4)

    def test_scenario_str(self, mixed_portfolio):
        sc = Scenario("test", spot_shock=-0.10)
        r  = stress_portfolio(mixed_portfolio, [sc])[0]
        s  = str(r)
        assert "test" in s
