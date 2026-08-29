from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio import correlation as correlation_mod
from portfolio.hedge_optimizer import Position, optimize_hedge


def test_estimate_correlation_from_synthetic_returns(monkeypatch):
    rng = np.random.default_rng(7)
    n = 300
    common = rng.standard_normal(n)
    idiosyncratic_a = rng.standard_normal(n)
    idiosyncratic_b = rng.standard_normal(n)
    # true correlation ~0.7 by construction (shared common factor)
    returns_a = 0.02 * (0.85 * common + 0.53 * idiosyncratic_a)
    returns_b = 0.02 * (0.85 * common + 0.53 * idiosyncratic_b)
    price_a = 100 * np.exp(np.cumsum(returns_a))
    price_b = 100 * np.exp(np.cumsum(returns_b))
    dates = pd.date_range("2025-01-01", periods=n, freq="D")

    def fake_get_history(ticker, period="1y"):
        series = price_a if ticker == "A" else price_b
        return pd.DataFrame({"Close": series}, index=dates)

    monkeypatch.setattr(correlation_mod, "get_history", fake_get_history)

    result = correlation_mod.estimate_correlation(["A", "B"])

    assert result.missing == []
    assert result.tickers == ["A", "B"]
    assert result.correlation[0][1] == pytest.approx(0.7, abs=0.15)
    assert 0.1 < result.volatility["A"] < 1.0


def test_estimate_correlation_reports_missing_ticker(monkeypatch):
    def fake_get_history(ticker, period="1y"):
        if ticker == "BAD":
            raise RuntimeError("no data")
        return pd.DataFrame({"Close": np.full(100, 100.0)})

    monkeypatch.setattr(correlation_mod, "get_history", fake_get_history)
    result = correlation_mod.estimate_correlation(["BAD"])
    assert result.missing == ["BAD"]
    assert result.tickers == []


def _flat_portfolio(n_names: int = 2) -> tuple[list[Position], dict[str, float], list[list[float]]]:
    tickers = [f"T{i}" for i in range(n_names)]
    positions = [Position(ticker=t, notional=500.0) for t in tickers]
    volatility = {t: 0.30 for t in tickers}
    correlation = [[1.0 if i == j else 0.3 for j in range(n_names)] for i in range(n_names)]
    return positions, volatility, correlation


def test_hedge_ratio_zero_when_target_already_met():
    positions, vol, corr = _flat_portfolio()
    result = optimize_hedge(
        positions=positions, volatility=vol, correlation=corr,
        max_drawdown_target_pct=95.0, n_paths=3000,
    )
    assert result.hedge_ratio == 0.0
    assert result.target_met is True
    assert result.premium_cost == 0.0


def test_hedge_cannot_meet_impossible_target():
    positions, vol, corr = _flat_portfolio()
    result = optimize_hedge(
        positions=positions, volatility=vol, correlation=corr,
        max_drawdown_target_pct=0.01, n_paths=3000,
    )
    assert result.hedge_ratio == 1.0
    assert result.target_met is False


def test_hedge_finds_intermediate_ratio_for_achievable_target():
    positions, vol, corr = _flat_portfolio()
    unhedged = optimize_hedge(
        positions=positions, volatility=vol, correlation=corr,
        max_drawdown_target_pct=95.0, n_paths=6000,
    ).unhedged

    target = (unhedged.drawdown_p95_pct + 20.0) / 2  # comfortably between "always met" and "impossible"
    result = optimize_hedge(
        positions=positions, volatility=vol, correlation=corr,
        max_drawdown_target_pct=target, hedge_strike_pct=0.85, n_paths=6000,
    )
    assert result.target_met is True
    assert 0.0 < result.hedge_ratio < 1.0
    assert result.hedged.drawdown_p95_pct <= target + 1.0  # small MC/bisection slack
    assert result.premium_cost > 0.0


def test_frontier_is_monotone_and_covers_full_range():
    positions, vol, corr = _flat_portfolio()
    result = optimize_hedge(
        positions=positions, volatility=vol, correlation=corr,
        max_drawdown_target_pct=25.0, hedge_strike_pct=0.90, n_paths=6000,
    )
    ratios = [p.hedge_ratio for p in result.frontier]
    drawdowns = [p.drawdown_p95_pct for p in result.frontier]
    costs = [p.premium_cost_pct for p in result.frontier]

    assert ratios == sorted(ratios)
    assert ratios[0] == 0.0
    assert ratios[-1] == 1.0
    # more hedge -> lower drawdown, higher cost (monotone in both directions)
    assert all(a >= b for a, b in zip(drawdowns, drawdowns[1:]))
    assert all(a <= b for a, b in zip(costs, costs[1:]))
    # the optimizer's chosen ratio is included in the grid
    assert any(r == pytest.approx(result.hedge_ratio, abs=1e-6) for r in ratios)


def test_tighter_target_costs_more_premium():
    positions, vol, corr = _flat_portfolio()
    loose = optimize_hedge(
        positions=positions, volatility=vol, correlation=corr,
        max_drawdown_target_pct=28.0, hedge_strike_pct=0.85, n_paths=6000,
    )
    tight = optimize_hedge(
        positions=positions, volatility=vol, correlation=corr,
        max_drawdown_target_pct=22.0, hedge_strike_pct=0.85, n_paths=6000,
    )
    assert tight.premium_cost_pct >= loose.premium_cost_pct
