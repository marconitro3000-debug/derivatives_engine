"""
portfolio/stress.py
Scenario stress testing for option / equity portfolios.

Scenarios apply simultaneous shocks to:
  - spot price (multiplicative or additive)
  - implied volatility (additive, in vol pts)
  - risk-free rate (additive, in percent)

Built-in scenarios are based on documented historical events.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from options.black_scholes import price as bs_price


@dataclass
class Scenario:
    name:        str
    spot_shock:  float = 0.0   # multiplicative: +0.10 = +10% spot
    vol_shock:   float = 0.0   # additive: +0.20 = +20 vol pts
    rate_shock:  float = 0.0   # additive: +0.01 = +100bps
    description: str = ""


STANDARD_SCENARIOS: list[Scenario] = [
    Scenario("2008 Lehman",      spot_shock=-0.40, vol_shock=+0.50,
             description="Lehman collapse: S&P -40%, VIX +50 pts"),
    Scenario("COVID Mar 2020",   spot_shock=-0.35, vol_shock=+0.80,
             description="COVID crash: S&P -35%, VIX peak ~85"),
    Scenario("2020 Recovery",    spot_shock=+0.70, vol_shock=-0.30,
             description="Post-COVID rally through 2020"),
    Scenario("Dot-com crash",    spot_shock=-0.50, vol_shock=+0.30,
             description="NASDAQ -78% (2000-02), broad market -50%"),
    Scenario("Flash crash 2010", spot_shock=-0.10, vol_shock=+0.30,
             description="May 6 2010 intraday -10%"),
    Scenario("Rate +200bps",     rate_shock=+0.02,
             description="Fed tightening shock +200bps"),
    Scenario("Rate -200bps",     rate_shock=-0.02,
             description="Emergency cut / zero-rate environment"),
    Scenario("Vol spike +30pts", vol_shock=+0.30,
             description="Implied vol jump (VIX-like)"),
    Scenario("Vol crush -15pts", vol_shock=-0.15,
             description="Post-event vol crush"),
    Scenario("Spot +20%",        spot_shock=+0.20,
             description="Bull run / squeeze"),
    Scenario("Spot -20%",        spot_shock=-0.20,
             description="Bear market / correction"),
]


@dataclass
class StressResult:
    scenario:   Scenario
    base_value: float
    stressed_value: float
    pnl:        float
    pnl_pct:    float
    position_results: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"{self.scenario.name:<24} "
                f"PnL={self.pnl:>+12,.0f}  "
                f"({self.pnl_pct:>+6.1f}%)")


def _stressed_option_value(pos, scenario: Scenario) -> float:
    """Revalue a single option position under a scenario."""
    p = pos.params
    S_new     = p["S"] * (1 + scenario.spot_shock)
    sigma_new = max(p["sigma"] + scenario.vol_shock, 0.001)
    r_new     = p["r"] + scenario.rate_shock
    m         = p.get("multiplier", 100)

    return pos.quantity * bs_price(
        S_new, p["K"], p["T"], r_new, sigma_new, pos.instrument
    ) * m


def _stressed_stock_value(pos, scenario: Scenario) -> float:
    p = pos.params
    S_new = p["S"] * (1 + scenario.spot_shock)
    return pos.quantity * S_new * p.get("multiplier", 1)


def _stressed_value(pos, scenario: Scenario) -> float:
    if pos.instrument in ("call", "put"):
        return _stressed_option_value(pos, scenario)
    if pos.instrument == "stock":
        return _stressed_stock_value(pos, scenario)
    return pos.market_value()


def stress_portfolio(portfolio, scenarios: list[Scenario] | None = None,
                     include_position_breakdown: bool = True) -> list[StressResult]:
    """
    Apply a list of scenarios to a portfolio.

    Parameters
    ----------
    portfolio  : Portfolio instance
    scenarios  : list of Scenario; defaults to STANDARD_SCENARIOS
    include_position_breakdown : whether to compute per-position impact

    Returns
    -------
    list of StressResult, one per scenario
    """
    if scenarios is None:
        scenarios = STANDARD_SCENARIOS

    base_value = portfolio.total_value()
    results    = []

    for sc in scenarios:
        stressed_value = sum(_stressed_value(pos, sc) for pos in portfolio.positions)
        pnl            = stressed_value - base_value
        pnl_pct        = pnl / abs(base_value) * 100 if base_value != 0 else 0.0

        pos_results = []
        if include_position_breakdown:
            for pos in portfolio.positions:
                sv   = _stressed_value(pos, sc)
                bv   = pos.market_value()
                pos_results.append({
                    "label":          pos.label or pos.instrument,
                    "base_value":     bv,
                    "stressed_value": sv,
                    "pnl":            sv - bv,
                })

        results.append(StressResult(
            scenario=sc,
            base_value=base_value,
            stressed_value=float(stressed_value),
            pnl=float(pnl),
            pnl_pct=float(pnl_pct),
            position_results=pos_results,
        ))

    return results


def spot_vol_grid(portfolio, spot_shocks: np.ndarray,
                  vol_shocks: np.ndarray) -> np.ndarray:
    """
    PnL grid over a 2D space of (spot shock, vol shock).

    Returns array of shape (len(spot_shocks), len(vol_shocks)).
    """
    base = portfolio.total_value()
    grid = np.zeros((len(spot_shocks), len(vol_shocks)))

    for i, ds in enumerate(spot_shocks):
        for j, dv in enumerate(vol_shocks):
            sc = Scenario("grid", spot_shock=ds, vol_shock=dv)
            stressed = sum(_stressed_value(pos, sc) for pos in portfolio.positions)
            grid[i, j] = stressed - base

    return grid
