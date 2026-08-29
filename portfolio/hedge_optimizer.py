"""Automatic protective-put hedge sizing for a multi-name portfolio.

v1 scope (deliberately small): a single scalar `hedge_ratio` in [0, 1],
applied uniformly (weighted by each position's notional) across every name
as a protective put struck at a configurable OTM level. Bisection search
picks the smallest hedge_ratio whose simulated p95 max drawdown meets the
target — cheap and sound because more hedge strictly reduces simulated
downside, so drawdown is monotone non-increasing in hedge_ratio.

Explicitly out of scope for v1: swaps, per-name independent strikes/ratios,
dynamic/rebalanced hedging, transaction costs beyond the option premium.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from portfolio.multi_asset_gbm import simulate_correlated_gbm_paths
from portfolio.risk import RiskReport, risk_report


def _vectorized_put_price(S: np.ndarray, K: np.ndarray, T, r: float, sigma: np.ndarray) -> np.ndarray:
    """Black-Scholes put price, vectorized over S/K/sigma (and optionally T).

    `options/black_scholes.py::price()` only accepts scalar T/sigma (its
    validation does `if T <= 0 or sigma <= 0`, which raises on arrays) — this
    is a small local vectorized version rather than touching that shared,
    already-tested module for every other caller.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    intrinsic = np.maximum(K - S, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bs_value = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(T <= 1e-12, intrinsic, bs_value)


@dataclass
class Position:
    ticker: str
    notional: float


@dataclass
class FrontierPoint:
    hedge_ratio: float
    drawdown_p95_pct: float
    premium_cost_pct: float


@dataclass
class HedgeResult:
    tickers: list[str]
    hedge_ratio: float
    target_met: bool
    premium_cost: float
    premium_cost_pct: float
    unhedged: RiskReport
    hedged: RiskReport
    correlation: list[list[float]]
    volatility: dict[str, float]
    frontier: list[FrontierPoint]


def _portfolio_value_paths(asset_paths: np.ndarray, quantities: np.ndarray) -> np.ndarray:
    """asset_paths: (n_paths, n_grid, n_assets); quantities: (n_assets,) -> (n_paths, n_grid)."""
    return asset_paths @ quantities


def _put_overlay_value_paths(
    asset_paths: np.ndarray,
    strikes: np.ndarray,
    hedge_quantities: np.ndarray,
    rate: float,
    vols: np.ndarray,
    maturity_years: float,
    n_steps: int,
) -> np.ndarray:
    """Time-varying mark-to-market value of a basket of protective puts (one
    per asset), Black-Scholes at each remaining-maturity grid point, falling
    back to intrinsic value at the final step where T=0."""
    n_paths, n_grid, _n_assets = asset_paths.shape
    dt = maturity_years / n_steps
    overlay = np.zeros((n_paths, n_grid))
    for step in range(n_grid):
        remaining = maturity_years - step * dt
        s_t = asset_paths[:, step, :]
        put_values = _vectorized_put_price(s_t, strikes, remaining, rate, vols)
        overlay[:, step] = put_values @ hedge_quantities
    return overlay


def optimize_hedge(
    positions: list[Position],
    volatility: dict[str, float],
    correlation: list[list[float]],
    max_drawdown_target_pct: float,
    hedge_strike_pct: float = 0.90,
    horizon_years: float = 1.0,
    risk_free_rate: float = 0.045,
    n_paths: int = 20_000,
    steps_per_year: int = 52,
    seed: int = 42,
    tolerance: float = 0.01,
    max_iterations: int = 12,
) -> HedgeResult:
    tickers = [p.ticker for p in positions]
    spots = np.array([100.0 for _ in positions])  # positions are notional-denominated; spot normalized to 100
    notionals = np.array([p.notional for p in positions], dtype=float)
    total_notional = float(notionals.sum())
    quantities = notionals / spots  # "shares" at the normalized spot of 100
    vols = np.array([volatility[t] for t in tickers])
    strikes = spots * hedge_strike_pct
    n_steps = max(1, int(round(horizon_years * steps_per_year)))

    asset_paths = simulate_correlated_gbm_paths(
        spots=list(spots), rate=risk_free_rate, volatilities=list(vols), correlation=correlation,
        maturity_years=horizon_years, n_paths=n_paths, n_steps=n_steps, seed=seed,
    )
    unhedged_value_paths = _portfolio_value_paths(asset_paths, quantities)
    unhedged = risk_report(unhedged_value_paths, initial_value=total_notional)

    per_unit_premium = _vectorized_put_price(spots, strikes, horizon_years, risk_free_rate, vols)
    cache: dict[float, tuple[RiskReport, float]] = {}

    def evaluate(hedge_ratio: float) -> tuple[RiskReport, float]:
        key = round(hedge_ratio, 6)
        if key in cache:
            return cache[key]
        hedge_quantities = quantities * hedge_ratio
        premium_cost = float(np.sum(per_unit_premium * hedge_quantities))
        overlay_paths = _put_overlay_value_paths(
            asset_paths, strikes, hedge_quantities, risk_free_rate, vols, horizon_years, n_steps
        )
        hedged_value_paths = unhedged_value_paths + overlay_paths - premium_cost
        result = (risk_report(hedged_value_paths, initial_value=total_notional), premium_cost)
        cache[key] = result
        return result

    def build_frontier(*extra_ratios: float) -> list[FrontierPoint]:
        """Cost-vs-protection curve across a fixed hedge_ratio grid (plus the
        chosen point) — lets the caller see the trade-off the optimizer
        picked from, instead of trusting a single black-box answer."""
        grid = sorted({0.0, 0.25, 0.5, 0.75, 1.0, *(round(r, 4) for r in extra_ratios)})
        points = []
        for ratio in grid:
            report, cost = evaluate(ratio)
            points.append(
                FrontierPoint(
                    hedge_ratio=ratio,
                    drawdown_p95_pct=report.drawdown_p95_pct,
                    premium_cost_pct=100.0 * cost / total_notional,
                )
            )
        return points

    # hedge_ratio=1.0 is the best this hedge design can do — check it can even reach the target
    best_report, best_cost = evaluate(1.0)
    if best_report.drawdown_p95_pct > max_drawdown_target_pct:
        return HedgeResult(
            tickers=tickers, hedge_ratio=1.0, target_met=False,
            premium_cost=best_cost, premium_cost_pct=100.0 * best_cost / total_notional,
            unhedged=unhedged, hedged=best_report, correlation=correlation, volatility=volatility,
            frontier=build_frontier(),
        )

    zero_report, _ = evaluate(0.0)
    if zero_report.drawdown_p95_pct <= max_drawdown_target_pct:
        return HedgeResult(
            tickers=tickers, hedge_ratio=0.0, target_met=True,
            premium_cost=0.0, premium_cost_pct=0.0,
            unhedged=unhedged, hedged=zero_report, correlation=correlation, volatility=volatility,
            frontier=build_frontier(),
        )

    lo, hi = 0.0, 1.0
    hi_report, hi_cost = best_report, best_cost
    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        mid_report, mid_cost = evaluate(mid)
        if mid_report.drawdown_p95_pct <= max_drawdown_target_pct:
            hi, hi_report, hi_cost = mid, mid_report, mid_cost
        else:
            lo = mid
        if hi - lo <= tolerance:
            break

    chosen = round(hi, 4)
    return HedgeResult(
        tickers=tickers, hedge_ratio=chosen, target_met=True,
        premium_cost=hi_cost, premium_cost_pct=100.0 * hi_cost / total_notional,
        unhedged=unhedged, hedged=hi_report, correlation=correlation, volatility=volatility,
        frontier=build_frontier(chosen),
    )
