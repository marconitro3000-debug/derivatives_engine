"""
portfolio/position.py
Position and Portfolio containers for Greeks aggregation and risk.

Supported instrument types:
  "call", "put"     — European vanilla options (Black-Scholes)
  "stock"           — equity spot position
  "future"          — futures contract (delta = quantity × multiplier)
  "bond"            — zero-coupon bond
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from options.black_scholes import price as bs_price, greeks as bs_greeks


@dataclass
class Position:
    """
    Single instrument position.

    Parameters
    ----------
    instrument : "call" | "put" | "stock" | "future" | "bond"
    quantity   : number of contracts (+ve = long, -ve = short)
    params     : instrument-specific pricing parameters (see below)

    Option params : S, K, T, r, sigma, multiplier (default 100)
    Stock params  : S, multiplier (default 1)
    Future params : S, multiplier (default 1), r, T
    Bond params   : face, T, r
    """
    instrument: str
    quantity: float
    params: dict = field(default_factory=dict)
    label: str = ""

    # ── value ──────────────────────────────────────────────────────────────────

    def unit_value(self) -> float:
        """Value of one unit (one contract, before quantity)."""
        p = self.params
        if self.instrument in ("call", "put"):
            m = p.get("multiplier", 100)
            return bs_price(p["S"], p["K"], p["T"], p["r"],
                            p["sigma"], self.instrument) * m
        if self.instrument == "stock":
            return p["S"] * p.get("multiplier", 1)
        if self.instrument == "future":
            return 0.0   # daily mark, initial value = 0
        if self.instrument == "bond":
            return p["face"] * np.exp(-p["r"] * p["T"])
        raise ValueError(f"Unknown instrument: {self.instrument!r}")

    def market_value(self) -> float:
        return self.quantity * self.unit_value()

    # ── greeks (dollar-denominated) ────────────────────────────────────────────

    def _option_greeks(self) -> dict:
        """Return per-unit, per-contract Greeks (scaled by multiplier)."""
        p   = self.params
        m   = p.get("multiplier", 100)
        raw = bs_greeks(p["S"], p["K"], p["T"], p["r"], p["sigma"])
        t   = self.instrument   # "call" or "put"
        return {
            "delta": raw[f"delta_{t}"] * m,
            "gamma": raw["gamma"]      * m,
            "vega":  raw["vega"]       * m,
            "theta": raw[f"theta_{t}"] * m,
            "rho":   raw[f"rho_{t}"]   * m,
        }

    def delta(self) -> float:
        """Dollar delta: ΔV / ΔS"""
        p = self.params
        if self.instrument in ("call", "put"):
            return self.quantity * self._option_greeks()["delta"]
        if self.instrument == "stock":
            return self.quantity * p.get("multiplier", 1)
        if self.instrument == "future":
            return self.quantity * p.get("multiplier", 1)
        return 0.0

    def gamma(self) -> float:
        if self.instrument in ("call", "put"):
            return self.quantity * self._option_greeks()["gamma"]
        return 0.0

    def vega(self) -> float:
        """Vega in dollars per 1% move in vol."""
        if self.instrument in ("call", "put"):
            return self.quantity * self._option_greeks()["vega"] / 100.0
        return 0.0

    def theta(self) -> float:
        """Theta in dollars per calendar day."""
        if self.instrument in ("call", "put"):
            return self.quantity * self._option_greeks()["theta"] / 365.0
        return 0.0

    def rho(self) -> float:
        if self.instrument in ("call", "put"):
            return self.quantity * self._option_greeks()["rho"] / 100.0
        return 0.0

    # ── pnl simulation ─────────────────────────────────────────────────────────

    def pnl_vector(self, spot_returns: np.ndarray,
                   vol_shock: float = 0.0,
                   rate_shock: float = 0.0) -> np.ndarray:
        """
        Simulate P&L over a vector of spot return scenarios.

        Parameters
        ----------
        spot_returns : array of daily log-returns (e.g. shape (N,))
        vol_shock    : shift in implied vol (e.g. +0.05 = +5 vol pts)
        rate_shock   : shift in risk-free rate

        Returns
        -------
        pnl : array of P&L values (same length as spot_returns)
        """
        p = self.params
        V0 = self.market_value()

        if self.instrument in ("call", "put"):
            S_new = p["S"] * np.exp(spot_returns)
            new_sigma = max(p["sigma"] + vol_shock, 1e-6)
            new_r     = p["r"] + rate_shock
            V1 = self.quantity * np.array([
                bs_price(s, p["K"], p["T"], new_r, new_sigma, self.instrument)
                * p.get("multiplier", 100)
                for s in S_new
            ])
            return V1 - V0

        if self.instrument == "stock":
            S_new = p["S"] * np.exp(spot_returns)
            return self.quantity * (S_new - p["S"]) * p.get("multiplier", 1)

        return np.zeros(len(spot_returns))

    def __repr__(self) -> str:
        lbl = self.label or self.instrument
        return f"Position({lbl!r}, qty={self.quantity:+.1f})"


# ── Portfolio ─────────────────────────────────────────────────────────────────

class Portfolio:
    """Collection of positions with aggregate risk metrics."""

    def __init__(self, positions: list[Position] | None = None):
        self.positions: list[Position] = list(positions or [])

    def add(self, pos: Position) -> "Portfolio":
        self.positions.append(pos)
        return self

    # ── value ──────────────────────────────────────────────────────────────────

    def total_value(self) -> float:
        return sum(p.market_value() for p in self.positions)

    # ── greeks ─────────────────────────────────────────────────────────────────

    def aggregate_greeks(self) -> dict:
        return {
            "delta": sum(p.delta() for p in self.positions),
            "gamma": sum(p.gamma() for p in self.positions),
            "vega":  sum(p.vega()  for p in self.positions),
            "theta": sum(p.theta() for p in self.positions),
            "rho":   sum(p.rho()   for p in self.positions),
        }

    def greeks_by_position(self) -> list[dict]:
        out = []
        for pos in self.positions:
            out.append({
                "label":  pos.label or pos.instrument,
                "qty":    pos.quantity,
                "value":  pos.market_value(),
                "delta":  pos.delta(),
                "gamma":  pos.gamma(),
                "vega":   pos.vega(),
                "theta":  pos.theta(),
            })
        return out

    # ── pnl simulation ─────────────────────────────────────────────────────────

    def pnl_vector(self, spot_returns: np.ndarray,
                   vol_shock: float = 0.0,
                   rate_shock: float = 0.0) -> np.ndarray:
        total = np.zeros(len(spot_returns))
        for pos in self.positions:
            total += pos.pnl_vector(spot_returns, vol_shock, rate_shock)
        return total

    def __repr__(self) -> str:
        return f"Portfolio({len(self.positions)} positions, value={self.total_value():,.2f})"
