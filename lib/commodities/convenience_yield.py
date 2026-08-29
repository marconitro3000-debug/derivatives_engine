"""
commodities/convenience_yield.py
Cost-of-carry model and convenience yield term structure.

Cost-of-carry model:
    F(T) = S · e^{(r + u − δ)T}

where:
    r — risk-free rate (continuously compounded)
    u — storage cost (fraction of spot per year, e.g. 0.02 = 2%)
    δ — convenience yield (fraction per year)

Solving for implied convenience yield:
    δ(T) = r + u − (1/T) · ln(F(T)/S)

The convenience yield reflects the benefit of holding the physical commodity:
  - Inventory scarcity → high δ → backwardation (F < S·e^{rT})
  - Ample inventory   → low δ  → contango      (F > S·e^{rT})

Spread options (Margrabe 1978):
    Exchange option: right to exchange asset 2 for asset 1
    V = F₁·N(d₁) − F₂·N(d₂)
    d₁ = [ln(F₁/F₂) + ½σ²T] / (σ√T)
    σ = √(σ₁² + σ₂² − 2ρσ₁σ₂)   (spread vol)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass

_N = norm.cdf


@dataclass
class ConvenienceYieldCurve:
    """
    Implied convenience yield term structure from futures prices.

    Parameters
    ----------
    maturities : list of maturities in years
    yields     : implied convenience yields at each maturity
    """
    maturities: np.ndarray
    yields:     np.ndarray
    r:          float = 0.05
    u:          float = 0.02

    def yield_at(self, T: float) -> float:
        """Linearly interpolated convenience yield at maturity T."""
        return float(np.interp(T, self.maturities, self.yields))

    def average_yield(self, T1: float, T2: float) -> float:
        T_grid = np.linspace(T1, T2, 100)
        return float(np.mean([self.yield_at(t) for t in T_grid]))

    def implied_futures(self, S: float, T: float) -> float:
        """Implied futures price using this convenience yield term structure."""
        delta = self.yield_at(T)
        return S * np.exp((self.r + self.u - delta) * T)


# ── convenience yield from futures ───────────────────────────────────────────

def implied_convenience_yield(futures_price: float, spot: float, T: float,
                               r: float, storage_cost: float = 0.0) -> float:
    """
    δ(T) = r + u − (1/T) · ln(F/S)

    Parameters
    ----------
    storage_cost : proportional storage cost per year (e.g. 0.02 = 2% of spot)
    """
    if T <= 0:
        raise ValueError("T must be > 0")
    return float(r + storage_cost - np.log(futures_price / spot) / T)


def convenience_yield_curve(futures_prices: np.ndarray, maturities: np.ndarray,
                              spot: float, r: float,
                              storage_cost: float = 0.02) -> ConvenienceYieldCurve:
    """Build a ConvenienceYieldCurve from observed futures prices."""
    yields = np.array([
        implied_convenience_yield(F, spot, T, r, storage_cost)
        for F, T in zip(futures_prices, maturities)
    ])
    return ConvenienceYieldCurve(
        maturities=np.asarray(maturities),
        yields=yields,
        r=r,
        u=storage_cost,
    )


# ── Futures pricing formulas ──────────────────────────────────────────────────

def futures_fair_price(S: float, T: float, r: float,
                        convenience_yield: float = 0.0,
                        storage_cost: float = 0.0) -> float:
    """
    F = S · e^{(r + u − δ)T}

    Parameters
    ----------
    S               : spot price
    T               : years to delivery
    r               : risk-free rate
    convenience_yield: δ (fraction per year)
    storage_cost    : u (fraction per year)
    """
    return float(S * np.exp((r + storage_cost - convenience_yield) * T))


def implied_spot(F: float, T: float, r: float,
                  convenience_yield: float, storage_cost: float = 0.0) -> float:
    """S = F · e^{−(r + u − δ)T}"""
    return float(F * np.exp(-(r + storage_cost - convenience_yield) * T))


# ── Spread options (Margrabe 1978) ────────────────────────────────────────────

def spread_option_margrabe(F1: float, F2: float, T: float,
                            sigma1: float, sigma2: float, rho: float,
                            r: float, option_type: str = "call") -> dict:
    """
    Exchange option / spread option using Margrabe (1978) formula.

    Payoff: max(F1 − F2 − K, 0) with K=0 (exchange option).
    For K≠0 use Kirk's approximation (implemented in `spread_option_kirk`).

    Parameters
    ----------
    F1, F2  : futures prices of the two assets
    sigma1, sigma2 : individual vols
    rho     : correlation between log-price changes
    """
    sigma = np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)
    df    = np.exp(-r * T)

    if sigma <= 0 or T <= 0:
        if option_type == "call":
            return {"price": max(F1 - F2, 0) * df}
        return {"price": max(F2 - F1, 0) * df}

    d1 = (np.log(F1 / F2) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        px = df * (F1 * _N(d1) - F2 * _N(d2))
    else:
        px = df * (F2 * _N(-d2) - F1 * _N(-d1))

    return {
        "price":       float(px),
        "spread_vol":  float(sigma),
        "d1":          float(d1),
        "d2":          float(d2),
    }


def spread_option_kirk(F1: float, F2: float, K: float, T: float,
                        sigma1: float, sigma2: float, rho: float,
                        r: float) -> dict:
    """
    Kirk's (1995) approximation for spread options with non-zero strike K.

    Approximates: max(F1 − F2 − K, 0)
    by treating G = F2 + K as a single asset with effective vol σ_G.
    """
    G      = F2 + K
    sigma_G = sigma2 * F2 / G   # approximate vol of G
    sigma_spread = np.sqrt(sigma1**2 + sigma_G**2 - 2 * rho * sigma1 * sigma_G)
    df    = np.exp(-r * T)

    if sigma_spread <= 0 or T <= 0:
        return {"price": max(F1 - F2 - K, 0) * df}

    d1 = (np.log(F1 / G) + 0.5 * sigma_spread**2 * T) / (sigma_spread * np.sqrt(T))
    d2 = d1 - sigma_spread * np.sqrt(T)
    px  = df * (F1 * _N(d1) - G * _N(d2))

    return {
        "price":       float(px),
        "spread_vol":  float(sigma_spread),
        "d1":          float(d1),
        "d2":          float(d2),
    }


# ── Oil-specific metrics ──────────────────────────────────────────────────────

def wti_brent_spread(wti: float, brent: float) -> float:
    """WTI-Brent spread. Normally negative (Brent premium)."""
    return float(wti - brent)


def time_spread_carry(curve, T1: float, T2: float,
                       r: float, storage_cost: float) -> dict:
    """
    Fair value of calendar spread vs actual, and implied carry.

    Returns fair_spread (cost-of-carry), actual_spread, carry_alpha.
    """
    F1 = float(curve.price(T1))
    F2 = float(curve.price(T2))
    dt = T2 - T1
    fair_spread    = F1 * (np.exp((r + storage_cost) * dt) - 1)
    actual_spread  = F2 - F1
    carry_alpha    = actual_spread - fair_spread
    return {
        "F1":           F1,
        "F2":           F2,
        "fair_spread":  float(fair_spread),
        "actual_spread":float(actual_spread),
        "carry_alpha":  float(carry_alpha),
    }
