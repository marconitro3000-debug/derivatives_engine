"""
options/implied_vol.py
Implied volatility extraction via Newton-Raphson (primary) with Brent fallback.
"""

import numpy as np
from scipy.optimize import brentq

from .black_scholes import price, greeks


# ── constants ─────────────────────────────────────────────────────────────────

_MAX_ITER   = 100
_TOL_PRICE  = 1e-8
_TOL_VEGA   = 1e-12
_SIGMA_LO   = 1e-6
_SIGMA_HI   = 10.0      # 1000% vol ceiling


# ── initial guess (Brenner-Subrahmanyam approximation) ───────────────────────

def _sigma_init(S: float, K: float, T: float, r: float, market_price: float,
                option: str) -> float:
    """ATM approximation as Newton-Raphson seed."""
    atm_approx = market_price / (S * np.sqrt(T / (2 * np.pi)))
    return np.clip(atm_approx, 0.01, 5.0)


# ── Newton-Raphson ────────────────────────────────────────────────────────────

def _newton(S: float, K: float, T: float, r: float, market_price: float,
            option: str) -> float | None:
    """
    Newton-Raphson iteration: σ_{n+1} = σ_n − (BS(σ_n) − market) / vega(σ_n).
    Returns None if it fails to converge.
    """
    sigma = _sigma_init(S, K, T, r, market_price, option)
    for _ in range(_MAX_ITER):
        p    = price(S, K, T, r, sigma, option)
        vega = greeks(S, K, T, r, sigma)["vega"] * 100   # undo the /100 scaling
        if abs(vega) < _TOL_VEGA:
            return None
        sigma -= (p - market_price) / vega
        sigma  = np.clip(sigma, _SIGMA_LO, _SIGMA_HI)
        if abs(price(S, K, T, r, sigma, option) - market_price) < _TOL_PRICE:
            return sigma
    return None


# ── public API ────────────────────────────────────────────────────────────────

def implied_vol(S: float, K: float, T: float, r: float,
                market_price: float, option: str = "call") -> float:
    """
    Compute implied volatility from a market option price.

    Uses Newton-Raphson first; falls back to Brent's method if NR diverges.

    Parameters
    ----------
    S            : underlying price
    K            : strike
    T            : time to expiry (years)
    r            : risk-free rate
    market_price : observed market price of the option
    option       : 'call' or 'put'

    Returns
    -------
    Implied volatility (annualised, e.g. 0.20 = 20%).

    Raises
    ------
    ValueError if no solution exists in [_SIGMA_LO, _SIGMA_HI].
    """
    disc = np.exp(-r * T)
    if option == "call":
        lower_bound = max(S - K * disc, 0.0)
    else:
        lower_bound = max(K * disc - S, 0.0)

    if market_price < lower_bound - _TOL_PRICE:
        raise ValueError(
            f"market_price={market_price:.4f} is below intrinsic value "
            f"{lower_bound:.4f} — arbitrage-free IV does not exist."
        )

    sigma = _newton(S, K, T, r, market_price, option)
    if sigma is not None:
        return float(sigma)

    f = lambda sig: price(S, K, T, r, sig, option) - market_price
    try:
        return float(brentq(f, _SIGMA_LO, _SIGMA_HI, xtol=_TOL_PRICE, maxiter=500))
    except ValueError:
        raise ValueError(
            f"Could not find IV for market_price={market_price:.4f}. "
            "Check inputs or whether the price is arbitrage-free."
        )


def iv_surface(S: float, strikes: list[float], maturities: list[float],
               r: float, market_prices: dict, option: str = "call") -> dict:
    """
    Compute a grid of implied vols.

    Parameters
    ----------
    S             : spot price
    strikes       : list of strikes K_i
    maturities    : list of maturities T_j (years)
    r             : risk-free rate
    market_prices : dict keyed (K, T) -> market_price
    option        : 'call' or 'put'

    Returns
    -------
    dict keyed (K, T) -> implied_vol  (None if computation fails)
    """
    surface = {}
    for K in strikes:
        for T in maturities:
            mp = market_prices.get((K, T))
            if mp is None:
                surface[(K, T)] = None
                continue
            try:
                surface[(K, T)] = implied_vol(S, K, T, r, mp, option)
            except ValueError:
                surface[(K, T)] = None
    return surface
