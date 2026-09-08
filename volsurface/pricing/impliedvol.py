"""
volsurface/impliedvol.py
Implied volatility extraction via Newton-Raphson (primary) with Brent fallback.
"""

import numpy as np
from scipy.optimize import brentq

from .blackscholes import price, greeks


# ── constants ─────────────────────────────────────────────────────────────────

_MAX_ITER   = 100
_TOL_SIGMA  = 1e-10     # convergence is judged on sigma, not on price (see below)
_TOL_PRICE  = 1e-8      # arbitrage-bound slack only
_TOL_VEGA   = 1e-12
_SIGMA_LO   = 1e-6
_SIGMA_HI   = 10.0      # 1000% vol ceiling
_MIN_VEGA   = 1e-6      # below this the price does not identify a volatility


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

    Convergence is declared on the *step in σ*, not on the price residual. Those
    are not equivalent: a deep in- or out-of-the-money option has vega near zero,
    so a wide band of volatilities reprices it to within any absolute price
    tolerance, and a price-based stop happily returns whichever σ the iteration
    happened to be standing on. Stopping when σ itself stops moving is the
    condition that actually means the root has been located.

    This is also why `volsurface.data` fits only out-of-the-money quotes: no
    stopping rule can recover a volatility the price does not encode, and for a
    deep ITM option it barely does.
    """
    sigma = _sigma_init(S, K, T, r, market_price, option)
    for _ in range(_MAX_ITER):
        p    = price(S, K, T, r, sigma, option)
        vega = greeks(S, K, T, r, sigma)["vega"] * 100   # undo the /100 scaling
        if abs(vega) < _TOL_VEGA:
            return None
        step   = (p - market_price) / vega
        sigma  = float(np.clip(sigma - step, _SIGMA_LO, _SIGMA_HI))
        if abs(step) < _TOL_SIGMA:
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
    ValueError
        If the price is below intrinsic value (no arbitrage-free IV exists), if
        no root is found in [_SIGMA_LO, _SIGMA_HI], or if the option's vega is so
        small that the price does not identify a volatility at all -- see the
        note at the end of this function.
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
    if sigma is None:
        f = lambda sig: price(S, K, T, r, sig, option) - market_price
        try:
            sigma = float(brentq(f, _SIGMA_LO, _SIGMA_HI, xtol=_TOL_PRICE, maxiter=500))
        except ValueError:
            raise ValueError(
                f"Could not find IV for market_price={market_price:.4f}. "
                "Check inputs or whether the price is arbitrage-free."
            )

    # Identifiability check. Deep in-the-money options are worth their intrinsic
    # value to the last bit of a float64 across a wide band of volatilities: at
    # S=100, K=70, T=0.6, every sigma below ~7% produces the *same* double. Any
    # root finder will return a number there, and that number is meaningless.
    # Refusing is the only honest answer -- a silently wrong IV propagates into
    # the surface fit as a real data point.
    vega = greeks(S, K, T, r, sigma)["vega"] * 100.0
    if vega < _MIN_VEGA:
        raise ValueError(
            f"implied vol is not identifiable at K={K:g}, T={T:g}: vega={vega:.2e} "
            f"means the price does not distinguish volatilities. Use an "
            f"out-of-the-money quote for this strike."
        )
    return float(sigma)


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
