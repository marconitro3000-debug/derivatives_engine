"""
volsurface/pricing/blackscholes.py
Analytical Black-Scholes pricing and Greeks for European options.
"""

import numpy as np
from scipy.stats import norm


# ── helpers ──────────────────────────────────────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes (with continuous dividend yield q)."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


# ── pricing ───────────────────────────────────────────────────────────────────

def price(S: float, K: float, T: float, r: float, sigma: float, option: str = "call", q: float = 0.0) -> float:
    """
    Black-Scholes closed-form price for a European option.

    Parameters
    ----------
    S      : current underlying price
    K      : strike price
    T      : time to expiry in years
    r      : continuously compounded risk-free rate
    sigma  : annualised volatility
    option : 'call' or 'put'
    q      : continuous dividend yield (default 0 — preserves prior behavior for
             callers that don't pass it)

    Returns
    -------
    Option price.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if option == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif option == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError("option must be 'call' or 'put'.")


def put_call_parity_check(S: float, K: float, T: float, r: float,
                           call_price: float, put_price: float, q: float = 0.0) -> dict:
    """Verify put-call parity: C - P = S*e^(-qT) - K*e^(-rT)."""
    lhs = call_price - put_price
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    return {"C - P": lhs, "S - Ke^(-rT)": rhs, "error": abs(lhs - rhs)}


# ── greeks ────────────────────────────────────────────────────────────────────

def greeks(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> dict:
    """
    Compute all standard Greeks for a European call (delta for put = delta_call - 1).

    Returns
    -------
    dict with keys: delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put
    All per-unit except:
      - vega  : per 1% change in sigma
      - theta : per calendar day
      - rho   : per 1% change in r
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrtT  = np.sqrt(T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    npd1   = norm.pdf(d1)

    delta_call = disc_q * norm.cdf(d1)
    delta_put  = disc_q * (norm.cdf(d1) - 1.0)
    gamma      = disc_q * npd1 / (S * sigma * sqrtT)
    vega       = S * disc_q * npd1 * sqrtT / 100.0            # per 1% σ
    theta_call = (-(S * disc_q * npd1 * sigma) / (2 * sqrtT)
                  - r * K * disc_r * norm.cdf(d2)
                  + q * S * disc_q * norm.cdf(d1)) / 365.0
    theta_put  = (-(S * disc_q * npd1 * sigma) / (2 * sqrtT)
                  + r * K * disc_r * norm.cdf(-d2)
                  - q * S * disc_q * norm.cdf(-d1)) / 365.0
    rho_call   =  K * T * disc_r * norm.cdf(d2)  / 100.0   # per 1% r
    rho_put    = -K * T * disc_r * norm.cdf(-d2) / 100.0

    return {
        "delta_call": delta_call,
        "delta_put":  delta_put,
        "gamma":      gamma,
        "vega":       vega,
        "theta_call": theta_call,
        "theta_put":  theta_put,
        "rho_call":   rho_call,
        "rho_put":    rho_put,
    }
