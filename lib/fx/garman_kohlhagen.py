"""
fx/garman_kohlhagen.py
Garman-Kohlhagen (1983) model for European FX options.

The GK model is Black-Scholes with a foreign risk-free rate playing the role
of a continuous dividend yield:

    d1 = [ln(S/K) + (r_d - r_f + ½σ²)T] / (σ√T)
    d2 = d1 - σ√T

    Call = S·e^{-r_f T}·N(d1) − K·e^{-r_d T}·N(d2)
    Put  = K·e^{-r_d T}·N(−d2) − S·e^{-r_f T}·N(−d1)

FX forward: F = S·e^{(r_d − r_f)T}   (covered interest parity)

Notation:
    S    — spot exchange rate (domestic per 1 unit of foreign, e.g. USD/EUR = 1.08)
    K    — strike
    r_d  — domestic risk-free rate (continuously compounded)
    r_f  — foreign risk-free rate  (continuously compounded)
    sigma — FX vol (annualized)
    T    — years to expiry
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass

_N   = norm.cdf
_phi = norm.pdf


# ── core pricer ───────────────────────────────────────────────────────────────

def fx_forward(S: float, r_d: float, r_f: float, T: float) -> float:
    """FX forward rate: F = S · e^{(r_d − r_f)T}"""
    return S * np.exp((r_d - r_f) * T)


def _d1d2(S, K, r_d, r_f, sigma, T):
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def price_gk(S: float, K: float, T: float, r_d: float, r_f: float,
             sigma: float, option_type: str = "call",
             notional: float = 1_000_000.0) -> dict:
    """
    Garman-Kohlhagen option price.

    Parameters
    ----------
    S, K     : spot and strike (same quote convention, e.g. USD per EUR)
    T        : years to expiry
    r_d      : domestic risk-free rate (continuously compounded)
    r_f      : foreign risk-free rate
    sigma    : implied vol (annualized, e.g. 0.08 = 8%)
    option_type : "call" | "put"
    notional : foreign currency notional (default 1M)

    Returns
    -------
    dict with price, greeks, d1, d2, forward
    """
    if sigma <= 0 or T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return {"price": intrinsic * np.exp(-r_d * T) * notional,
                "forward": fx_forward(S, r_d, r_f, T)}

    sqrtT = np.sqrt(T)
    d1, d2 = _d1d2(S, K, r_d, r_f, sigma, T)
    df_d = np.exp(-r_d * T)
    df_f = np.exp(-r_f * T)

    if option_type == "call":
        px    = S * df_f * _N(d1) - K * df_d * _N(d2)
        delta = df_f * _N(d1)
    else:
        px    = K * df_d * _N(-d2) - S * df_f * _N(-d1)
        delta = -df_f * _N(-d1)

    gamma = df_f * _phi(d1) / (S * sigma * sqrtT)
    vega  = S * df_f * _phi(d1) * sqrtT / 100.0   # per 1% vol
    theta = (-(S * df_f * _phi(d1) * sigma / (2 * sqrtT))
             - r_d * K * df_d * (_N(d2) if option_type == "call" else _N(-d2))
             + r_f * S * df_f * (_N(d1) if option_type == "call" else _N(-d1))
             ) / 365.0   # per calendar day
    rho_d = (K * T * df_d * (_N(d2) if option_type == "call" else -_N(-d2))) / 100.0
    rho_f = (-S * T * df_f * (_N(d1) if option_type == "call" else -_N(-d1))) / 100.0

    # Vanna: dDelta/dVol = -df_f * d2 / sigma * phi(d1)
    vanna  = -df_f * _phi(d1) * d2 / sigma
    # Volga: d²Price/dVol² = S * df_f * phi(d1) * sqrt(T) * d1 * d2 / sigma
    volga  = S * df_f * _phi(d1) * sqrtT * d1 * d2 / sigma

    return {
        "price":   float(px * notional),
        "unit_px": float(px),
        "delta":   float(delta),
        "gamma":   float(gamma),
        "vega":    float(vega * notional),
        "theta":   float(theta * notional),
        "rho_d":   float(rho_d * notional),
        "rho_f":   float(rho_f * notional),
        "vanna":   float(vanna),
        "volga":   float(volga),
        "d1":      float(d1),
        "d2":      float(d2),
        "forward": float(fx_forward(S, r_d, r_f, T)),
    }


# ── put-call parity ───────────────────────────────────────────────────────────

def put_call_parity_check(S: float, K: float, T: float,
                           r_d: float, r_f: float, sigma: float) -> dict:
    """
    Verify: Call − Put = S·e^{−r_f T} − K·e^{−r_d T}

    Returns dict with call, put, lhs (C-P), rhs (fwd_pv), error.
    """
    c = price_gk(S, K, T, r_d, r_f, sigma, "call", 1.0)["unit_px"]
    p = price_gk(S, K, T, r_d, r_f, sigma, "put",  1.0)["unit_px"]
    lhs = c - p
    rhs = S * np.exp(-r_f * T) - K * np.exp(-r_d * T)
    return {"call": c, "put": p, "lhs": lhs, "rhs": rhs,
            "error": float(abs(lhs - rhs))}


# ── implied vol ───────────────────────────────────────────────────────────────

def implied_vol_gk(market_price: float, S: float, K: float, T: float,
                    r_d: float, r_f: float, option_type: str = "call",
                    notional: float = 1_000_000.0) -> float:
    """
    Garman-Kohlhagen implied vol via Brent root-finding.

    Returns implied vol (annualized, decimal).
    """
    def obj(sigma):
        return price_gk(S, K, T, r_d, r_f, sigma, option_type, notional)["price"] - market_price

    # Bounds check
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    intrinsic *= np.exp(-r_d * T) * notional
    if market_price <= intrinsic:
        return 0.0

    return float(brentq(obj, 1e-4, 5.0, xtol=1e-8, maxiter=200))


# ── delta → strike mapping ────────────────────────────────────────────────────

def delta_to_strike(delta: float, S: float, T: float, r_d: float, r_f: float,
                     sigma: float, option_type: str = "call",
                     delta_convention: str = "spot") -> float:
    """
    Convert an FX delta (e.g. 0.25 = 25-delta call) to a strike.

    delta_convention: "spot" (default) or "forward"
    For call:  Δ = e^{-r_f T} N(d1)  →  K = S exp(-d1 σ√T + (r_d-r_f+½σ²)T)
    """
    df_f  = np.exp(-r_f * T) if delta_convention == "spot" else 1.0
    sqrtT = np.sqrt(T)
    if option_type == "call":
        d1 = norm.ppf(delta / df_f)
    else:
        # delta passed as positive absolute value (25P → delta=0.25)
        # Δ_put = -df_f·N(-d1) = -|Δ|  →  d1 = -N⁻¹(|Δ|/df_f)
        d1 = -norm.ppf(delta / df_f)
    K = S * np.exp(-d1 * sigma * sqrtT + (r_d - r_f + 0.5 * sigma**2) * T)
    return float(K)


def strike_to_delta(K: float, S: float, T: float, r_d: float, r_f: float,
                     sigma: float, option_type: str = "call") -> float:
    """Spot delta for a given strike."""
    d1, _ = _d1d2(S, K, r_d, r_f, sigma, T)
    df_f  = np.exp(-r_f * T)
    if option_type == "call":
        return float(df_f * _N(d1))
    return float(-df_f * _N(-d1))


# ── ATM strike conventions ────────────────────────────────────────────────────

def atm_dns_strike(S: float, T: float, r_d: float, r_f: float,
                    sigma: float) -> float:
    """
    ATM Delta-Neutral Straddle (DNS) strike: call delta + put delta = 0.
    K_DNS = F · e^{+½σ²T}
    """
    F = fx_forward(S, r_d, r_f, T)
    return float(F * np.exp(0.5 * sigma**2 * T))


def atm_forward_strike(S: float, T: float, r_d: float, r_f: float) -> float:
    """ATM forward: K = F."""
    return fx_forward(S, r_d, r_f, T)
