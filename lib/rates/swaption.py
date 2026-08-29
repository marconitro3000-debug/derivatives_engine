"""
rates/swaption.py
Swaption pricing via Black's model (log-normal forward swap rate) and
Bachelier model (normal / displaced-diffusion).

A payer swaption gives the right to enter a pay-fixed / receive-floating
swap at rate K starting at expiry T_exp and running for `tenor` years.

Black's model (market convention for swaptions):
    V_payer   = A × [F_s N(d₁) − K N(d₂)]
    V_receiver = A × [K N(−d₂) − F_s N(−d₁)]
    d₁ = (ln(F_s/K) + ½σ²T) / (σ√T)
    d₂ = d₁ − σ√T

Bachelier / Normal model (preferred when rates are near zero or negative):
    V_payer   = A × [(F_s−K) N(d_N) + σ_N√T · φ(d_N)]
    d_N = (F_s − K) / (σ_N√T)

Forward swap rate and annuity:
    F_s = (P(0, T_exp) − P(0, T_N)) / A
    A   = Σᵢ Δtᵢ · P(0, tᵢ)           (annuity / PV01)

    where tᵢ = T_exp + i/pay_freq, T_N = T_exp + tenor
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

_Phi = norm.cdf
_phi = norm.pdf


# ── building blocks ───────────────────────────────────────────────────────────

def forward_swap_rate(discount_curve, T_exp: float, tenor: float,
                       pay_freq: int = 2) -> float:
    """
    Fair forward swap rate for a swap starting at T_exp, tenor years long.

    F_s = [P(0, T_exp) − P(0, T_exp + tenor)] / A
    A   = Σᵢ (1/pay_freq) · P(0, T_exp + i/pay_freq)
    """
    dt      = 1.0 / pay_freq
    T_N     = T_exp + tenor
    pay_times = np.arange(T_exp + dt, T_N + 1e-9, dt)

    annuity = sum(dt * discount_curve.discount_factor(t) for t in pay_times)
    P_exp   = discount_curve.discount_factor(T_exp)
    P_N     = discount_curve.discount_factor(T_N)

    return float((P_exp - P_N) / annuity), float(annuity)


def annuity(discount_curve, T_exp: float, tenor: float,
            pay_freq: int = 2) -> float:
    """Swap annuity A = Σᵢ Δtᵢ · P(0, tᵢ)."""
    _, A = forward_swap_rate(discount_curve, T_exp, tenor, pay_freq)
    return A


# ── Black's model ─────────────────────────────────────────────────────────────

def price_swaption_black(discount_curve, T_exp: float, tenor: float,
                          strike: float, sigma: float,
                          swaption_type: str = "payer",
                          pay_freq: int = 2,
                          notional: float = 1_000_000.0) -> dict:
    """
    Swaption price under Black's (log-normal) model.

    Parameters
    ----------
    discount_curve : DiscountCurve
    T_exp   : option expiry in years
    tenor   : underlying swap tenor in years
    strike  : fixed rate (e.g. 0.03 = 3%)
    sigma   : Black vol (log-normal, e.g. 0.20 = 20%)
    swaption_type : 'payer' or 'receiver'
    pay_freq: fixed-leg payment frequency (2 = semi-annual)
    notional: swap notional

    Returns
    -------
    dict: price, forward_swap_rate, annuity, d1, d2, delta_rate, vega
    """
    F, A = forward_swap_rate(discount_curve, T_exp, tenor, pay_freq)
    sqrtT = np.sqrt(T_exp)

    if sigma <= 0 or T_exp <= 0:
        # Intrinsic value
        if swaption_type == "payer":
            px = notional * A * max(F - strike, 0.0)
        else:
            px = notional * A * max(strike - F, 0.0)
        return {"price": px, "forward_swap_rate": F, "annuity": A,
                "d1": np.nan, "d2": np.nan}

    d1 = (np.log(F / strike) + 0.5 * sigma ** 2 * T_exp) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if swaption_type == "payer":
        px = notional * A * (F * _Phi(d1) - strike * _Phi(d2))
    else:
        px = notional * A * (strike * _Phi(-d2) - F * _Phi(-d1))

    # Greeks
    vega       = notional * A * F * _phi(d1) * sqrtT
    delta_rate = notional * A * (_Phi(d1) if swaption_type == "payer"
                                 else -_Phi(-d1))

    return {
        "price":            float(px),
        "forward_swap_rate": float(F),
        "annuity":          float(A),
        "d1":               float(d1),
        "d2":               float(d2),
        "vega":             float(vega),
        "delta_rate":       float(delta_rate),
        "dv01":             float(notional * A * 0.0001),
    }


# ── Bachelier / Normal model ──────────────────────────────────────────────────

def price_swaption_bachelier(discount_curve, T_exp: float, tenor: float,
                              strike: float, sigma_n: float,
                              swaption_type: str = "payer",
                              pay_freq: int = 2,
                              notional: float = 1_000_000.0) -> dict:
    """
    Swaption price under the Bachelier (normal) model.

    Preferred when rates are near zero or can go negative.

    V_payer = A × [(F−K) N(d_N) + σ_N√T · φ(d_N)]
    """
    F, A  = forward_swap_rate(discount_curve, T_exp, tenor, pay_freq)
    sqrtT = np.sqrt(T_exp)

    if sigma_n <= 0 or T_exp <= 0:
        if swaption_type == "payer":
            px = notional * A * max(F - strike, 0.0)
        else:
            px = notional * A * max(strike - F, 0.0)
        return {"price": px, "forward_swap_rate": F, "annuity": A}

    d_N  = (F - strike) / (sigma_n * sqrtT)

    # Bachelier parity: V_payer - V_receiver = A × (F - K)
    # V_payer   = A × [(F-K) Φ(d_N)  + σ_N√T φ(d_N)]
    # V_receiver = A × [(K-F) Φ(-d_N) + σ_N√T φ(d_N)]
    if swaption_type == "payer":
        px = notional * A * ((F - strike) * _Phi(d_N) + sigma_n * sqrtT * _phi(d_N))
    else:
        px = notional * A * ((strike - F) * _Phi(-d_N) + sigma_n * sqrtT * _phi(d_N))

    vega = notional * A * sqrtT * _phi(d_N)

    return {
        "price":            float(px),
        "forward_swap_rate": float(F),
        "annuity":          float(A),
        "d_N":              float(d_N),
        "vega":             float(vega),
    }


# ── implied vol ───────────────────────────────────────────────────────────────

def implied_black_vol(market_price: float, discount_curve,
                       T_exp: float, tenor: float, strike: float,
                       swaption_type: str = "payer",
                       pay_freq: int = 2,
                       notional: float = 1_000_000.0) -> float:
    """
    Black implied vol from market swaption price.

    Uses Brent root-finding on price_swaption_black(sigma) = market_price.
    """
    def obj(sigma):
        px = price_swaption_black(discount_curve, T_exp, tenor, strike,
                                   sigma, swaption_type, pay_freq, notional)
        return px["price"] - market_price

    # Check if market price within bounds
    F, A = forward_swap_rate(discount_curve, T_exp, tenor, pay_freq)
    intrinsic = notional * A * max(F - strike, 0.0) if swaption_type == "payer" \
                else notional * A * max(strike - F, 0.0)
    if market_price <= intrinsic:
        return 0.0

    return float(brentq(obj, 1e-6, 5.0, xtol=1e-8, maxiter=200))


# ── swaption surface ──────────────────────────────────────────────────────────

def swaption_grid(discount_curve,
                   expiries: list[float], tenors: list[float],
                   sigma_grid: np.ndarray,
                   strike_offset: float = 0.0,
                   swaption_type: str = "payer",
                   pay_freq: int = 2,
                   notional: float = 1_000_000.0) -> list[dict]:
    """
    Price a grid of swaptions (expiry × tenor).

    sigma_grid : 2-D array of shape (len(expiries), len(tenors))
    strike_offset : ATM + offset (e.g. 0.0 = ATM, 0.01 = ATM+100bps)

    Returns list of dicts sorted by (expiry, tenor).
    """
    results = []
    for i, T_exp in enumerate(expiries):
        for j, tenor in enumerate(tenors):
            F, _ = forward_swap_rate(discount_curve, T_exp, tenor, pay_freq)
            K    = F + strike_offset
            sigma = float(sigma_grid[i, j])
            res   = price_swaption_black(discount_curve, T_exp, tenor, K, sigma,
                                          swaption_type, pay_freq, notional)
            res["expiry"] = T_exp
            res["tenor"]  = tenor
            res["strike"] = K
            res["sigma"]  = sigma
            results.append(res)
    return results
