"""
volatility/variance_swap.py
Variance swap pricing and volatility risk premium (VRP).

A variance swap pays the difference between realized variance and a fixed
strike K²_var at expiry:

    Payoff = N × (σ²_realized − K²_var)

Fair variance strike (model-free, Demeterfi et al. 1999):
    K²_var = (2/T) × e^{rT} × [Σ_{K<F} P(K)/K² ΔK + Σ_{K≥F} C(K)/K² ΔK]

For a flat BS vol σ, this reduces to K²_var = σ² exactly.

Volatility Risk Premium (VRP):
    VRP = IV − RV   (implied minus realized vol, same tenor)
    Typically negative: implied vol > realized vol on average
    ("sellers of vol are compensated for providing insurance")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


# ── fair variance strike ──────────────────────────────────────────────────────

def fair_variance_strike_bs(sigma: float, T: float) -> float:
    """
    Fair variance strike under flat BS vol.
    Under any log-normal model: K²_var = σ²  (exact).

    Returns variance strike in decimal² (e.g. 0.04 = 20%² vol).
    """
    return float(sigma ** 2)


def fair_variance_strike_mf(strikes: np.ndarray,
                             call_prices: np.ndarray,
                             put_prices: np.ndarray,
                             F: float, T: float, r: float = 0.0) -> dict:
    """
    Model-free fair variance strike (Demeterfi-Derman-Kamal-Zou 1999).

    K²_var = (2/T) × e^{rT} × Σᵢ ΔKᵢ × OTM_i(Kᵢ) / Kᵢ²

    where OTM_i is the out-of-the-money option price:
      - Put if Kᵢ ≤ F  (OTM put)
      - Call if Kᵢ > F  (OTM call)

    Parameters
    ----------
    strikes     : sorted array of strike prices
    call_prices : call prices at each strike
    put_prices  : put prices at each strike
    F           : forward price (= S × e^{rT})
    T           : maturity in years
    r           : risk-free rate (for discounting; default 0)

    Returns
    -------
    dict: 'fair_var', 'fair_vol', 'put_contribution', 'call_contribution'
    """
    K    = np.asarray(strikes,     dtype=float)
    C    = np.asarray(call_prices, dtype=float)
    P    = np.asarray(put_prices,  dtype=float)

    # OTM option at each strike
    otm  = np.where(K <= F, P, C)

    # Trapezoidal widths
    dK   = np.zeros_like(K)
    if len(K) > 1:
        dK[0]    = K[1]  - K[0]
        dK[-1]   = K[-1] - K[-2]
        dK[1:-1] = (K[2:] - K[:-2]) / 2.0

    weights   = dK / K ** 2
    total     = np.dot(weights, otm)

    put_mask  = K <= F
    put_cont  = float(np.dot(weights[put_mask],  otm[put_mask]))
    call_cont = float(np.dot(weights[~put_mask], otm[~put_mask]))

    fair_var  = (2.0 / T) * np.exp(r * T) * total
    fair_vol  = float(np.sqrt(max(fair_var, 0.0)))

    return {
        "fair_var":          float(fair_var),
        "fair_vol":          float(fair_vol),
        "put_contribution":  put_cont,
        "call_contribution": call_cont,
    }


# ── variance swap MtM ─────────────────────────────────────────────────────────

def variance_swap_pv(realized_var_so_far: float,
                     current_fair_var: float,
                     K_var: float,
                     t_elapsed: float,
                     T_total: float,
                     r: float = 0.0,
                     notional: float = 1_000_000,
                     vega_notional: bool = True) -> dict:
    """
    Mark-to-market of a long variance swap during its life.

    At time t (0 < t < T), the MtM value is:
        V(t) = N × e^{-r(T-t)} × [
            (t/T) × σ²_realized(0,t)
          + (T-t)/T × K²_var_current(t,T)
          − K²_var
        ]

    If vega_notional=True, `notional` is in vega terms ($ per 1% vol move),
    and the variance notional is derived as: N_var = N_vega / (2 × K_vol)
    where K_vol = sqrt(K_var).

    Parameters
    ----------
    realized_var_so_far : annualized realized variance from trade inception to now
    current_fair_var    : current fair variance strike for remaining life
    K_var               : original variance strike (at trade inception)
    t_elapsed           : fraction of life elapsed (0 to T)
    T_total             : total swap life (years)
    r                   : risk-free rate
    notional            : vega notional ($) or variance notional
    vega_notional       : if True, notional is in vega $ per 1% vol

    Returns
    -------
    dict: 'value', 'locked_var', 'remaining_var', 'var_pnl'
    """
    t = t_elapsed
    T = T_total
    w_past   = t / T
    w_future = (T - t) / T

    # Blended "current realized" expectation
    locked_var    = w_past * realized_var_so_far + w_future * current_fair_var
    var_pnl       = locked_var - K_var

    # Variance notional
    if vega_notional:
        K_vol      = np.sqrt(K_var)
        N_var      = notional / (2.0 * K_vol * 100)   # vega notional → var notional
    else:
        N_var = notional

    value = N_var * np.exp(-r * (T - t)) * var_pnl

    return {
        "value":         float(value),
        "locked_var":    float(locked_var),
        "locked_vol":    float(np.sqrt(max(locked_var, 0))),
        "remaining_var": float(current_fair_var),
        "var_pnl":       float(var_pnl),
        "vol_pnl":       float(np.sqrt(max(locked_var, 0)) - np.sqrt(max(K_var, 0))),
    }


# ── volatility risk premium ───────────────────────────────────────────────────

def vrp(implied_vol: pd.Series, realized_vol: pd.Series) -> pd.Series:
    """
    Volatility Risk Premium: VRP = IV − RV

    Both must be annualized and aligned on the same index.
    Positive VRP: implied > realized (most common, market pays for insurance).
    Negative VRP: market is "too cheap" relative to realized (rare).

    Returns a pd.Series of VRP values (same units as inputs, e.g. decimal vol).
    """
    return (implied_vol - realized_vol).rename("VRP")


def vrp_summary(vrp_series: pd.Series) -> dict:
    """Statistics on a VRP time series."""
    v = vrp_series.dropna()
    return {
        "mean":      float(v.mean()),
        "std":       float(v.std()),
        "median":    float(v.median()),
        "pct_pos":   float((v > 0).mean()),
        "sharpe":    float(v.mean() / v.std()) if v.std() > 0 else np.nan,
        "min":       float(v.min()),
        "max":       float(v.max()),
    }


def realized_variance_from_prices(close: pd.Series,
                                   window: int = 21) -> pd.Series:
    """
    Annualized realized variance over rolling `window` days.
    σ²_rv = (252/n) × Σ r²_t
    """
    r = np.log(close / close.shift(1))
    return (r ** 2).rolling(window, min_periods=window // 2).sum() * (_TRADING_DAYS / window)
