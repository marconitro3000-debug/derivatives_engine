"""
rates/capfloor.py
Interest rate caps, floors, and collars using Black's model on forward rates.

A cap is a strip of caplets. Each caplet pays:
    max(L(tᵢ, tᵢ₊₁) − K, 0) × Δtᵢ × Notional   at time tᵢ₊₁

Under Black's model the forward rate L(tᵢ, tᵢ₊₁) is log-normal:
    Caplet(i) = P(0, tᵢ₊₁) × Δtᵢ × [F_i N(d₁) − K N(d₂)]
    d₁ = (ln(F_i/K) + ½σ²tᵢ) / (σ√tᵢ)
    d₂ = d₁ − σ√tᵢ

Floorlet(i) = P(0, tᵢ₊₁) × Δtᵢ × [K N(−d₂) − F_i N(−d₁)]

Cap-floor parity (same vol σ, same K):
    Cap − Floor = PV(Floating leg) − PV(Fixed leg) = Swap value

A collar is: long cap at K_cap + short floor at K_floor.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

_Phi = norm.cdf
_phi = norm.pdf


# ── forward rates ─────────────────────────────────────────────────────────────

def forward_libor(discount_curve, t_start: float, t_end: float) -> float:
    """
    Simply-compounded forward rate for period [t_start, t_end].
    F(t_s, t_e) = (P(t_s)/P(t_e) − 1) / (t_e − t_s)
    """
    dt   = t_end - t_start
    P_s  = discount_curve.discount_factor(t_start)
    P_e  = discount_curve.discount_factor(t_end)
    return float((P_s / P_e - 1.0) / dt)


# ── caplet / floorlet ─────────────────────────────────────────────────────────

def caplet(discount_curve, t_fix: float, t_pay: float,
           strike: float, sigma: float,
           notional: float = 1_000_000.0) -> float:
    """
    Single caplet: fixes at t_fix, pays at t_pay = t_fix + Δt.

    Parameters
    ----------
    t_fix  : rate fixing date (years)
    t_pay  : payment date (years); Δt = t_pay − t_fix
    strike : cap rate
    sigma  : Black vol (log-normal, flat for simplicity)
    """
    dt   = t_pay - t_fix
    F    = forward_libor(discount_curve, t_fix, t_pay)
    P    = discount_curve.discount_factor(t_pay)
    sqrtT= np.sqrt(t_fix)

    if sigma <= 0 or t_fix <= 0 or F <= 0:
        return float(P * dt * max(F - strike, 0.0) * notional)

    d1 = (np.log(F / strike) + 0.5 * sigma ** 2 * t_fix) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return float(notional * P * dt * (F * _Phi(d1) - strike * _Phi(d2)))


def floorlet(discount_curve, t_fix: float, t_pay: float,
             strike: float, sigma: float,
             notional: float = 1_000_000.0) -> float:
    """Single floorlet: pays max(K − L, 0) × Δt at t_pay."""
    dt   = t_pay - t_fix
    F    = forward_libor(discount_curve, t_fix, t_pay)
    P    = discount_curve.discount_factor(t_pay)
    sqrtT= np.sqrt(t_fix)

    if sigma <= 0 or t_fix <= 0:
        return float(P * dt * max(strike - F, 0.0) * notional)

    if F <= 0:
        return float(P * dt * max(strike - F, 0.0) * notional)

    d1 = (np.log(F / strike) + 0.5 * sigma ** 2 * t_fix) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return float(notional * P * dt * (strike * _Phi(-d2) - F * _Phi(-d1)))


# ── cap / floor ───────────────────────────────────────────────────────────────

def cap(discount_curve, maturity: float, strike: float,
        sigma: float | np.ndarray,
        start: float = 0.0, pay_freq: int = 4,
        notional: float = 1_000_000.0) -> dict:
    """
    Interest rate cap: sum of caplets from `start` to `maturity`.

    Parameters
    ----------
    sigma : scalar (flat vol) or array of per-caplet vols
    start : cap start date (first reset usually at 0 or in the future)

    Returns
    -------
    dict: price, caplet_prices, forward_rates, schedule
    """
    dt        = 1.0 / pay_freq
    schedule  = [(round(t - dt, 8), round(t, 8))
                 for t in np.arange(start + dt, maturity + 1e-9, dt)
                 if t - dt >= 0]

    if isinstance(sigma, (int, float)):
        vols = [float(sigma)] * len(schedule)
    else:
        vols = list(sigma)
        if len(vols) < len(schedule):
            vols += [vols[-1]] * (len(schedule) - len(vols))

    caplet_prices = []
    fwd_rates     = []
    for (t_fix, t_pay), vol in zip(schedule, vols):
        if t_fix <= 0:
            t_eff = max(t_fix, 1e-6)
        else:
            t_eff = t_fix
        cp = caplet(discount_curve, t_eff, t_pay, strike, vol, notional)
        caplet_prices.append(cp)
        fwd_rates.append(forward_libor(discount_curve, t_eff, t_pay))

    return {
        "price":         float(sum(caplet_prices)),
        "caplet_prices": caplet_prices,
        "forward_rates": fwd_rates,
        "schedule":      schedule,
        "n_caplets":     len(schedule),
    }


def floor(discount_curve, maturity: float, strike: float,
          sigma: float | np.ndarray,
          start: float = 0.0, pay_freq: int = 4,
          notional: float = 1_000_000.0) -> dict:
    """
    Interest rate floor: sum of floorlets.
    """
    dt       = 1.0 / pay_freq
    schedule = [(round(t - dt, 8), round(t, 8))
                for t in np.arange(start + dt, maturity + 1e-9, dt)
                if t - dt >= 0]

    if isinstance(sigma, (int, float)):
        vols = [float(sigma)] * len(schedule)
    else:
        vols = list(sigma)
        if len(vols) < len(schedule):
            vols += [vols[-1]] * (len(schedule) - len(vols))

    floorlet_prices = []
    fwd_rates       = []
    for (t_fix, t_pay), vol in zip(schedule, vols):
        t_eff = max(t_fix, 1e-6)
        fp    = floorlet(discount_curve, t_eff, t_pay, strike, vol, notional)
        floorlet_prices.append(fp)
        fwd_rates.append(forward_libor(discount_curve, t_eff, t_pay))

    return {
        "price":           float(sum(floorlet_prices)),
        "floorlet_prices": floorlet_prices,
        "forward_rates":   fwd_rates,
        "schedule":        schedule,
        "n_floorlets":     len(schedule),
    }


def collar(discount_curve, maturity: float,
           cap_strike: float, floor_strike: float,
           sigma_cap: float, sigma_floor: float,
           start: float = 0.0, pay_freq: int = 4,
           notional: float = 1_000_000.0) -> dict:
    """
    Collar = Long Cap(K_cap) − Short Floor(K_floor).

    Typically K_floor < K_cap. A zero-cost collar is when cap premium = floor premium.
    """
    cap_res   = cap(discount_curve, maturity, cap_strike, sigma_cap,
                    start, pay_freq, notional)
    floor_res = floor(discount_curve, maturity, floor_strike, sigma_floor,
                      start, pay_freq, notional)
    net_cost  = cap_res["price"] - floor_res["price"]

    return {
        "net_cost":   float(net_cost),
        "cap_price":  cap_res["price"],
        "floor_price":floor_res["price"],
        "cap":        cap_res,
        "floor":      floor_res,
    }


# ── parity and implied vol ────────────────────────────────────────────────────

def cap_floor_parity_check(discount_curve, maturity: float, strike: float,
                            sigma: float, start: float = 0.0,
                            pay_freq: int = 4,
                            notional: float = 1_000_000.0) -> dict:
    """
    Cap − Floor = Swap (floating − fixed) for same K, σ, notional.

    Returns dict with cap price, floor price, difference, and swap PV.
    """
    cap_price   = cap(discount_curve, maturity, strike, sigma,
                      start, pay_freq, notional)["price"]
    floor_price = floor(discount_curve, maturity, strike, sigma,
                        start, pay_freq, notional)["price"]

    # PV of floating leg: P(start) − P(maturity)
    P_start = discount_curve.discount_factor(max(start, 1e-9))
    P_end   = discount_curve.discount_factor(maturity)
    pv_float = notional * (P_start - P_end)

    # PV of fixed leg
    dt        = 1.0 / pay_freq
    pay_times = np.arange(start + dt, maturity + 1e-9, dt)
    pv_fixed  = notional * strike * sum(dt * discount_curve.discount_factor(t)
                                         for t in pay_times)
    swap_pv   = pv_float - pv_fixed

    return {
        "cap_price":      cap_price,
        "floor_price":    floor_price,
        "cap_minus_floor":cap_price - floor_price,
        "swap_pv":        swap_pv,
        "parity_error":   abs((cap_price - floor_price) - swap_pv),
    }


def implied_cap_vol(market_price: float, discount_curve,
                    maturity: float, strike: float,
                    instrument: str = "cap",
                    start: float = 0.0, pay_freq: int = 4,
                    notional: float = 1_000_000.0) -> float:
    """
    Black implied vol from market cap or floor price.
    """
    pricer = cap if instrument == "cap" else floor

    def obj(sigma):
        return pricer(discount_curve, maturity, strike, sigma,
                      start, pay_freq, notional)["price"] - market_price

    return float(brentq(obj, 1e-6, 5.0, xtol=1e-8, maxiter=200))
