"""
options/greeks_fd.py
Generic finite-difference Greeks for pricers without a closed form
(binomial tree, Monte Carlo, Heston characteristic-function pricing).
"""

from __future__ import annotations

from typing import Callable

Pricer = Callable[[float, float, float, float, float, str], float]


def fd_greeks(
    pricer: Pricer,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option: str = "call",
    bump_s: float = 0.01,
    bump_vol: float = 0.01,
    bump_r: float = 0.0001,
) -> dict:
    """
    Central-difference delta/gamma/vega/rho, forward-difference theta (1 day),
    around a black-box `pricer(S, K, T, r, sigma, option) -> price` callable.
    """
    h_s = S * bump_s
    base = pricer(S, K, T, r, sigma, option)

    up_s = pricer(S + h_s, K, T, r, sigma, option)
    dn_s = pricer(S - h_s, K, T, r, sigma, option)
    delta = (up_s - dn_s) / (2 * h_s)
    gamma = (up_s - 2 * base + dn_s) / (h_s**2)

    up_vol = pricer(S, K, T, r, sigma + bump_vol, option)
    dn_vol = pricer(S, K, T, r, sigma - bump_vol, option)
    vega = (up_vol - dn_vol) / (2 * bump_vol)

    up_r = pricer(S, K, T, r + bump_r, sigma, option)
    dn_r = pricer(S, K, T, r - bump_r, sigma, option)
    rho = (up_r - dn_r) / (2 * bump_r)

    dt = 1.0 / 365.0
    theta_1d = pricer(S, K, max(T - dt, 1e-6), r, sigma, option) - base

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega) / 100,
        "rho": float(rho) / 100,
        "theta_1d": float(theta_1d),
    }
