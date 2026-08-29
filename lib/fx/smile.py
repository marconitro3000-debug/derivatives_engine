"""
fx/smile.py
FX volatility smile conventions (25-delta / 10-delta market quotes).

FX markets quote the smile via three instruments:
  ATM    — At-the-money straddle vol σ_ATM
  RR25   — 25-delta risk reversal: RR = σ_25C − σ_25P  (tilt)
  BF25   — 25-delta butterfly/strangle: BF = ½(σ_25C + σ_25P) − σ_ATM  (curvature)

Reconstruction:
  σ_25C = σ_ATM + BF + ½·RR
  σ_25P = σ_ATM + BF − ½·RR

Vanna-Volga (VV) pricing for FX exotics:
The VV method adds a hedge cost to the BS price to account for the market's
smile. For an exotic with vanna Λ_v and volga Λ_g:

  P_VV ≈ P_BS + p₁(Λ_v/Λ_v1)·(V_1−BS_1) + p₂(Λ_g/Λ_g2)·(V_2−BS_2)

where (V_i − BS_i) is the overhedge cost of the ith vega instrument.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq

from .garman_kohlhagen import (
    price_gk, implied_vol_gk, delta_to_strike, strike_to_delta,
    atm_dns_strike, fx_forward, _d1d2,
    _N, _phi,
)


@dataclass
class FXSmileQuotes:
    """Market quotes for FX vol smile at one expiry."""
    S:       float    # spot
    T:       float    # years to expiry
    r_d:     float    # domestic rate
    r_f:     float    # foreign rate
    atm:     float    # ATM DNS straddle vol
    rr25:    float    # 25-delta risk reversal (positive = upside skew)
    bf25:    float    # 25-delta butterfly (convexity premium)
    rr10:    float = 0.0   # 10-delta risk reversal (optional)
    bf10:    float = 0.0   # 10-delta butterfly (optional)


@dataclass
class FXSmile:
    """Fitted FX smile: 5 strikes with vols."""
    S: float; T: float; r_d: float; r_f: float
    strikes:   np.ndarray   # [K_10P, K_25P, K_ATM, K_25C, K_10C]
    vols:      np.ndarray   # corresponding implied vols
    labels:    list[str]
    quotes:    FXSmileQuotes

    def vol_at_strike(self, K: float) -> float:
        """Linear interpolation on strike grid."""
        return float(np.interp(K, self.strikes, self.vols))

    def vol_at_delta(self, delta: float, option_type: str = "call") -> float:
        """Interpolated vol at a given delta level."""
        K = delta_to_strike(delta, self.S, self.T, self.r_d, self.r_f,
                             self.vol_at_strike(self.S), option_type)
        return self.vol_at_strike(K)


def build_smile(q: FXSmileQuotes) -> FXSmile:
    """
    Build a 5-point (or 3-point) smile from ATM/RR/BF market quotes.

    Procedure:
    1. Recover individual vols from market quotes
    2. Convert each delta to a strike using its own vol (iterative)
    3. Return FXSmile with (strike, vol) pairs
    """
    σ_25C = q.atm + q.bf25 + 0.5 * q.rr25
    σ_25P = q.atm + q.bf25 - 0.5 * q.rr25
    σ_ATM = q.atm

    has_10 = q.rr10 != 0.0 or q.bf10 != 0.0
    if has_10:
        σ_10C = q.atm + q.bf10 + 0.5 * q.rr10
        σ_10P = q.atm + q.bf10 - 0.5 * q.rr10

    # Convert delta → strike self-consistently for each node
    # Iterate: K = delta_to_strike(delta, ..., sigma=vol_at_K)
    def solve_strike(target_delta: float, sigma: float, otype: str) -> float:
        return delta_to_strike(target_delta, q.S, q.T, q.r_d, q.r_f, sigma, otype)

    K_ATM = atm_dns_strike(q.S, q.T, q.r_d, q.r_f, σ_ATM)
    K_25C = solve_strike(0.25, σ_25C, "call")
    K_25P = solve_strike(0.25, σ_25P, "put")

    if has_10:
        K_10C = solve_strike(0.10, σ_10C, "call")
        K_10P = solve_strike(0.10, σ_10P, "put")
        strikes = np.array([K_10P, K_25P, K_ATM, K_25C, K_10C])
        vols    = np.array([σ_10P, σ_25P, σ_ATM, σ_25C, σ_10C])
        labels  = ["10P", "25P", "ATM", "25C", "10C"]
    else:
        strikes = np.array([K_25P, K_ATM, K_25C])
        vols    = np.array([σ_25P, σ_ATM, σ_25C])
        labels  = ["25P", "ATM", "25C"]

    # Sort by strike
    order   = np.argsort(strikes)
    return FXSmile(
        S=q.S, T=q.T, r_d=q.r_d, r_f=q.r_f,
        strikes=strikes[order],
        vols=vols[order],
        labels=[labels[i] for i in order],
        quotes=q,
    )


# ── Vanna-Volga pricing ───────────────────────────────────────────────────────

def vanna_volga_price(option_px_bs: float, vanna: float, volga: float,
                      smile: FXSmile, option_type: str,
                      survival_prob: float = 1.0) -> float:
    """
    Vanna-Volga approximation for FX exotic options.

    Adds smile adjustment to Black-Scholes exotic price.

    Parameters
    ----------
    option_px_bs  : BS price of the exotic (unit, no notional)
    vanna, volga  : exotic's vanna and volga greeks
    smile         : FXSmile with 3 or 5 points
    survival_prob : probability option survives to expiry (for barriers, < 1)

    Returns
    -------
    VV price (unit)

    Reference: Castagna & Mercurio (2007), "The Vanna-Volga Method for Implied
    Volatilities", Risk, Jan 2007.
    """
    q   = smile.quotes
    S   = q.S; T = q.T; r_d = q.r_d; r_f = q.r_f
    σ_a = q.atm

    # Three hedging instruments: ATM straddle, 25C, 25P
    K_atm = atm_dns_strike(S, T, r_d, r_f, σ_a)
    σ_25C = σ_a + q.bf25 + 0.5 * q.rr25
    σ_25P = σ_a + q.bf25 - 0.5 * q.rr25
    K_25C = delta_to_strike(0.25, S, T, r_d, r_f, σ_25C, "call")
    K_25P = delta_to_strike(0.25, S, T, r_d, r_f, σ_25P, "put")

    # Vanna and volga of the 3 hedging instruments (at flat ATM vol)
    def _vanna_volga(K, otype):
        d1, d2 = _d1d2(S, K, r_d, r_f, σ_a, T)
        df_f   = np.exp(-r_f * T)
        vanna_ = -df_f * _phi(d1) * d2 / σ_a
        volga_ = S * df_f * _phi(d1) * np.sqrt(T) * d1 * d2 / σ_a
        return vanna_, volga_

    va_25C, vg_25C = _vanna_volga(K_25C, "call")
    va_25P, vg_25P = _vanna_volga(K_25P, "put")
    va_atm, vg_atm = _vanna_volga(K_atm, "call")

    # Weights solving: w1·va_25P + w2·va_atm + w3·va_25C = vanna
    #                  w1·vg_25P + w2·vg_atm + w3·vg_25C = volga
    # Using simplified 2-instrument version (common practice):
    eps = 1e-12
    denom = (va_25C * vg_25P - va_25P * vg_25C) + eps
    w1 = (vanna * vg_25C - volga * va_25C) / denom
    w2 = (volga * va_25P - vanna * vg_25P) / denom

    # Overhedge cost: market price − BS price at flat vol for each instrument
    df_d = np.exp(-r_d * T)

    def _bs_unit(K, sigma, otype):
        d1, d2 = _d1d2(S, K, r_d, r_f, sigma, T)
        df_f_  = np.exp(-r_f * T)
        if otype == "call":
            return S * df_f_ * _N(d1) - K * df_d * _N(d2)
        return K * df_d * _N(-d2) - S * df_f_ * _N(-d1)

    oh_25P = _bs_unit(K_25P, σ_25P, "put") - _bs_unit(K_25P, σ_a, "put")
    oh_25C = _bs_unit(K_25C, σ_25C, "call") - _bs_unit(K_25C, σ_a, "call")

    vv_adjustment = survival_prob * (w1 * oh_25P + w2 * oh_25C)
    return float(option_px_bs + vv_adjustment)


# ── FX barrier option (VV) ────────────────────────────────────────────────────

def price_fx_barrier_vv(S: float, K: float, H: float, T: float,
                         r_d: float, r_f: float, smile: FXSmile,
                         option_type: str = "call",
                         barrier_type: str = "down-out",
                         notional: float = 1_000_000.0) -> dict:
    """
    FX barrier option price via closed-form BS + Vanna-Volga smile adjustment.

    barrier_type: "down-out" | "down-in" | "up-out" | "up-in"
    """
    from scipy.stats import norm as _norm_dist

    σ = smile.vol_at_strike(K)

    # Closed-form barrier under BS (Reiner-Rubinstein-like for FX)
    mu = (r_d - r_f) / σ**2 - 0.5
    lam = np.sqrt(mu**2 + 2 * r_d / σ**2)
    z   = np.log(H / S) / (σ * np.sqrt(T)) + lam * σ * np.sqrt(T)

    x1 = np.log(S / K) / (σ * np.sqrt(T)) + (1 + mu) * σ * np.sqrt(T)
    x2 = np.log(S / H) / (σ * np.sqrt(T)) + (1 + mu) * σ * np.sqrt(T)
    y1 = np.log(H**2 / (S * K)) / (σ * np.sqrt(T)) + (1 + mu) * σ * np.sqrt(T)
    y2 = np.log(H / S) / (σ * np.sqrt(T)) + (1 + mu) * σ * np.sqrt(T)

    eta  = -1 if barrier_type.startswith("down") else +1
    phi  = +1 if option_type == "call" else -1
    df_d = np.exp(-r_d * T)
    df_f = np.exp(-r_f * T)

    def _bs_term(d):
        return S * df_f * phi * _norm_dist.cdf(phi * d) - K * df_d * phi * _norm_dist.cdf(phi * (d - σ * np.sqrt(T)))

    A = phi * (_bs_term(x1))
    B = phi * (_bs_term(x2))
    C = phi * (H / S)**(2 * (mu + 1)) * (
        S * df_f * _norm_dist.cdf(eta * y1) * phi
        - K * df_d * _norm_dist.cdf(eta * (y1 - σ * np.sqrt(T))) * phi
    )
    D = phi * (H / S)**(2 * (mu + 1)) * (
        S * df_f * _norm_dist.cdf(eta * y2) * phi
        - K * df_d * _norm_dist.cdf(eta * (y2 - σ * np.sqrt(T))) * phi
    )

    if barrier_type in ("down-out", "up-out"):
        if (barrier_type == "down-out" and S > H) or (barrier_type == "up-out" and S < H):
            px_bs = A - C if eta * phi < 0 else A - B + D - C
        else:
            px_bs = 0.0
    else:  # down-in or up-in: barrier_in = vanilla - barrier_out
        from .garman_kohlhagen import price_gk as _gk
        vanilla_bs = _gk(S, K, T, r_d, r_f, σ, option_type, 1.0)["unit_px"]
        out_type   = barrier_type.replace("in", "out")
        out        = price_fx_barrier_vv(S, K, H, T, r_d, r_f, smile,
                                          option_type, out_type, 1.0)
        return {"price": (vanilla_bs - out["price_bs"]) * notional,
                "price_bs": (vanilla_bs - out["price_bs"]) * notional,
                "sigma_used": σ}

    # Vanna-Volga adjustment
    d1, d2  = _d1d2(S, K, r_d, r_f, σ, T)
    df_f_   = np.exp(-r_f * T)
    vanna   = -df_f_ * _phi(d1) * d2 / σ
    volga   = S * df_f_ * _phi(d1) * np.sqrt(T) * d1 * d2 / σ

    # Survival probability approximation for barrier options
    if barrier_type == "down-out" and S > H:
        surv = max(0, min(1, np.log(S / H) / (σ * np.sqrt(T))))
    elif barrier_type == "up-out" and S < H:
        surv = max(0, min(1, np.log(H / S) / (σ * np.sqrt(T))))
    else:
        surv = 0.0

    px_vv = vanna_volga_price(px_bs, vanna, volga, smile, option_type, surv)

    return {
        "price":    float(px_vv * notional),
        "price_bs": float(px_bs * notional),
        "vv_adj":   float((px_vv - px_bs) * notional),
        "sigma_used": float(σ),
    }
