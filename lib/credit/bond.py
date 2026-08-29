"""
credit/bond.py
Risky (credit-risky) bond pricing, Z-spread, and asset swap spread.

Under the Duffie-Singleton model, a bond's cash flows are discounted
by both the risk-free rate and the hazard rate:

    P = Σᵢ CFᵢ × DF(tᵢ) × Q(tᵢ)              (survival-weighted cash flows)
      + R × face × Σᵢ DF(tᵢ) × [Q(tᵢ₋₁) − Q(tᵢ)]  (recovery on default)

Z-spread: constant z added to every point of the risk-free curve such that
    P = Σᵢ CFᵢ × exp(−(r(tᵢ) + z) × tᵢ)

Asset swap spread (ASW): fixed-rate bond package sold at par.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def _coupon_times(maturity: float, pay_freq: int) -> np.ndarray:
    n     = max(1, int(round(maturity * pay_freq)))
    times = np.arange(1, n + 1) / pay_freq
    times[-1] = maturity
    return times


def risky_bond_price(face: float, coupon_rate: float, maturity: float,
                     hazard_curve, discount_curve,
                     pay_freq: int = 2,
                     recovery: float = None) -> dict:
    """
    Price a fixed-coupon risky bond (Duffie-Singleton, recovery of face value).

    Parameters
    ----------
    face        : par / face value
    coupon_rate : annual coupon rate (decimal), e.g. 0.05 = 5%
    maturity    : years to maturity
    pay_freq    : coupons per year (2 = semi-annual, standard)
    recovery    : override hazard_curve.recovery

    Returns
    -------
    dict: 'price', 'price_pct', 'pv_coupons', 'pv_recovery',
          'yield_to_maturity', 'credit_spread_bps'
    """
    R      = recovery if recovery is not None else hazard_curve.recovery
    c      = face * coupon_rate / pay_freq
    times  = _coupon_times(maturity, pay_freq)

    pv_cf       = 0.0
    pv_recovery = 0.0
    Q_prev      = 1.0

    for t in times:
        Q_t  = hazard_curve.survival_prob(t)
        DF_t = discount_curve.discount_factor(t)

        # Coupon + principal at maturity
        cf   = c + (face if abs(t - maturity) < 1e-10 else 0.0)
        pv_cf       += cf * DF_t * Q_t
        pv_recovery += R * face * DF_t * (Q_prev - Q_t)
        Q_prev = Q_t

    price = pv_cf + pv_recovery
    ytm   = _ytm(price, face, coupon_rate, maturity, pay_freq)
    z_spr = discount_curve.zero_rate(maturity)
    cs    = (ytm - z_spr) * 10_000

    return {
        "price":              float(price),
        "price_pct":          float(price / face * 100),
        "pv_coupons":         float(pv_cf),
        "pv_recovery":        float(pv_recovery),
        "yield_to_maturity":  float(ytm),
        "credit_spread_bps":  float(cs),
    }


def _ytm(price: float, face: float, coupon_rate: float,
         maturity: float, pay_freq: int) -> float:
    """Internal rate of return (YTM) of the bond's cash flows."""
    c     = face * coupon_rate / pay_freq
    times = _coupon_times(maturity, pay_freq)

    def f(y):
        return sum(
            (c + (face if abs(t - maturity) < 1e-10 else 0.0)) * np.exp(-y * t)
            for t in times
        ) - price

    try:
        return float(brentq(f, -0.50, 5.0, xtol=1e-10))
    except ValueError:
        return np.nan


def z_spread(market_price: float, face: float, coupon_rate: float,
             maturity: float, discount_curve,
             pay_freq: int = 2) -> float:
    """
    Z-spread: constant z (decimal) such that
        market_price = Σᵢ CFᵢ × exp(−(r(tᵢ) + z) × tᵢ)

    Returns z in decimal; multiply by 10 000 for basis points.
    """
    c     = face * coupon_rate / pay_freq
    times = _coupon_times(maturity, pay_freq)

    def f(z):
        return sum(
            (c + (face if abs(t - maturity) < 1e-10 else 0.0))
            * np.exp(-(discount_curve.zero_rate(t) + z) * t)
            for t in times
        ) - market_price

    return float(brentq(f, -0.50, 5.0, xtol=1e-10))


def asset_swap_spread(market_price: float, face: float, coupon_rate: float,
                      maturity: float, discount_curve,
                      pay_freq: int = 2) -> float:
    """
    Asset swap spread (ASW): the flat spread over LIBOR/SOFR such that
    selling the bond at par and entering the swap has zero NPV.

        ASW = (PV_fixed_leg − market_price) / (face × annuity)

    where PV_fixed_leg is the risk-free PV of the bond's fixed cash flows
    and annuity = Σᵢ DF(tᵢ) × Δtᵢ.

    Returns ASW in decimal; multiply by 10 000 for basis points.
    """
    c     = face * coupon_rate / pay_freq
    times = _coupon_times(maturity, pay_freq)
    dt    = 1.0 / pay_freq

    annuity   = sum(discount_curve.discount_factor(t) * dt for t in times)
    pv_fixed  = sum(
        (c + (face if abs(t - maturity) < 1e-10 else 0.0))
        * discount_curve.discount_factor(t)
        for t in times
    )

    asw = (pv_fixed - market_price) / (face * annuity) if annuity > 1e-12 else np.nan
    return float(asw)


def bond_summary(face: float, coupon_rate: float, maturity: float,
                 hazard_curve, discount_curve,
                 pay_freq: int = 2, recovery: float = None) -> str:
    """Print-ready bond analysis summary."""
    res  = risky_bond_price(face, coupon_rate, maturity, hazard_curve,
                            discount_curve, pay_freq, recovery)
    z    = z_spread(res["price"], face, coupon_rate, maturity, discount_curve, pay_freq)
    asw  = asset_swap_spread(res["price"], face, coupon_rate, maturity, discount_curve, pay_freq)

    lines = [
        f"Risky Bond  face={face:,.0f}  coupon={coupon_rate:.2%}  T={maturity:.1f}Y",
        f"  Price:          ${res['price']:>10,.4f}  ({res['price_pct']:.4f}%)",
        f"  PV coupons:     ${res['pv_coupons']:>10,.4f}",
        f"  PV recovery:    ${res['pv_recovery']:>10,.4f}",
        f"  YTM:            {res['yield_to_maturity']:.4%}",
        f"  Credit spread:  {res['credit_spread_bps']:.1f} bps  (YTM − risk-free zero)",
        f"  Z-spread:       {z * 10_000:.1f} bps",
        f"  Asset swap spr: {asw * 10_000:.1f} bps",
    ]
    return "\n".join(lines)
