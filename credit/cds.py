"""
credit/cds.py
Credit Default Swap (CDS) pricing.

A CDS transfers credit risk from buyer (long protection) to seller (short protection):
  - Buyer pays periodic spread s on notional (premium leg) until default or maturity T
  - Seller pays (1 − R) × notional on a credit event (protection leg)

Fair (par) spread:  s* = ProtectionLeg / RPV01

    ProtLeg  = (1−R) × Σᵢ DF(tᵢ) × [Q(tᵢ₋₁) − Q(tᵢ)]
    RPV01    = Σᵢ Δtᵢ × DF(tᵢ) × Q(tᵢ)

MtM value (protection buyer):   V = ProtLeg − s_contract × RPV01
"""

from __future__ import annotations

import numpy as np


# ── private helpers ───────────────────────────────────────────────────────────

class _BumpedCurve:
    """Wraps a DiscountCurve with a parallel zero-rate bump."""
    def __init__(self, base, bump: float):
        self._base = base
        self._bump = bump

    def discount_factor(self, T):
        return self._base.discount_factor(T) * np.exp(-self._bump * T)

    def zero_rate(self, T):
        return self._base.zero_rate(T) + self._bump


def _coupon_times(maturity: float, pay_freq: int) -> np.ndarray:
    dt    = 1.0 / pay_freq
    n     = max(1, int(round(maturity * pay_freq)))
    times = np.arange(1, n + 1) * dt
    times[-1] = maturity          # exact maturity for last coupon
    return times


# ── core legs ────────────────────────────────────────────────────────────────

def protection_leg(hazard_curve, discount_curve, maturity: float,
                   recovery: float = None, n_steps: int = 360) -> float:
    """
    PV of the protection leg per unit notional:
        (1−R) × Σᵢ DF(tᵢ) × [Q(tᵢ₋₁) − Q(tᵢ)]

    Uses a fine time grid so continuous-monitoring default is well approximated.
    """
    R  = recovery if recovery is not None else hazard_curve.recovery
    t  = np.linspace(0.0, maturity, n_steps + 1)
    Q  = np.array([hazard_curve.survival_prob(ti) for ti in t])
    DF = np.array([discount_curve.discount_factor(ti) for ti in t[1:]])
    return float((1.0 - R) * np.dot(DF, Q[:-1] - Q[1:]))


def risky_pv01(hazard_curve, discount_curve, maturity: float,
               pay_freq: int = 4) -> float:
    """
    Risky PV01 (RPV01): present value of paying 1 unit per year contingent
    on survival, at `pay_freq` coupons per year.

        RPV01 = Σᵢ Δtᵢ × DF(tᵢ) × Q(tᵢ)
    """
    times  = _coupon_times(maturity, pay_freq)
    t_prev = 0.0
    result = 0.0
    for t in times:
        dt   = t - t_prev
        Q    = hazard_curve.survival_prob(t)
        DF   = discount_curve.discount_factor(t)
        result += dt * DF * Q
        t_prev = t
    return float(result)


def par_spread(hazard_curve, discount_curve, maturity: float,
               recovery: float = None, pay_freq: int = 4,
               n_steps: int = 360) -> float:
    """
    Par CDS spread: s* = ProtectionLeg / RPV01  (decimal, e.g. 0.01 = 100 bps).
    """
    prot  = protection_leg(hazard_curve, discount_curve, maturity, recovery, n_steps)
    rpv01 = risky_pv01(hazard_curve, discount_curve, maturity, pay_freq)
    return float(prot / rpv01) if rpv01 > 1e-14 else np.inf


# ── CDS mark-to-market ────────────────────────────────────────────────────────

def cds_value(hazard_curve, discount_curve, maturity: float,
              spread: float, recovery: float = None,
              position: str = "buyer", notional: float = 1_000_000,
              pay_freq: int = 4, n_steps: int = 360) -> dict:
    """
    Mark-to-market value of an existing CDS.

    Parameters
    ----------
    spread   : float — contractual spread (decimal, e.g. 0.01 = 100 bps)
    position : 'buyer'  (long protection — pay spread, receive on default)
               'seller' (short protection — receive spread, pay on default)
    notional : contract notional in currency units

    Returns
    -------
    dict:
      'value'          — MtM value in notional units
      'protection_leg' — PV of protection (seller pays on default)
      'premium_leg'    — PV of premium (buyer pays periodically)
      'par_spread_bps' — current fair spread in bps
      'rpv01'          — risky PV01 in notional units
      'cs01'           — DV01 per 1bp spread move
    """
    R    = recovery if recovery is not None else hazard_curve.recovery
    prot = protection_leg(hazard_curve, discount_curve, maturity, R, n_steps)
    rpv  = risky_pv01(hazard_curve, discount_curve, maturity, pay_freq)
    prem = spread * rpv
    s_par = (prot / rpv * 10_000) if rpv > 1e-14 else np.inf

    v_buyer = (prot - prem) * notional
    value   = v_buyer if position == "buyer" else -v_buyer

    return {
        "value":           float(value),
        "protection_leg":  float(prot * notional),
        "premium_leg":     float(prem * notional),
        "par_spread_bps":  float(s_par),
        "rpv01":           float(rpv * notional),
        "cs01":            float(rpv * notional * 1e-4),   # per 1bp (signed for buyer)
        "position":        position,
        "notional":        float(notional),
    }


# ── sensitivities ─────────────────────────────────────────────────────────────

def cs01(hazard_curve, discount_curve, maturity: float,
         spread: float, recovery: float = None,
         position: str = "buyer", notional: float = 1_000_000,
         pay_freq: int = 4, bump_bps: float = 1.0) -> float:
    """
    CS01: change in CDS value for a `bump_bps` parallel upward shift
    of the credit spread (i.e., all hazard rates scaled proportionally).

    For a flat curve, approximately −RPV01 × notional × bump for buyer.
    """
    from .hazard_rate import HazardCurve

    R    = recovery if recovery is not None else hazard_curve.recovery
    bump = bump_bps / 10_000.0

    # Scale hazard rates to produce a +1bp spread shift
    h_bump = hazard_curve.hazard_rates + bump / (1.0 - R)
    hc_bump = HazardCurve(hazard_curve.tenors, h_bump, R)

    v0 = cds_value(hazard_curve, discount_curve, maturity,
                   spread, R, position, notional, pay_freq)["value"]
    v1 = cds_value(hc_bump,      discount_curve, maturity,
                   spread, R, position, notional, pay_freq)["value"]
    return float(v1 - v0)


def ir01(hazard_curve, discount_curve, maturity: float,
         spread: float, recovery: float = None,
         position: str = "buyer", notional: float = 1_000_000,
         n_steps: int = 360, bump_bps: float = 1.0) -> float:
    """
    IR01: change in CDS value for a +1bp parallel shift of the risk-free curve.
    """
    R    = recovery if recovery is not None else hazard_curve.recovery
    bump = bump_bps / 10_000.0
    dc_bump = _BumpedCurve(discount_curve, bump)

    v0 = cds_value(hazard_curve, discount_curve, maturity,
                   spread, R, position, notional, n_steps=n_steps)["value"]
    v1 = cds_value(hazard_curve, dc_bump,        maturity,
                   spread, R, position, notional, n_steps=n_steps)["value"]
    return float(v1 - v0)
