"""Vanilla fixed-for-floating interest-rate swap valuation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import DiscountCurve


def fixed_leg_pv(
    notional: float,
    fixed_rate: float,
    maturity: float,
    curve: DiscountCurve,
    pay_freq: int = 1,
) -> float:
    if maturity <= 0:
        raise ValueError("maturity must be positive.")
    if pay_freq <= 0:
        raise ValueError("pay_freq must be positive.")
    alpha = 1.0 / pay_freq
    n_payments = int(round(maturity * pay_freq))
    pay_times = np.arange(1, n_payments + 1, dtype=float) * alpha
    coupons = notional * fixed_rate * alpha * curve.discount_factor(pay_times)
    return float(np.sum(coupons))


def floating_leg_pv(
    notional: float,
    maturity: float,
    curve: DiscountCurve,
    pay_freq: int = 1,
) -> float:
    if maturity <= 0:
        raise ValueError("maturity must be positive.")
    n_payments = int(round(maturity * pay_freq))
    end = n_payments / pay_freq
    return float(notional * (1.0 - curve.discount_factor(end)))


def par_swap_rate(maturity: float, curve: DiscountCurve, pay_freq: int = 1) -> float:
    return curve.par_swap_rate(maturity, pay_freq=pay_freq)


def swap_pv(
    notional: float,
    fixed_rate: float,
    maturity: float,
    curve: DiscountCurve,
    pay_freq: int = 1,
    position: str = "payer",
) -> float:
    fixed = fixed_leg_pv(notional, fixed_rate, maturity, curve, pay_freq)
    floating = floating_leg_pv(notional, maturity, curve, pay_freq)
    if position == "payer":
        return float(floating - fixed)
    if position == "receiver":
        return float(fixed - floating)
    raise ValueError("position must be 'payer' or 'receiver'.")


@dataclass(frozen=True)
class InterestRateSwap:
    notional: float
    fixed_rate: float
    maturity: float
    pay_freq: int = 1
    position: str = "payer"

    def fixed_leg_pv(self, curve: DiscountCurve) -> float:
        return fixed_leg_pv(self.notional, self.fixed_rate, self.maturity, curve, self.pay_freq)

    def floating_leg_pv(self, curve: DiscountCurve) -> float:
        return floating_leg_pv(self.notional, self.maturity, curve, self.pay_freq)

    def par_rate(self, curve: DiscountCurve) -> float:
        return par_swap_rate(self.maturity, curve, self.pay_freq)

    def pv(self, curve: DiscountCurve) -> float:
        return swap_pv(
            self.notional,
            self.fixed_rate,
            self.maturity,
            curve,
            self.pay_freq,
            self.position,
        )
