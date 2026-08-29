"""
Interest-rate curve construction and transformations.

The curve stores discount factors and interpolates linearly in log-discount
space. That is equivalent to piecewise-constant instantaneous forward rates and
keeps interpolated discount factors positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable

import numpy as np


def _as_sorted_arrays(times: Iterable[float], dfs: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(list(times), dtype=float)
    d = np.asarray(list(dfs), dtype=float)
    if t.ndim != 1 or d.ndim != 1 or len(t) != len(d):
        raise ValueError("times and discount_factors must be one-dimensional arrays of equal length.")
    if len(t) == 0:
        raise ValueError("curve requires at least one pillar.")
    if np.any(t <= 0):
        raise ValueError("curve times must be positive.")
    if np.any(d <= 0):
        raise ValueError("discount factors must be positive.")
    order = np.argsort(t)
    t = t[order]
    d = d[order]
    if np.any(np.diff(t) <= 0):
        raise ValueError("curve times must be unique.")
    return t, d


@dataclass(frozen=True)
class DiscountCurve:
    """Continuously-compounded zero curve represented by discount factors."""

    times: np.ndarray
    discount_factors: np.ndarray
    name: str = "discount_curve"

    def __post_init__(self):
        t, d = _as_sorted_arrays(self.times, self.discount_factors)
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "discount_factors", d)

    @classmethod
    def flat(cls, rate: float, max_maturity: float = 30.0, name: str = "flat") -> "DiscountCurve":
        if max_maturity <= 0:
            raise ValueError("max_maturity must be positive.")
        times = np.array([1.0 / 365.0, max_maturity], dtype=float)
        dfs = np.exp(-rate * times)
        return cls(times, dfs, name=name)

    @classmethod
    def from_zero_rates(
        cls,
        times: Iterable[float],
        zero_rates: Iterable[float],
        name: str = "zero_curve",
    ) -> "DiscountCurve":
        t = np.asarray(list(times), dtype=float)
        z = np.asarray(list(zero_rates), dtype=float)
        if len(t) != len(z):
            raise ValueError("times and zero_rates must have equal length.")
        return cls(t, np.exp(-z * t), name=name)

    _TENOR_YEARS = {"1M": 1 / 12, "3M": 3 / 12, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "5Y": 5.0, "10Y": 10.0, "30Y": 30.0}

    @classmethod
    def from_tenor_dict(cls, tenors: dict[str, float], name: str = "tenor_curve") -> "DiscountCurve":
        """Build a curve from a `{tenor_label: zero_rate}` dict (e.g. FRED Treasury
        points). Treats each rate as a continuously-compounded zero rate at the
        tenor's year-fraction — an approximation of the CMT bond-equivalent yields
        FRED actually reports, acceptable for a research-prototype discount curve."""
        known = {k: v for k, v in tenors.items() if k in cls._TENOR_YEARS}
        if not known:
            raise ValueError(f"no recognized tenors in {list(tenors)}")
        times = [cls._TENOR_YEARS[k] for k in known]
        rates = [known[k] for k in known]
        return cls.from_zero_rates(times, rates, name=name)

    def discount_factor(self, maturity: float | np.ndarray) -> float | np.ndarray:
        scalar = np.ndim(maturity) == 0
        m = np.atleast_1d(np.asarray(maturity, dtype=float))
        if np.any(m < 0):
            raise ValueError("maturity must be non-negative.")

        log_dfs = np.log(self.discount_factors)
        left_slope = log_dfs[0] / self.times[0]
        right_slope = (log_dfs[-1] - log_dfs[-2]) / (self.times[-1] - self.times[-2]) if len(self.times) > 1 else left_slope

        out = np.empty_like(m, dtype=float)
        out[m == 0] = 1.0
        left = (m > 0) & (m < self.times[0])
        out[left] = np.exp(left_slope * m[left])

        mid = (m >= self.times[0]) & (m <= self.times[-1])
        out[mid] = np.exp(np.interp(m[mid], self.times, log_dfs))

        right = m > self.times[-1]
        out[right] = np.exp(log_dfs[-1] + right_slope * (m[right] - self.times[-1]))

        return float(out[0]) if scalar else out

    def zero_rate(self, maturity: float | np.ndarray) -> float | np.ndarray:
        scalar = np.ndim(maturity) == 0
        m = np.atleast_1d(np.asarray(maturity, dtype=float))
        if np.any(m <= 0):
            raise ValueError("maturity must be positive.")
        z = -np.log(self.discount_factor(m)) / m
        return float(z[0]) if scalar else z

    def forward_rate(self, start: float, end: float) -> float:
        if start < 0 or end <= start:
            raise ValueError("require 0 <= start < end.")
        df1 = self.discount_factor(start)
        df2 = self.discount_factor(end)
        return float(-log(df2 / df1) / (end - start))

    def par_swap_rate(self, maturity: float, pay_freq: int = 1) -> float:
        if maturity <= 0:
            raise ValueError("maturity must be positive.")
        if pay_freq <= 0:
            raise ValueError("pay_freq must be positive.")
        alpha = 1.0 / pay_freq
        n_payments = int(round(maturity * pay_freq))
        if n_payments <= 0:
            raise ValueError("maturity is shorter than one payment period.")
        pay_times = np.arange(1, n_payments + 1, dtype=float) * alpha
        annuity = float(np.sum(alpha * self.discount_factor(pay_times)))
        return float((1.0 - self.discount_factor(pay_times[-1])) / annuity)

    def bumped(self, parallel_bump: float, name: str | None = None) -> "DiscountCurve":
        z = self.zero_rate(self.times) + parallel_bump
        return DiscountCurve.from_zero_rates(self.times, z, name=name or f"{self.name}_bumped")


def bootstrap_deposit_swap_curve(
    deposit_quotes: dict[float, float] | None = None,
    swap_quotes: dict[float, float] | None = None,
    fixed_freq: int = 1,
    name: str = "bootstrapped",
) -> DiscountCurve:
    """
    Build a simple discount curve from money-market deposits and par swap rates.

    Deposit quotes are simple annualized rates by maturity:
        DF(T) = 1 / (1 + rT)

    Swap quotes are annual par rates. Unknown longer discount factors are solved
    from:
        S * sum(alpha_i * DF(T_i)) = 1 - DF(T_n)
    """
    deposit_quotes = deposit_quotes or {}
    swap_quotes = swap_quotes or {}
    if not deposit_quotes and not swap_quotes:
        raise ValueError("at least one deposit or swap quote is required.")
    if fixed_freq <= 0:
        raise ValueError("fixed_freq must be positive.")

    pillars: dict[float, float] = {}
    for maturity, rate in sorted(deposit_quotes.items()):
        if maturity <= 0:
            raise ValueError("deposit maturities must be positive.")
        pillars[float(maturity)] = 1.0 / (1.0 + float(rate) * float(maturity))

    for maturity, swap_rate in sorted(swap_quotes.items()):
        maturity = float(maturity)
        swap_rate = float(swap_rate)
        if maturity <= 0:
            raise ValueError("swap maturities must be positive.")

        alpha = 1.0 / fixed_freq
        n_payments = int(round(maturity * fixed_freq))
        if n_payments <= 0:
            raise ValueError("swap maturity is shorter than one payment period.")
        pay_times = np.arange(1, n_payments + 1, dtype=float) * alpha

        known_annuity = 0.0
        unknown_time = float(pay_times[-1])
        temp_curve = None
        if pillars:
            temp_curve = DiscountCurve(np.array(list(pillars)), np.array(list(pillars.values())))

        for t in pay_times[:-1]:
            if float(t) in pillars:
                df = pillars[float(t)]
            elif temp_curve is not None:
                df = temp_curve.discount_factor(float(t))
            else:
                raise ValueError("need shorter pillars before bootstrapping swaps.")
            known_annuity += alpha * df

        df_n = (1.0 - swap_rate * known_annuity) / (1.0 + swap_rate * alpha)
        if df_n <= 0:
            raise ValueError("bootstrapped discount factor is non-positive.")
        pillars[unknown_time] = float(df_n)

    return DiscountCurve(np.array(list(pillars)), np.array(list(pillars.values())), name=name)


def present_value(cashflows: Iterable[tuple[float, float]], curve: DiscountCurve) -> float:
    """Present value of `(time, amount)` cashflows."""
    return float(sum(amount * curve.discount_factor(time) for time, amount in cashflows))
