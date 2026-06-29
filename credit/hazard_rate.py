"""
credit/hazard_rate.py
Piecewise-constant hazard rate (default intensity) curve.

The hazard rate h(t) is the instantaneous conditional default rate:
    P(τ ∈ [t, t+dt] | τ > t) = h(t) dt

Survival probability:
    Q(τ > T) = exp(-∫₀ᵀ h(s) ds)

For piecewise-constant rates with breakpoints t₁ < t₂ < ... < tₙ and
rates h₁, ..., hₙ (hⱼ applies on (tⱼ₋₁, tⱼ]):
    Q(T) = exp(-Σⱼ hⱼ · (min(T, tⱼ) - tⱼ₋₁)⁺)

Approximate par spread (single-period): s ≈ h × (1 - R)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


class HazardCurve:
    """
    Piecewise-constant hazard rate curve.

    Parameters
    ----------
    tenors : array-like
        Breakpoint maturities in years, strictly increasing.
        Segment j covers (tenors[j-1], tenors[j]] with tenors[-1]=0.
    hazard_rates : array-like
        Constant hazard rate (decimal per year) in each segment.
        len(hazard_rates) == len(tenors).
    recovery : float
        Recovery rate on default (default 0.40 = 40%).
    """

    def __init__(self, tenors, hazard_rates, recovery: float = 0.40):
        self.tenors       = np.asarray(tenors,       dtype=float)
        self.hazard_rates = np.asarray(hazard_rates, dtype=float)
        self.recovery     = float(recovery)
        if len(self.tenors) != len(self.hazard_rates):
            raise ValueError("tenors and hazard_rates must have the same length")
        if np.any(self.hazard_rates < 0):
            raise ValueError("hazard rates must be non-negative")

    # ── core probabilities ────────────────────────────────────────────────────

    def survival_prob(self, T: float) -> float:
        """Q(τ > T) = exp(-∫₀ᵀ h(s) ds)"""
        T = float(T)
        if T <= 0:
            return 1.0
        integral = 0.0
        t_prev   = 0.0
        for t_j, h_j in zip(self.tenors, self.hazard_rates):
            if T <= t_j:
                integral += h_j * (T - t_prev)
                return float(np.exp(-integral))
            integral += h_j * (t_j - t_prev)
            t_prev = t_j
        # T beyond last breakpoint: extend flat at last rate
        integral += self.hazard_rates[-1] * (T - t_prev)
        return float(np.exp(-integral))

    def default_prob(self, T: float) -> float:
        """PD(T) = 1 − Q(τ > T)"""
        return 1.0 - self.survival_prob(T)

    def hazard_rate_at(self, T: float) -> float:
        """Piecewise-constant hazard rate h(T)."""
        T = float(T)
        for t_j, h_j in zip(self.tenors, self.hazard_rates):
            if T <= t_j:
                return float(h_j)
        return float(self.hazard_rates[-1])

    def cumulative_hazard(self, T: float) -> float:
        """Λ(T) = ∫₀ᵀ h(s) ds = −ln Q(T)"""
        q = self.survival_prob(T)
        return -np.log(max(q, 1e-300))

    def credit_spread(self, T: float) -> float:
        """
        Implied flat credit spread at tenor T (decimal per year):
            s(T) = Λ(T) / T × (1 − R)
        This equals the approximate par CDS spread for a single-period contract.
        """
        T = float(T)
        if T <= 0:
            return 0.0
        return self.cumulative_hazard(T) / T * (1.0 - self.recovery)

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_flat(cls, hazard_rate: float, recovery: float = 0.40,
                  max_tenor: float = 30.0) -> "HazardCurve":
        """Flat (constant) hazard rate curve."""
        return cls([max_tenor], [hazard_rate], recovery)

    @classmethod
    def from_spread(cls, spread_bps: float, recovery: float = 0.40,
                    max_tenor: float = 30.0) -> "HazardCurve":
        """
        Flat curve from a single credit spread (basis points per year).
        Approximation:  h ≈ spread / (1 − R)
        """
        h = (spread_bps / 10_000.0) / (1.0 - recovery)
        return cls.from_flat(h, recovery, max_tenor)

    @classmethod
    def bootstrap(cls, tenors, spreads_bps, discount_curve,
                  recovery: float = 0.40, n_steps: int = 100) -> "HazardCurve":
        """
        Bootstrap piecewise-constant hazard rates from CDS par spreads.

        For each tenor T_i we solve for h_i such that the model par spread
        (using h_1..h_{i-1} from previous steps and h_i for the current
        segment) matches the market quote s_i.

        Parameters
        ----------
        tenors      : list of float — CDS maturities in years
        spreads_bps : list of float — par CDS spreads in basis points
        discount_curve : DiscountCurve (risk-free)
        recovery    : float — recovery rate (default 0.40)
        n_steps     : int  — integration steps per unit time in CDS pricing

        Returns
        -------
        HazardCurve with len(tenors) piecewise-constant segments
        """
        from .cds import par_spread as _par_spread

        tenors      = list(tenors)
        spreads_bps = list(spreads_bps)
        hazard_rates = []

        for i, (T_i, s_target_bps) in enumerate(zip(tenors, spreads_bps)):
            prev_tenors = tenors[:i]
            prev_rates  = hazard_rates[:]

            def residual(h_i, T_i=T_i, prev_tenors=prev_tenors,
                         prev_rates=prev_rates, s_target_bps=s_target_bps):
                hc = cls(prev_tenors + [T_i], prev_rates + [h_i], recovery)
                s_model = _par_spread(hc, discount_curve, T_i,
                                      recovery=recovery, n_steps=n_steps)
                return s_model * 10_000 - s_target_bps

            h_i = brentq(residual, 1e-8, 0.99, xtol=1e-10, maxiter=300)
            hazard_rates.append(h_i)

        return cls(tenors, hazard_rates, recovery)

    # ── display ───────────────────────────────────────────────────────────────

    def summary(self, max_tenor: float = None) -> str:
        T_max = max_tenor or float(self.tenors[-1])
        lines = [f"HazardCurve  (R = {self.recovery:.0%})\n",
                 f"  {'Tenor':>6}  {'h (bps/yr)':>12}  {'Q(τ>T)':>8}  {'PD(T)':>8}  {'Spread(bps)':>12}"]
        for t, h in zip(self.tenors, self.hazard_rates):
            if t > T_max:
                break
            q = self.survival_prob(t)
            s = self.credit_spread(t) * 10_000
            lines.append(f"  {t:>6.2f}  {h*10000:>12.1f}  {q:>8.4f}  {1-q:>8.4f}  {s:>12.1f}")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()
