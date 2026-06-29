"""
commodities/futures_curve.py
Commodity futures term structure analysis.

Key concepts:
  Contango    : F(T) > S · e^{rT}  — futures > cost of carry  (storage cost > convenience yield)
  Backwardation: F(T) < S · e^{rT} — futures < cost of carry  (convenience yield > storage cost)
  Calendar spread: F(T2) − F(T1)
  Roll yield  : gain/loss from rolling a long futures position as it approaches expiry
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


@dataclass
class FuturesCurve:
    """
    Commodity futures term structure.

    Parameters
    ----------
    maturities    : array of contract maturities in years (sorted)
    futures_prices: corresponding futures prices
    spot          : current spot price (if known)
    commodity     : name string
    """
    maturities:     np.ndarray
    futures_prices: np.ndarray
    spot:           float = np.nan
    commodity:      str   = ""
    _spline: object = field(default=None, repr=False)

    def __post_init__(self):
        self.maturities     = np.asarray(self.maturities, dtype=float)
        self.futures_prices = np.asarray(self.futures_prices, dtype=float)
        order = np.argsort(self.maturities)
        self.maturities     = self.maturities[order]
        self.futures_prices = self.futures_prices[order]
        # Cubic spline for interpolation (log prices for positivity)
        if len(self.maturities) >= 4:
            self._spline = CubicSpline(self.maturities,
                                        np.log(self.futures_prices))

    def price(self, T: float | np.ndarray) -> np.ndarray:
        """Interpolate futures price at maturity T."""
        T = np.asarray(T, dtype=float)
        if self._spline is not None:
            return np.exp(self._spline(T))
        return np.exp(np.interp(T, self.maturities, np.log(self.futures_prices)))

    def forward_price(self, T1: float, T2: float) -> float:
        """Implied forward price for delivery at T2 observed at T1."""
        return float(self.price(T2))   # in practice: adjust for carry

    def calendar_spread(self, T1: float, T2: float) -> float:
        """F(T2) − F(T1): positive = contango, negative = backwardation."""
        return float(self.price(T2) - self.price(T1))

    def annualized_basis(self, r: float = 0.05) -> np.ndarray:
        """
        Annualized convenience yield implied by each contract.
        cy_T = r − (1/T) ln(F(T)/S)  (storage cost omitted)
        """
        if np.isnan(self.spot):
            raise ValueError("spot price required for annualized_basis")
        T = self.maturities
        cy = r - np.log(self.futures_prices / self.spot) / T
        return cy

    def is_contango(self) -> bool:
        """True if curve is generally in contango (upward sloping)."""
        diffs = np.diff(self.futures_prices)
        return bool(np.sum(diffs > 0) > len(diffs) / 2)

    def roll_yield(self, T1: float, T2: float) -> float:
        """
        Annualized roll yield for a position rolled from T1 to T2 contract.
        Roll yield = (F(T1) − F(T2)) / F(T2) × 1/(T2−T1)
        Positive in backwardation (profits from rolling).
        """
        f1 = float(self.price(T1))
        f2 = float(self.price(T2))
        return float((f1 - f2) / f2 / (T2 - T1))

    def total_return_index(self, roll_schedule: list[tuple[float, float]],
                            r_collateral: float = 0.05) -> np.ndarray:
        """
        Total return index for rolling strategy.
        Returns array of index values at each roll date.

        roll_schedule: list of (T_current_contract, T_next_contract) pairs
        """
        idx = [1.0]
        for T_old, T_new in roll_schedule:
            ry   = self.roll_yield(T_old, T_new)
            dt   = T_new - T_old
            idx.append(idx[-1] * np.exp((ry + r_collateral) * dt))
        return np.array(idx)

    def summary(self) -> str:
        shape = "Contango" if self.is_contango() else "Backwardation"
        lines = [
            f"Futures Curve: {self.commodity or 'Unnamed'}  [{shape}]",
            f"  Spot: {self.spot:.3f}" if not np.isnan(self.spot) else "",
            f"  {'Maturity':>10}  {'Futures':>10}  {'Basis':>10}",
        ]
        for T, F in zip(self.maturities, self.futures_prices):
            basis = (F - self.spot) if not np.isnan(self.spot) else np.nan
            b_str = f"{basis:>+10.3f}" if not np.isnan(basis) else "         —"
            lines.append(f"  {T:>10.3f}y  {F:>10.3f}  {b_str}")
        return "\n".join(l for l in lines if l)


def crack_spread(crude_price: float, gasoline_price: float,
                 heating_oil_price: float = 0.0,
                 ratio_321: bool = True) -> float:
    """
    3-2-1 crack spread: value of refining 3 barrels of crude.
    crack = (2 × gasoline + 1 × heating_oil − 3 × crude) / 3

    Prices in $/barrel (or $/gallon × 42 for gasoline/HO).
    """
    if ratio_321:
        return (2 * gasoline_price + heating_oil_price - 3 * crude_price) / 3
    return gasoline_price - crude_price


def spark_spread(power_price: float, gas_price: float,
                 heat_rate: float = 7.5) -> float:
    """
    Spark spread: profitability of gas-fired power generation.
    spark = power_price − heat_rate × gas_price
    heat_rate: MMBtu per MWh (typical: 6–10)
    """
    return power_price - heat_rate * gas_price
