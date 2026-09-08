"""
volsurface/surfaces/base.py
The one interface every volatility surface in this repo implements.

A surface is defined by its **total implied variance**

    w(k, T) = sigma_BS(k, T)^2 * T,        k = log(K / F_T)

rather than by implied vol directly. That is not a stylistic choice: both static
no-arbitrage conditions (calendar spread, butterfly) are clean statements about
``w`` and ugly ones about ``sigma``, and ``w`` is the quantity that must vanish
at ``T -> 0`` and grow with maturity. Models are therefore compared, penalised
and plotted in ``w`` space, and implied vol is a derived quantity.

Implementing `total_variance` is enough to get IV, prices, and the full
arbitrage diagnostic suite in `volsurface.evaluation.diagnostics` for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VolSurface(ABC):
    """Base class for an implied-volatility surface in (log-moneyness, maturity)."""

    name: str = "surface"

    # -- the single thing a subclass must provide ----------------------------

    @abstractmethod
    def total_variance(self, k: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Total implied variance ``w(k, T)``, broadcast over aligned arrays."""

    # -- derived quantities ---------------------------------------------------

    def implied_vol(self, k: np.ndarray, T: np.ndarray) -> np.ndarray:
        """``sigma(k, T) = sqrt(w / T)``."""
        k, T = np.asarray(k, dtype=float), np.asarray(T, dtype=float)
        w = np.maximum(self.total_variance(k, T), 1e-12)
        return np.sqrt(w / np.maximum(T, 1e-12))

    def call_price(self, k: np.ndarray, T: np.ndarray,
                   forward: np.ndarray, discount: np.ndarray = 1.0) -> np.ndarray:
        """Undiscounted-then-discounted Black-76 call price off the surface.

        Lets a surface be scored in *price* space against the market mids it was
        fitted to in vol space -- the only comparison a trading desk cares about.
        """
        from scipy.stats import norm

        k, T = np.asarray(k, dtype=float), np.asarray(T, dtype=float)
        w = np.maximum(self.total_variance(k, T), 1e-12)
        sqrt_w = np.sqrt(w)
        d1 = -k / sqrt_w + 0.5 * sqrt_w
        d2 = d1 - sqrt_w
        strike = forward * np.exp(k)
        return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))

    # -- finite-difference derivatives used by the diagnostics ---------------

    def dw_dT(self, k: np.ndarray, T: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """Central difference in maturity, one-sided near ``T = 0``."""
        k, T = np.asarray(k, dtype=float), np.asarray(T, dtype=float)
        lo = np.maximum(T - h, 1e-6)
        hi = T + h
        return (self.total_variance(k, hi) - self.total_variance(k, lo)) / (hi - lo)

    def dw_dk(self, k: np.ndarray, T: np.ndarray, h: float = 1e-4) -> np.ndarray:
        k, T = np.asarray(k, dtype=float), np.asarray(T, dtype=float)
        return (self.total_variance(k + h, T) - self.total_variance(k - h, T)) / (2 * h)

    def d2w_dk2(self, k: np.ndarray, T: np.ndarray, h: float = 1e-3) -> np.ndarray:
        k, T = np.asarray(k, dtype=float), np.asarray(T, dtype=float)
        return (
            self.total_variance(k + h, T)
            - 2 * self.total_variance(k, T)
            + self.total_variance(k - h, T)
        ) / h ** 2

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
