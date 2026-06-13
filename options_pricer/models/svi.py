"""
models/svi.py
SVI (Stochastic Volatility Inspired) parametrisation of the implied vol smile.

Gatheral's raw SVI: for a single maturity slice, total implied variance w(k) as a
function of log-moneyness k = ln(K/F):

    w(k) = a + b * ( rho*(k - m) + sqrt( (k - m)^2 + sigma^2 ) )

where:
    a     : overall level of variance        (a >= 0)
    b     : slope of the wings               (b >= 0)
    rho   : skew / rotation                   (-1 < rho < 1)
    m     : horizontal shift (smile centre)
    sigma : curvature / ATM smoothness        (sigma > 0)

Implied vol for that slice:  iv(k) = sqrt( w(k) / T )
"""

import numpy as np
from dataclasses import dataclass, asdict


# ── parameter container ───────────────────────────────────────────────────────

@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, arr) -> "SVIParams":
        return cls(*[float(x) for x in arr])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SVIParams":
        return cls(a=d["a"], b=d["b"], rho=d["rho"], m=d["m"], sigma=d["sigma"])


# ── core formula ──────────────────────────────────────────────────────────────

def total_variance(k: np.ndarray, p: SVIParams) -> np.ndarray:
    """Total implied variance w(k) for log-moneyness array k."""
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma ** 2))


def implied_vol_svi(k: np.ndarray, T: float, p: SVIParams) -> np.ndarray:
    """Implied vol from SVI slice: iv = sqrt(w/T)."""
    w = np.maximum(total_variance(k, p), 1e-12)
    return np.sqrt(w / T)


# ── no-arbitrage check (Gatheral-Jacquier) ────────────────────────────────────

def is_butterfly_arbitrage_free(p: SVIParams) -> bool:
    """
    Lee's wing condition and basic positivity: necessary (not full) check that
    the slice is free of static butterfly arbitrage.
    """
    if p.b < 0 or p.sigma <= 0 or abs(p.rho) >= 1:
        return False
    # wing slopes must satisfy b*(1+|rho|) <= 4 (Lee's moment formula bound)
    if p.b * (1 + abs(p.rho)) > 4.0:
        return False
    # minimum total variance must be non-negative
    w_min = p.a + p.b * p.sigma * np.sqrt(1 - p.rho ** 2)
    return w_min >= 0


# ── default seed ──────────────────────────────────────────────────────────────

def default_params(atm_var: float = 0.04) -> SVIParams:
    """Reasonable seed for calibration when no warm start is available."""
    return SVIParams(a=atm_var * 0.5, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
