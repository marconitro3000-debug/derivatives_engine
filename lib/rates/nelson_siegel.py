"""
rates/nelson_siegel.py
Nelson-Siegel (1987) and Svensson (1994) yield curve models.

Nelson-Siegel:
    r(t) = β₀ + β₁ · f(t, λ) + β₂ · g(t, λ)
    f(t,λ) = (1 − e^{−t/λ}) / (t/λ)       (loading on slope)
    g(t,λ) = f(t,λ) − e^{−t/λ}             (loading on curvature)

Interpretation:
    β₀ : long-run level (limit as t→∞)
    β₁ : slope (β₀ + β₁ = instantaneous short rate)
    β₂ : curvature / hump
    λ  : decay parameter (controls where hump peaks)

Svensson (two-hump extension):
    r(t) = β₀ + β₁·f(t,λ₁) + β₂·g(t,λ₁) + β₃·g(t,λ₂)

Both are fit by non-linear least squares to observed zero rates.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from typing import Optional


# ── Nelson-Siegel ─────────────────────────────────────────────────────────────

@dataclass
class NSParams:
    beta0: float   # long-run level
    beta1: float   # slope
    beta2: float   # curvature
    lam:   float   # decay (> 0)

    @property
    def short_rate(self) -> float:
        """Instantaneous short rate = β₀ + β₁"""
        return self.beta0 + self.beta1

    @property
    def long_rate(self) -> float:
        """Long-run rate = β₀"""
        return self.beta0


@dataclass
class SvenssonParams:
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    lam1:  float
    lam2:  float

    @property
    def short_rate(self) -> float:
        return self.beta0 + self.beta1

    @property
    def long_rate(self) -> float:
        return self.beta0


def _f(t: np.ndarray, lam: float) -> np.ndarray:
    """Loading function f(t,λ) = (1 − e^{−t/λ}) / (t/λ). Handle t→0 limit."""
    x = t / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(x < 1e-8, 1.0 - x / 2.0, (1.0 - np.exp(-x)) / x)
    return result


def _g(t: np.ndarray, lam: float) -> np.ndarray:
    """Curvature loading g(t,λ) = f(t,λ) − e^{−t/λ}"""
    return _f(t, lam) - np.exp(-t / lam)


def ns_yield(t: np.ndarray, params: NSParams) -> np.ndarray:
    """Nelson-Siegel zero rate at maturities t."""
    t = np.asarray(t, dtype=float)
    return (params.beta0
            + params.beta1 * _f(t, params.lam)
            + params.beta2 * _g(t, params.lam))


def svensson_yield(t: np.ndarray, params: SvenssonParams) -> np.ndarray:
    """Svensson zero rate at maturities t."""
    t = np.asarray(t, dtype=float)
    return (params.beta0
            + params.beta1 * _f(t, params.lam1)
            + params.beta2 * _g(t, params.lam1)
            + params.beta3 * _g(t, params.lam2))


def ns_discount_factor(t: np.ndarray, params: NSParams) -> np.ndarray:
    """P(0, t) = exp(−r(t) · t)"""
    t = np.asarray(t, dtype=float)
    return np.exp(-ns_yield(t, params) * t)


def svensson_discount_factor(t: np.ndarray, params: SvenssonParams) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.exp(-svensson_yield(t, params) * t)


# ── calibration ───────────────────────────────────────────────────────────────

def fit_ns(maturities: np.ndarray, zero_rates: np.ndarray,
           weights: np.ndarray = None,
           lam_bounds: tuple[float, float] = (0.1, 10.0)) -> NSParams:
    """
    Fit Nelson-Siegel to observed zero rates via non-linear least squares.

    Parameters
    ----------
    maturities : array of maturities (years)
    zero_rates : array of zero rates (decimal, e.g. 0.04 = 4%)
    weights    : optional observation weights
    lam_bounds : search bounds for λ

    Returns
    -------
    NSParams
    """
    T  = np.asarray(maturities, dtype=float)
    r  = np.asarray(zero_rates,  dtype=float)
    w  = np.ones(len(T)) if weights is None else np.asarray(weights)

    def objective(x):
        b0, b1, b2, lam = x
        if lam <= 0:
            return 1e10
        try:
            p   = NSParams(b0, b1, b2, lam)
            fit = ns_yield(T, p)
            return float(np.sum(w * (fit - r) ** 2))
        except Exception:
            return 1e10

    # Grid search over λ + OLS for betas
    best_result, best_obj = None, np.inf
    for lam_try in np.linspace(lam_bounds[0], lam_bounds[1], 20):
        # Linear OLS for betas given lam
        F1 = _f(T, lam_try)
        F2 = _g(T, lam_try)
        X  = np.column_stack([np.ones_like(T), F1, F2])
        try:
            betas, *_ = np.linalg.lstsq(X * w[:, None], r * w, rcond=None)
            x0 = [betas[0], betas[1], betas[2], lam_try]
            val = objective(x0)
            if val < best_obj:
                best_obj    = val
                best_result = x0
        except Exception:
            continue

    res = minimize(objective, best_result, method="Nelder-Mead",
                   options={"maxiter": 10_000, "xatol": 1e-10, "fatol": 1e-12})
    x = res.x
    return NSParams(beta0=float(x[0]), beta1=float(x[1]),
                    beta2=float(x[2]), lam=float(abs(x[3])))


def fit_svensson(maturities: np.ndarray, zero_rates: np.ndarray,
                  weights: np.ndarray = None) -> SvenssonParams:
    """
    Fit Svensson model to observed zero rates.
    Uses differential evolution for global optimization (6 free parameters).
    """
    T = np.asarray(maturities, dtype=float)
    r = np.asarray(zero_rates,  dtype=float)
    w = np.ones(len(T)) if weights is None else np.asarray(weights)

    bounds = [
        (-0.10, 0.20),   # beta0
        (-0.15, 0.15),   # beta1
        (-0.20, 0.20),   # beta2
        (-0.20, 0.20),   # beta3
        (0.05,  10.0),   # lam1
        (0.05,  10.0),   # lam2
    ]

    def objective(x):
        b0, b1, b2, b3, l1, l2 = x
        if l1 <= 0 or l2 <= 0 or abs(l1 - l2) < 0.05:
            return 1e10
        try:
            p   = SvenssonParams(b0, b1, b2, b3, l1, l2)
            fit = svensson_yield(T, p)
            return float(np.sum(w * (fit - r) ** 2))
        except Exception:
            return 1e10

    res = differential_evolution(objective, bounds, seed=0,
                                  maxiter=1000, tol=1e-10,
                                  popsize=12, mutation=(0.5, 1.5))
    x = res.x
    return SvenssonParams(
        beta0=float(x[0]), beta1=float(x[1]),
        beta2=float(x[2]), beta3=float(x[3]),
        lam1=float(x[4]),  lam2=float(x[5]),
    )


# ── diagnostics ───────────────────────────────────────────────────────────────

def fit_summary(maturities: np.ndarray, zero_rates: np.ndarray,
                params, model: str = "NS") -> str:
    """Print fit quality."""
    T   = np.asarray(maturities, dtype=float)
    r   = np.asarray(zero_rates,  dtype=float)
    fit = ns_yield(T, params) if model == "NS" else svensson_yield(T, params)
    err = (fit - r) * 10_000   # in bps

    if model == "NS":
        p = params
        lines = [
            f"Nelson-Siegel Fit:",
            f"  β₀  = {p.beta0:.6f}  ({p.beta0*100:.4f}%)",
            f"  β₁  = {p.beta1:.6f}",
            f"  β₂  = {p.beta2:.6f}",
            f"  λ   = {p.lam:.4f} years",
            f"  Short rate: {p.short_rate*100:.4f}%",
            f"  Long rate:  {p.long_rate*100:.4f}%",
        ]
    else:
        p = params
        lines = [
            f"Svensson Fit:",
            f"  β₀={p.beta0:.5f}  β₁={p.beta1:.5f}  β₂={p.beta2:.5f}  β₃={p.beta3:.5f}",
            f"  λ₁={p.lam1:.4f}  λ₂={p.lam2:.4f}",
            f"  Short rate: {p.short_rate*100:.4f}%",
        ]

    lines += [
        f"",
        f"  Fit errors (bps):",
        f"    RMSE:  {np.sqrt(np.mean(err**2)):.3f}",
        f"    MAE:   {np.mean(np.abs(err)):.3f}",
        f"    Max:   {np.max(np.abs(err)):.3f}",
    ]
    for t, r_obs, r_fit, e in zip(T, r * 100, fit * 100, err):
        lines.append(f"    t={t:.1f}y: mkt={r_obs:.4f}%  fit={r_fit:.4f}%  err={e:+.2f}bps")
    return "\n".join(lines)
