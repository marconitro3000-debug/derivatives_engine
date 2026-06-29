"""
portfolio/var.py
Value-at-Risk and Conditional VaR (Expected Shortfall).

Methods
-------
Historical simulation  — empirical quantile of P&L distribution
Parametric (normal)    — Gaussian closed-form
Monte Carlo            — full simulation of portfolio P&L
Cornish-Fisher         — skewness/kurtosis correction to normal VaR

All VaR figures are positive numbers representing potential loss.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class VaRResult:
    method: str
    confidence: float
    horizon_days: int
    var: float          # loss (positive)
    cvar: float         # expected shortfall (positive)
    mean_pnl: float
    std_pnl: float
    pnl_series: np.ndarray

    def __str__(self) -> str:
        h = self.horizon_days
        c = self.confidence * 100
        lines = [
            f"VaR ({self.method})  {c:.0f}%  {h}d horizon",
            f"  VaR    : {self.var:>12,.2f}",
            f"  CVaR   : {self.cvar:>12,.2f}",
            f"  E[PnL] : {self.mean_pnl:>12,.2f}",
            f"  σ[PnL] : {self.std_pnl:>12,.2f}",
        ]
        return "\n".join(lines)


# ── core functions ────────────────────────────────────────────────────────────

def var_historical(pnl: np.ndarray, confidence: float = 0.99,
                   horizon_days: int = 1) -> VaRResult:
    """
    Historical simulation VaR.

    Parameters
    ----------
    pnl        : array of daily P&L observations (positive = profit)
    confidence : VaR confidence level (e.g. 0.99)
    horizon_days : scaling horizon (square-root of time)
    """
    pnl = np.asarray(pnl, dtype=float)
    scale = np.sqrt(horizon_days)
    q     = np.quantile(pnl, 1 - confidence) * scale    # negative
    var   = -q                                            # positive
    cvar  = -np.mean(pnl[pnl <= q]) * scale

    return VaRResult(
        method="Historical",
        confidence=confidence,
        horizon_days=horizon_days,
        var=float(var),
        cvar=float(cvar),
        mean_pnl=float(np.mean(pnl) * scale),
        std_pnl=float(np.std(pnl) * scale),
        pnl_series=pnl * scale,
    )


def var_parametric(pnl: np.ndarray, confidence: float = 0.99,
                   horizon_days: int = 1) -> VaRResult:
    """
    Parametric (normal distribution) VaR.

    VaR = -(μ + z_α × σ)  where z_α = N^{-1}(1-confidence)
    CVaR = -(μ - σ × φ(z_α) / (1-confidence))
    """
    pnl   = np.asarray(pnl, dtype=float)
    mu    = np.mean(pnl)
    sigma = np.std(pnl, ddof=1)
    scale = np.sqrt(horizon_days)

    z_a   = stats.norm.ppf(1 - confidence)
    var   = -(mu + z_a * sigma) * scale
    cvar  = -(mu - sigma * stats.norm.pdf(-z_a) / (1 - confidence)) * scale

    return VaRResult(
        method="Parametric",
        confidence=confidence,
        horizon_days=horizon_days,
        var=float(var),
        cvar=float(cvar),
        mean_pnl=float(mu * scale),
        std_pnl=float(sigma * scale),
        pnl_series=pnl * scale,
    )


def var_cornish_fisher(pnl: np.ndarray, confidence: float = 0.99,
                       horizon_days: int = 1) -> VaRResult:
    """
    Cornish-Fisher expansion: adjusts normal quantile for skewness and excess kurtosis.

    z_CF = z + (z²-1)/6 × S + (z³-3z)/24 × K − (2z³-5z)/36 × S²
    """
    pnl    = np.asarray(pnl, dtype=float)
    mu     = np.mean(pnl)
    sigma  = np.std(pnl, ddof=1)
    S      = stats.skew(pnl)
    K      = stats.kurtosis(pnl)   # excess kurtosis
    scale  = np.sqrt(horizon_days)

    z      = stats.norm.ppf(1 - confidence)
    z_cf   = (z
              + (z**2 - 1) / 6.0 * S
              + (z**3 - 3*z) / 24.0 * K
              - (2*z**3 - 5*z) / 36.0 * S**2)

    var    = -(mu + z_cf * sigma) * scale

    # CVaR via numerical integration of tail
    n_tail = 10_000
    u      = np.linspace(1 - confidence, 1.0 - 1e-8, n_tail)
    q_u    = mu + stats.norm.ppf(u) * sigma
    cvar   = -np.mean(q_u) * scale

    return VaRResult(
        method="Cornish-Fisher",
        confidence=confidence,
        horizon_days=horizon_days,
        var=float(var),
        cvar=float(cvar),
        mean_pnl=float(mu * scale),
        std_pnl=float(sigma * scale),
        pnl_series=pnl * scale,
    )


def var_monte_carlo(portfolio, confidence: float = 0.99,
                    horizon_days: int = 1,
                    n_sims: int = 100_000,
                    annual_vol: float = 0.20,
                    seed: int = 42) -> VaRResult:
    """
    MC simulation VaR for a Portfolio object.

    Simulates daily log-returns ~ N(0, σ_daily) and revalues the portfolio.

    Parameters
    ----------
    portfolio    : portfolio.position.Portfolio
    annual_vol   : annualized vol for the single risk factor
    """
    rng        = np.random.default_rng(seed)
    daily_vol  = annual_vol / np.sqrt(252)
    returns    = rng.normal(0, daily_vol * np.sqrt(horizon_days), n_sims)
    pnl        = portfolio.pnl_vector(returns)

    var_level  = np.quantile(pnl, 1 - confidence)
    var        = -var_level
    cvar       = -np.mean(pnl[pnl <= var_level])

    return VaRResult(
        method="Monte Carlo",
        confidence=confidence,
        horizon_days=horizon_days,
        var=float(var),
        cvar=float(cvar),
        mean_pnl=float(np.mean(pnl)),
        std_pnl=float(np.std(pnl)),
        pnl_series=pnl,
    )


# ── comparison ────────────────────────────────────────────────────────────────

def var_comparison(pnl: np.ndarray, confidence: float = 0.99,
                   horizon_days: int = 1) -> dict[str, VaRResult]:
    """Run all non-MC methods and return a dict keyed by method name."""
    return {
        "historical":     var_historical(pnl, confidence, horizon_days),
        "parametric":     var_parametric(pnl, confidence, horizon_days),
        "cornish_fisher": var_cornish_fisher(pnl, confidence, horizon_days),
    }


def backtest_var(pnl: np.ndarray, var_series: np.ndarray,
                 confidence: float = 0.99) -> dict:
    """
    Kupiec proportion-of-failures (POF) test for VaR model validity.

    H₀: true exceedance probability = 1 - confidence
    """
    n           = len(pnl)
    exceptions  = np.sum(pnl < -var_series)
    expected    = int(round(n * (1 - confidence)))
    exc_rate    = exceptions / n

    # Kupiec likelihood ratio test
    p0          = 1 - confidence
    p_hat       = exc_rate + 1e-15
    llr         = -2 * (
        exceptions * np.log(p0 / p_hat)
        + (n - exceptions) * np.log((1 - p0) / (1 - p_hat))
    )
    p_value     = 1 - stats.chi2.cdf(llr, df=1)

    return {
        "n_observations":  n,
        "exceptions":      int(exceptions),
        "expected":        expected,
        "exception_rate":  float(exc_rate),
        "llr_statistic":   float(llr),
        "p_value":         float(p_value),
        "pass":            bool(p_value > 0.05),
    }
