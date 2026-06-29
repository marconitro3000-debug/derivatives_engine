"""
volatility/garch.py
GARCH(1,1) model: maximum-likelihood estimation, forecasting, simulation.

Dynamics:
    r_t = μ + ε_t,    ε_t = σ_t · z_t,    z_t ~ N(0,1)
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Constraints (stationarity):  ω > 0,  α ≥ 0,  β ≥ 0,  α + β < 1
Long-run variance:  σ²_∞ = ω / (1 − α − β)
h-step forecast:    E[σ²_{t+h}] = σ²_∞ + (α+β)^h · (σ²_t − σ²_∞)

MLE: maximise  L = −½ Σ_t [ln σ²_t + ε²_t / σ²_t]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

_TRADING_DAYS = 252


@dataclass
class GARCHResult:
    """Fitted GARCH(1,1) parameters and diagnostics."""
    mu:    float
    omega: float
    alpha: float
    beta:  float
    loglik:   float
    aic:      float
    bic:      float
    n_obs:    int
    converged: bool
    conditional_vol: pd.Series   # daily annualized vol fitted series
    residuals: pd.Series         # standardised residuals z_t = ε_t / σ_t

    @property
    def long_run_vol(self) -> float:
        """Annualized long-run (unconditional) vol = √(ω/(1−α−β)) × √252"""
        lr_var = self.omega / (1.0 - self.alpha - self.beta)
        return float(np.sqrt(lr_var * _TRADING_DAYS))

    @property
    def persistence(self) -> float:
        """α + β: how quickly shocks decay. Close to 1 = long memory."""
        return float(self.alpha + self.beta)

    @property
    def half_life(self) -> float:
        """Shocks decay to half their size in ln(0.5)/ln(α+β) days."""
        ab = self.persistence
        if ab >= 1.0 or ab <= 0.0:
            return np.inf
        return float(np.log(0.5) / np.log(ab))

    def summary(self) -> str:
        lines = [
            "GARCH(1,1) Fit",
            f"  n_obs     = {self.n_obs}",
            f"  converged = {self.converged}",
            f"  μ         = {self.mu:.6f}  ({self.mu * _TRADING_DAYS:.4%} ann.)",
            f"  ω         = {self.omega:.2e}",
            f"  α         = {self.alpha:.6f}",
            f"  β         = {self.beta:.6f}",
            f"  α+β       = {self.persistence:.6f}  (persistence)",
            f"  Long-run vol = {self.long_run_vol:.2%}",
            f"  Half-life    = {self.half_life:.1f} days",
            f"  LogLik    = {self.loglik:.4f}",
            f"  AIC       = {self.aic:.4f}",
            f"  BIC       = {self.bic:.4f}",
        ]
        return "\n".join(lines)


# ── internal: log-likelihood ──────────────────────────────────────────────────

def _garch_filter(params: np.ndarray, returns: np.ndarray):
    """
    Compute conditional variances and log-likelihood for GARCH(1,1).

    params = [mu, log(omega), logit_alpha, logit_beta_scaled]
    Transformation ensures omega>0, alpha>=0, beta>=0, alpha+beta<1.
    """
    mu        = params[0]
    omega     = np.exp(params[1])
    # alpha in (0, 1), beta in (0, 1-alpha)
    alpha_raw = 1.0 / (1.0 + np.exp(-params[2]))
    beta_raw  = 1.0 / (1.0 + np.exp(-params[3]))
    alpha = alpha_raw * 0.99
    beta  = beta_raw  * (1.0 - alpha) * 0.99

    n    = len(returns)
    eps  = returns - mu
    var  = np.empty(n)
    var[0] = omega / max(1.0 - alpha - beta, 1e-8)   # initialize at LR variance

    for t in range(1, n):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        var[t] = max(var[t], 1e-12)

    loglik = -0.5 * np.sum(np.log(var) + eps ** 2 / var)
    return var, loglik, alpha, beta, omega, mu


def _neg_loglik(params, returns):
    _, loglik, *_ = _garch_filter(params, returns)
    return -loglik


# ── public API ────────────────────────────────────────────────────────────────

def fit(returns: pd.Series, mu0: float = None) -> GARCHResult:
    """
    Fit GARCH(1,1) to a series of log-returns via MLE.

    Parameters
    ----------
    returns : pd.Series of daily log-returns (not annualized)
    mu0     : fixed mean (if None, estimated jointly)

    Returns
    -------
    GARCHResult
    """
    r = np.asarray(returns.dropna(), dtype=float)
    n = len(r)
    if n < 50:
        raise ValueError("Need at least 50 observations to fit GARCH(1,1)")

    # Initial parameters
    var0  = float(np.var(r))
    mu_i  = float(np.mean(r)) if mu0 is None else mu0

    # Transform: log(omega), logit(alpha), logit(beta_scaled)
    # ω_init = var0 * 0.05, α_init = 0.10, β_init = 0.85
    omega_i = var0 * 0.05
    alpha_i = 0.10
    beta_i  = 0.85 / (1.0 - alpha_i)   # scaled

    x0 = np.array([mu_i,
                   np.log(omega_i),
                   np.log(alpha_i / (1.0 - alpha_i)),
                   np.log(beta_i  / (1.0 - beta_i))])

    result = minimize(_neg_loglik, x0, args=(r,),
                      method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})

    var_t, loglik, alpha, beta, omega, mu = _garch_filter(result.x, r)

    k   = 4   # mu, omega, alpha, beta
    aic = -2 * loglik + 2 * k
    bic = -2 * loglik + k * np.log(n)

    # Build aligned series on the original index (dropna may have shifted)
    idx = returns.dropna().index
    cond_var = pd.Series(var_t, index=idx, name="conditional_variance")
    cond_vol = (np.sqrt(cond_var * _TRADING_DAYS)).rename("conditional_vol")
    resid    = pd.Series((r - mu) / np.sqrt(var_t), index=idx, name="std_residuals")

    return GARCHResult(
        mu=float(mu), omega=float(omega),
        alpha=float(alpha), beta=float(beta),
        loglik=float(loglik), aic=float(aic), bic=float(bic),
        n_obs=n, converged=result.success,
        conditional_vol=cond_vol,
        residuals=resid,
    )


def forecast(result: GARCHResult, h: int = 30,
             last_var: float = None) -> pd.DataFrame:
    """
    h-step ahead variance and vol forecast from fitted GARCH(1,1).

    E[σ²_{t+h}] = σ²_∞ + (α+β)^h · (σ²_t − σ²_∞)

    Parameters
    ----------
    result   : GARCHResult from fit()
    h        : forecast horizon (days)
    last_var : starting variance (daily); if None, uses last fitted value

    Returns
    -------
    DataFrame with columns: forecast_var, forecast_vol (annualized),
                             vol_lb_95, vol_ub_95
    (CI is approximate: using Engle & Bollerslev delta-method approximation)
    """
    om, a, b = result.omega, result.alpha, result.beta
    lr_var   = om / (1.0 - a - b)
    ab       = a + b

    if last_var is None:
        # Daily conditional variance from last observation
        last_var = float((result.conditional_vol.iloc[-1] / np.sqrt(_TRADING_DAYS)) ** 2)

    steps = np.arange(1, h + 1)
    fcast_var = lr_var + ab ** steps * (last_var - lr_var)
    fcast_vol = np.sqrt(np.maximum(fcast_var, 0) * _TRADING_DAYS)

    # Approximate 95% interval via simulation-based variance of forecast
    # Var[σ²_{t+h}] ≈ 2(α+β)^{2h} × σ⁴_t / (1 − (α+β)²)  (simplified)
    ab2     = ab ** 2
    var_fcast = 2 * ab ** (2 * steps) * last_var ** 2 / max(1.0 - ab2, 1e-8)
    std_fcast = np.sqrt(np.maximum(var_fcast, 0))
    vol_std   = std_fcast * _TRADING_DAYS / (2.0 * np.sqrt(np.maximum(fcast_var, 1e-12) * _TRADING_DAYS))

    lb = np.maximum(fcast_vol - 1.96 * vol_std, 0)
    ub = fcast_vol + 1.96 * vol_std

    return pd.DataFrame({
        "horizon":      steps,
        "forecast_var": fcast_var,
        "forecast_vol": fcast_vol,
        "vol_lb_95":    lb,
        "vol_ub_95":    ub,
    })


def simulate(result: GARCHResult, n: int = 252,
             n_paths: int = 1, seed: int = None) -> np.ndarray:
    """
    Simulate log-return paths from fitted GARCH(1,1).

    Parameters
    ----------
    n       : number of periods
    n_paths : number of Monte Carlo paths

    Returns
    -------
    ndarray of shape (n_paths, n) — simulated daily log-returns
    """
    rng  = np.random.default_rng(seed)
    om, a, b, mu = result.omega, result.alpha, result.beta, result.mu

    last_var = float((result.conditional_vol.iloc[-1] / np.sqrt(_TRADING_DAYS)) ** 2)
    paths    = np.empty((n_paths, n))

    for p in range(n_paths):
        var_t = last_var
        for t in range(n):
            z = rng.standard_normal()
            eps = np.sqrt(max(var_t, 1e-12)) * z
            paths[p, t] = mu + eps
            var_t = om + a * eps ** 2 + b * var_t

    return paths
