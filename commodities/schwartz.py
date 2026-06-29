"""
commodities/schwartz.py
Schwartz (1997) one-factor mean-reversion model for commodity spot prices.

Under the risk-neutral measure Q, the log-spot x = ln(S) follows:
    dx = κ(μ* − x) dt + σ dW

where:
    κ   — speed of mean-reversion (> 0)
    μ*  — risk-neutral long-run mean of ln(S)
    σ   — volatility of ln(S)

Futures price (closed-form):
    F(0, T) = exp[x₀ e^{−κT} + μ*(1 − e^{−κT}) + σ²/(4κ)(1 − e^{−2κT})]

where x₀ = ln(S₀).

As T → ∞: F → exp(μ* + σ²/(4κ))   (long-run futures price)
As T → 0:  F → S₀

Calibration: fit κ, μ*, σ to observed futures prices by nonlinear least squares.

Spot price option (Heston-like closed-form is complex;
here we use Monte Carlo simulation of the OU process).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm


@dataclass
class SchwartzParams:
    kappa:  float   # mean-reversion speed (> 0)
    mu_star:float   # risk-neutral long-run mean of ln(S)
    sigma:  float   # vol of ln(S) (> 0)

    def long_run_price(self) -> float:
        """E[S_∞] under Q = exp(μ* + σ²/(4κ))"""
        return float(np.exp(self.mu_star + self.sigma**2 / (4 * self.kappa)))

    def half_life(self) -> float:
        """Half-life of mean reversion: ln(2)/κ (years)"""
        return float(np.log(2) / self.kappa)

    def __str__(self) -> str:
        return (f"Schwartz1F: κ={self.kappa:.4f}  μ*={self.mu_star:.4f}  "
                f"σ={self.sigma:.4f}  t½={self.half_life():.2f}y  "
                f"F∞={self.long_run_price():.3f}")


# ── Futures pricing ───────────────────────────────────────────────────────────

def futures_price(S0: float, T: float | np.ndarray,
                  params: SchwartzParams) -> np.ndarray:
    """
    F(0, T) = exp[x₀ e^{−κT} + μ*(1 − e^{−κT}) + σ²/(4κ)(1 − e^{−2κT})]
    """
    T     = np.asarray(T, dtype=float)
    x0    = np.log(S0)
    kappa = params.kappa
    mu    = params.mu_star
    sigma = params.sigma

    lnF = (x0 * np.exp(-kappa * T)
           + mu * (1 - np.exp(-kappa * T))
           + sigma**2 / (4 * kappa) * (1 - np.exp(-2 * kappa * T)))
    return np.exp(lnF)


def futures_curve_schwartz(S0: float, maturities: np.ndarray,
                            params: SchwartzParams) -> np.ndarray:
    """Convenience wrapper: model futures curve."""
    return futures_price(S0, maturities, params)


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_schwartz(observed_futures: np.ndarray, maturities: np.ndarray,
                        S0: float, weights: np.ndarray | None = None,
                        method: str = "de") -> tuple[SchwartzParams, float]:
    """
    Calibrate Schwartz 1F model to observed futures prices.

    Parameters
    ----------
    observed_futures : array of market futures prices
    maturities       : corresponding maturities in years
    S0               : current spot price
    weights          : optional per-observation weights (e.g. 1/T for short-end fit)
    method           : "de" (differential evolution) or "nm" (Nelder-Mead)

    Returns
    -------
    (SchwartzParams, rmse)
    """
    T   = np.asarray(maturities, dtype=float)
    F_m = np.asarray(observed_futures, dtype=float)
    w   = np.ones(len(T)) if weights is None else np.asarray(weights)

    def objective(x):
        kappa_r, mu_r, sigma_r = x
        kappa = np.exp(kappa_r)     # > 0
        sigma = np.exp(sigma_r)     # > 0
        mu    = mu_r                # unconstrained
        params = SchwartzParams(kappa, mu, sigma)
        try:
            F_model = futures_price(S0, T, params)
            err = F_model - F_m
            return float(np.sum(w * err**2))
        except Exception:
            return 1e10

    x0 = [np.log(0.5), np.log(S0), np.log(0.20)]

    if method == "de":
        bounds = [(-3.0, 3.0), (np.log(S0 * 0.3), np.log(S0 * 3.0)), (-4.0, 0.0)]
        res    = differential_evolution(objective, bounds, seed=0,
                                         maxiter=500, tol=1e-10, popsize=12)
    else:
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 10_000, "xatol": 1e-8, "fatol": 1e-10})

    kappa_r, mu_r, sigma_r = res.x
    params = SchwartzParams(kappa=float(np.exp(kappa_r)),
                             mu_star=float(mu_r),
                             sigma=float(np.exp(sigma_r)))
    F_fit  = futures_price(S0, T, params)
    rmse   = float(np.sqrt(np.mean((F_fit - F_m)**2)))
    return params, rmse


# ── Monte Carlo simulation ────────────────────────────────────────────────────

def simulate_schwartz(S0: float, T: float, params: SchwartzParams,
                       n_sims: int = 10_000, n_steps: int = 252,
                       seed: int = 42) -> np.ndarray:
    """
    Simulate spot price paths under Schwartz 1F model.

    Returns
    -------
    paths : array of shape (n_sims, n_steps+1)
    """
    rng   = np.random.default_rng(seed)
    dt    = T / n_steps
    kappa = params.kappa
    mu    = params.mu_star
    sigma = params.sigma

    # Exact simulation of OU process:
    # x_{t+dt} = x_t·e^{-κdt} + μ*(1−e^{-κdt}) + σ√((1−e^{-2κdt})/(2κ))·Z
    e_kdt    = np.exp(-kappa * dt)
    std_step = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))

    x = np.log(S0) * np.ones(n_sims)
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0

    for i in range(n_steps):
        Z      = rng.standard_normal(n_sims)
        x      = x * e_kdt + mu * (1 - e_kdt) + std_step * Z
        paths[:, i + 1] = np.exp(x)

    return paths


# ── Option pricing via MC ─────────────────────────────────────────────────────

def price_commodity_option(S0: float, K: float, T: float,
                            r: float, params: SchwartzParams,
                            option_type: str = "call",
                            n_sims: int = 50_000, n_steps: int = 100,
                            seed: int = 42) -> dict:
    """
    European commodity option price under Schwartz 1F via Monte Carlo.

    Parameters
    ----------
    S0     : current spot
    K      : strike
    T      : years to expiry
    r      : domestic risk-free rate (for discounting)
    option_type : "call" | "put"
    """
    paths = simulate_schwartz(S0, T, params, n_sims, n_steps, seed)
    S_T   = paths[:, -1]

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0)
    else:
        payoff = np.maximum(K - S_T, 0)

    discounted = payoff * np.exp(-r * T)
    price      = float(np.mean(discounted))
    se         = float(np.std(discounted) / np.sqrt(n_sims))

    return {
        "price":       price,
        "std_error":   se,
        "conf_95_lo":  price - 1.96 * se,
        "conf_95_hi":  price + 1.96 * se,
        "spot_mean_T": float(np.mean(S_T)),
        "spot_std_T":  float(np.std(S_T)),
    }


# ── Model diagnostics ─────────────────────────────────────────────────────────

def fit_summary(S0: float, maturities: np.ndarray, futures_mkt: np.ndarray,
                params: SchwartzParams) -> str:
    F_fit = futures_price(S0, maturities, params)
    err   = F_fit - futures_mkt
    lines = [
        f"Schwartz 1F Model",
        f"  {params}",
        f"  RMSE: {np.sqrt(np.mean(err**2)):.4f}",
        f"  MAE : {np.mean(np.abs(err)):.4f}",
        f"",
        f"  {'T':>6}  {'F_mkt':>10}  {'F_model':>10}  {'Error':>8}",
    ]
    for T, Fm, Ff, e in zip(maturities, futures_mkt, F_fit, err):
        lines.append(f"  {T:>6.3f}  {Fm:>10.3f}  {Ff:>10.3f}  {e:>+8.3f}")
    return "\n".join(lines)
