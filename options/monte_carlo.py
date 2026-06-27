"""
options/monte_carlo.py
Monte Carlo pricing under GBM with variance-reduction techniques.

Supported payoffs
-----------------
  european_call / european_put   — plain vanilla
  asian_call    / asian_put      — arithmetic average strike
  barrier_call  / barrier_put    — down-and-out (knock-out on minimum path)
  lookback_call / lookback_put   — floating strike (call: S_T - S_min, put: S_max - S_T)
  digital_call  / digital_put    — cash-or-nothing ($1 payoff)

Variance-reduction techniques
------------------------------
  antithetic  : simulate Z and -Z pairs  → halves variance of the estimator
  control     : use European BS price as control variate → large further reduction
"""

import numpy as np
from typing import Literal

from .black_scholes import price as bs_price


# ── payoff functions ──────────────────────────────────────────────────────────

def _payoff(paths: np.ndarray, K: float, option_type: str,
            barrier: float | None = None) -> np.ndarray:
    """
    Compute terminal payoffs for an array of simulated paths.

    Parameters
    ----------
    paths       : shape (n_sims, n_steps+1) — price paths including t=0
    K           : strike
    option_type : one of the supported payoff types (see module docstring)
    barrier     : knock-out barrier level (only for barrier options)
    """
    S_T   = paths[:, -1]
    S_min = paths.min(axis=1)
    S_max = paths.max(axis=1)
    S_avg = paths[:, 1:].mean(axis=1)   # exclude t=0

    if option_type == "european_call":
        payoffs = np.maximum(S_T - K, 0)

    elif option_type == "european_put":
        payoffs = np.maximum(K - S_T, 0)

    elif option_type == "asian_call":
        payoffs = np.maximum(S_avg - K, 0)

    elif option_type == "asian_put":
        payoffs = np.maximum(K - S_avg, 0)

    elif option_type == "barrier_call":
        if barrier is None:
            raise ValueError("barrier_call requires a barrier level.")
        knocked_out = S_min <= barrier
        payoffs = np.where(knocked_out, 0.0, np.maximum(S_T - K, 0))

    elif option_type == "barrier_put":
        if barrier is None:
            raise ValueError("barrier_put requires a barrier level.")
        knocked_out = S_min <= barrier
        payoffs = np.where(knocked_out, 0.0, np.maximum(K - S_T, 0))

    elif option_type == "lookback_call":
        payoffs = np.maximum(S_T - S_min, 0)

    elif option_type == "lookback_put":
        payoffs = np.maximum(S_max - S_T, 0)

    elif option_type == "digital_call":
        payoffs = (S_T > K).astype(float)

    elif option_type == "digital_put":
        payoffs = (S_T < K).astype(float)

    else:
        raise ValueError(f"Unknown option_type: '{option_type}'.")

    return payoffs


# ── GBM path simulator ────────────────────────────────────────────────────────

def _simulate_gbm(S: float, T: float, r: float, sigma: float,
                  n_sims: int, n_steps: int,
                  antithetic: bool, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate GBM paths under the risk-neutral measure.

    Returns paths of shape (n_sims, n_steps+1).
    If antithetic=True, returns 2*n_sims paths (first half normal, second half flipped).
    """
    dt    = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)

    half = n_sims // 2 if antithetic else n_sims
    Z    = rng.standard_normal((half, n_steps))

    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)

    log_returns = drift + vol * Z
    log_paths   = np.cumsum(log_returns, axis=1)
    paths       = S * np.exp(np.hstack([np.zeros((len(Z), 1)), log_paths]))
    return paths


# ── control variate correction ────────────────────────────────────────────────

def _apply_control_variate(payoffs_mc: np.ndarray, paths: np.ndarray,
                            S: float, K: float, T: float, r: float,
                            sigma: float) -> np.ndarray:
    """
    Apply European call as control variate to reduce variance.

    Adjusts raw payoffs so the European estimator matches the BS analytical price.
    """
    S_T          = paths[:, -1]
    disc         = np.exp(-r * T)
    euro_payoffs = np.maximum(S_T - K, 0)
    euro_mc_mean = disc * euro_payoffs.mean()
    euro_bs      = bs_price(S, K, T, r, sigma, "call")

    cov_xe  = np.cov(payoffs_mc, euro_payoffs)[0, 1]
    var_e   = np.var(euro_payoffs, ddof=1)
    if var_e < 1e-14:
        return payoffs_mc

    beta = cov_xe / var_e
    return payoffs_mc - beta * (euro_payoffs - euro_mc_mean / disc)


# ── public API ────────────────────────────────────────────────────────────────

def mc_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "european_call",
             n_sims: int = 100_000,
             n_steps: int = 252,
             antithetic: bool = True,
             control_variate: bool = False,
             barrier: float | None = None,
             seed: int | None = 42) -> dict:
    """
    Price an option via Monte Carlo simulation.

    Parameters
    ----------
    S               : underlying price
    K               : strike
    T               : time to expiry (years)
    r               : risk-free rate
    sigma           : volatility
    option_type     : payoff type (see module docstring)
    n_sims          : number of simulated paths
    n_steps         : time steps per path
    antithetic      : use antithetic variates (halves variance)
    control_variate : use European BS as control variate (further reduces variance)
    barrier         : knock-out level (only for barrier options)
    seed            : random seed for reproducibility

    Returns
    -------
    dict with keys:
      price       : MC estimate
      std_error   : standard error of estimate
      conf_95_lo  : 95% CI lower bound
      conf_95_hi  : 95% CI upper bound
      n_sims      : actual number of paths used
    """
    rng   = np.random.default_rng(seed)
    paths = _simulate_gbm(S, T, r, sigma, n_sims, n_steps, antithetic, rng)
    disc  = np.exp(-r * T)

    payoffs = _payoff(paths, K, option_type, barrier)

    if control_variate and "european" not in option_type:
        payoffs = _apply_control_variate(payoffs, paths, S, K, T, r, sigma)

    discounted = disc * payoffs
    mean       = discounted.mean()
    se         = discounted.std(ddof=1) / np.sqrt(len(discounted))

    return {
        "price":      float(mean),
        "std_error":  float(se),
        "conf_95_lo": float(mean - 1.96 * se),
        "conf_95_hi": float(mean + 1.96 * se),
        "n_sims":     len(paths),
    }
