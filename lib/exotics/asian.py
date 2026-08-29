"""
exotics/asian.py
Asian (average-price) option pricing.

Payoff depends on the arithmetic mean of S over [0, T], not just S_T.

Geometric mean  → exact closed-form (lognormal average).
Arithmetic mean → Monte Carlo (primary) + Kemna-Vorst approximation.

Both fixed-strike and floating-strike variants are supported.
"""

import numpy as np
from scipy.stats import norm


# ── geometric asian: exact closed-form ───────────────────────────────────────

def price_asian_geo(S: float, K: float, T: float, r: float, sigma: float,
                    option_type: str = "call") -> dict:
    """
    Exact closed-form pricing for a European geometric-average Asian option
    (fixed strike, continuous monitoring).

    Derivation
    ----------
    The continuous geometric average of GBM satisfies:
        ln(G_T) ~ N(m, v²)
    where
        m  = ln S + (r - σ²/2)·T/2      (mean of log-average)
        v² = σ²·T/3                      (variance of log-average)

    This yields a BSM-like formula with adjusted parameters.
    """
    if T <= 0:
        raise ValueError("T must be positive")

    v   = sigma * np.sqrt(T / 3.0)                         # adjusted vol
    adj = np.log(S) + (r - 0.5 * sigma ** 2) * T / 2.0    # mean of ln G_T

    # Effective "spot" is the expected geometric average, discounted
    d1 = (adj - np.log(K) + v ** 2) / v
    d2 = d1 - v

    df  = np.exp(-r * T)
    fwd = np.exp(adj + 0.5 * v ** 2)  # E[G_T] under Q

    if option_type == "call":
        px = df * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        px = df * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))

    return {"price": float(px), "adj_vol": v, "fwd_geo": fwd}


# ── arithmetic asian: Monte Carlo ─────────────────────────────────────────────

def mc_asian_arith(S: float, K: float, T: float, r: float, sigma: float,
                   option_type: str = "call",
                   strike_type: str = "fixed",
                   n_sims: int = 100_000,
                   n_steps: int = 252,
                   seed: int = None) -> dict:
    """
    Monte Carlo pricing for arithmetic-average Asian options.

    Parameters
    ----------
    S, K, T, r, sigma : standard option parameters
    option_type  : 'call' or 'put'
    strike_type  : 'fixed'    → payoff = max(S_avg - K, 0)
                   'floating' → payoff = max(S_T - S_avg, 0) for call
                                         max(S_avg - S_T, 0) for put
    n_sims  : simulation paths
    n_steps : monitoring points (default 252 = daily for 1Y)
    seed    : RNG seed for reproducibility

    Returns
    -------
    dict: price, std_error, conf_95_lo, conf_95_hi, geo_price (for comparison)
    """
    rng    = np.random.default_rng(seed)
    dt     = T / n_steps
    drift  = (r - 0.5 * sigma ** 2) * dt
    vol    = sigma * np.sqrt(dt)

    Z       = rng.standard_normal((n_sims, n_steps))
    log_ret = drift + vol * Z
    log_S   = np.log(S) + np.hstack([np.zeros((n_sims, 1)), np.cumsum(log_ret, axis=1)])
    paths   = np.exp(log_S)   # shape (n_sims, n_steps+1), includes S_0

    # Average excludes S_0 (average over [dt, T])
    S_avg = paths[:, 1:].mean(axis=1)
    S_T   = paths[:, -1]

    if strike_type == "fixed":
        if option_type == "call":
            payoffs = np.maximum(S_avg - K, 0.0)
        else:
            payoffs = np.maximum(K - S_avg, 0.0)
    else:  # floating strike
        if option_type == "call":
            payoffs = np.maximum(S_T - S_avg, 0.0)
        else:
            payoffs = np.maximum(S_avg - S_T, 0.0)

    disc   = np.exp(-r * T) * payoffs
    price  = float(disc.mean())
    stderr = float(disc.std() / np.sqrt(n_sims))

    geo = price_asian_geo(S, K, T, r, sigma, option_type)["price"] \
          if strike_type == "fixed" else None

    return {
        "price":      price,
        "std_error":  stderr,
        "conf_95_lo": price - 1.96 * stderr,
        "conf_95_hi": price + 1.96 * stderr,
        "geo_price":  geo,
    }


# ── Kemna-Vorst approximation (arithmetic) ────────────────────────────────────

def price_asian_kv(S: float, K: float, T: float, r: float, sigma: float,
                   option_type: str = "call") -> dict:
    """
    Kemna-Vorst (1990) approximation for continuous arithmetic-average Asian.

    Idea: replace the arithmetic average with a shifted geometric average
    that has the same mean. This gives a closed-form via BSM with adjusted
    strike and parameters.

    Works well ATM / near-ATM. Accuracy degrades for deep OTM and high σ.
    """
    # Variance of ln(arithmetic average) ≈ σ²T/3 (geometric approx)
    # Adjusted parameters
    v  = sigma / np.sqrt(3.0)         # vol of geometric approx
    b  = 0.5 * (r - sigma ** 2 / 6)  # adjusted cost-of-carry

    F = S * np.exp(b * T)             # adjusted forward
    d1 = (np.log(F / K) + 0.5 * v ** 2 * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)

    df = np.exp(-r * T)
    if option_type == "call":
        px = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        px = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return {"price": float(px), "adj_vol": v, "adj_fwd": float(F)}


# ── unified entry point ───────────────────────────────────────────────────────

def price_asian(S: float, K: float, T: float, r: float, sigma: float,
                option_type: str = "call",
                average_type: str = "arithmetic",
                strike_type: str = "fixed",
                method: str = "mc",
                **kwargs) -> dict:
    """
    Unified Asian option pricer.

    Parameters
    ----------
    average_type : 'arithmetic' or 'geometric'
    strike_type  : 'fixed' or 'floating'
    method       : 'mc', 'kv' (Kemna-Vorst, arithmetic only), 'exact' (geometric only)
    """
    if average_type == "geometric" or method == "exact":
        return price_asian_geo(S, K, T, r, sigma, option_type)
    if method == "kv":
        return price_asian_kv(S, K, T, r, sigma, option_type)
    # default: MC
    return mc_asian_arith(S, K, T, r, sigma, option_type, strike_type, **kwargs)
