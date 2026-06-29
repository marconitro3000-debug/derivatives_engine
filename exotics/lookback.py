"""
exotics/lookback.py
Lookback option pricing.

Floating-strike lookback call   → payoff = S_T − min(S_t)   "bought at lowest"
Floating-strike lookback put    → payoff = max(S_t) − S_T   "sold at highest"
Fixed-strike lookback call      → payoff = max(max(S_t) − K, 0)
Fixed-strike lookback put       → payoff = max(K − min(S_t), 0)

Closed-form for continuous monitoring (Goldman-Sosin-Gatto 1979):
    Floating-strike call and put.

Monte Carlo for all types (used as cross-check and for discrete monitoring).
"""

import numpy as np
from scipy.stats import norm


# ── closed-form: floating-strike (Goldman-Sosin-Gatto 1979) ──────────────────

def price_lookback_float(S: float, T: float, r: float, sigma: float,
                         option_type: str = "call",
                         extremum: float = None) -> dict:
    """
    Closed-form floating-strike lookback option (continuous monitoring).

    For a fresh option: extremum = S (no history yet).
    For a seasoned option pass the observed min (call) or max (put) so far.

    Call payoff : S_T − min S_t     (you "bought at the lowest price")
    Put  payoff : max S_t − S_T     (you "sold at the highest price")

    Reference: Goldman, Sosin & Gatto (1979).
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if extremum is None:
        extremum = S     # fresh option

    sqrt_T = np.sqrt(T)
    a1 = (np.log(S / extremum) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    a2 = a1 - sigma * sqrt_T
    a3 = (np.log(S / extremum) - (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    # Note: a3 = a1 - 2r√T/σ  (for fresh option where S/extremum=1, a3 = -a2 when r>0)

    df    = np.exp(-r * T)
    ratio = (S / extremum) ** (-2 * r / sigma ** 2) if r > 1e-10 else 1.0

    if option_type == "call":
        # Floating call: S_T − S_min
        # C = S·N(a1) − S_min·e^{−rT}·N(a2) − S·(σ²/2r)·(N(−a1) − e^{−rT}·ratio·N(−a3))
        if r > 1e-10:
            correction = (sigma ** 2 / (2 * r)) * (norm.cdf(-a1) - df * ratio * norm.cdf(-a3))
        else:
            # r→0 limit: use L'Hôpital approximation
            correction = 0.5 * sigma ** 2 * T * norm.cdf(-a1)
        px = S * norm.cdf(a1) - extremum * df * norm.cdf(a2) - S * correction
    else:
        # Floating put: S_max − S_T
        # P = S_max·e^{−rT}·N(−a2) − S·N(−a1) + S·(σ²/2r)·(N(a1) − e^{−rT}·ratio·N(a3))
        if r > 1e-10:
            correction = (sigma ** 2 / (2 * r)) * (norm.cdf(a1) - df * ratio * norm.cdf(a3))
        else:
            correction = 0.5 * sigma ** 2 * T * norm.cdf(a1)
        px = extremum * df * norm.cdf(-a2) - S * norm.cdf(-a1) + S * correction

    return {"price": float(max(px, 0.0)), "extremum": extremum}


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def mc_lookback(S: float, T: float, r: float, sigma: float,
                option_type: str = "call",
                strike_type: str = "float",
                K: float = None,
                n_sims: int = 100_000,
                n_steps: int = 252,
                seed: int = None) -> dict:
    """
    Monte Carlo lookback option pricing with discrete monitoring.

    Parameters
    ----------
    S, T, r, sigma : standard parameters
    option_type    : 'call' or 'put'
    strike_type    : 'float' or 'fixed'
    K              : strike (required for 'fixed')
    n_sims         : number of paths
    n_steps        : monitoring steps (default 252 = daily for T=1Y)

    Payoffs
    -------
    float call : S_T − min_t S_t
    float put  : max_t S_t − S_T
    fixed call : max(max_t S_t − K, 0)
    fixed put  : max(K − min_t S_t, 0)
    """
    if strike_type == "fixed" and K is None:
        raise ValueError("K is required for fixed-strike lookback")

    rng    = np.random.default_rng(seed)
    dt     = T / n_steps
    drift  = (r - 0.5 * sigma ** 2) * dt
    vol    = sigma * np.sqrt(dt)

    Z       = rng.standard_normal((n_sims, n_steps))
    log_ret = drift + vol * Z
    log_S   = np.log(S) + np.hstack([np.zeros((n_sims, 1)), np.cumsum(log_ret, axis=1)])
    paths   = np.exp(log_S)  # (n_sims, n_steps+1)

    S_T  = paths[:, -1]
    S_max = paths.max(axis=1)
    S_min = paths.min(axis=1)

    if strike_type == "float":
        if option_type == "call":
            payoffs = S_T - S_min          # always ≥ 0
        else:
            payoffs = S_max - S_T          # always ≥ 0
    else:
        if option_type == "call":
            payoffs = np.maximum(S_max - K, 0.0)
        else:
            payoffs = np.maximum(K - S_min, 0.0)

    disc   = np.exp(-r * T) * payoffs
    price  = float(disc.mean())
    stderr = float(disc.std() / np.sqrt(n_sims))

    return {
        "price":      price,
        "std_error":  stderr,
        "conf_95_lo": price - 1.96 * stderr,
        "conf_95_hi": price + 1.96 * stderr,
    }


# ── unified entry point ───────────────────────────────────────────────────────

def price_lookback(S: float, T: float, r: float, sigma: float,
                   option_type: str = "call",
                   strike_type: str = "float",
                   K: float = None,
                   method: str = "closed",
                   **kwargs) -> dict:
    """
    Unified lookback option pricer.

    method='closed' → Goldman-Sosin-Gatto (float only; falls back to MC for fixed).
    method='mc'     → Monte Carlo.
    """
    if method == "mc" or (strike_type == "fixed" and method != "mc"):
        return mc_lookback(S, T, r, sigma, option_type, strike_type, K, **kwargs)
    return price_lookback_float(S, T, r, sigma, option_type)
