"""
exotics/digital.py
Digital (binary) option pricing.

Cash-or-nothing    → pays fixed cash amount Q at expiry if in-the-money
Asset-or-nothing   → pays the asset (S_T) at expiry if in-the-money
One-touch          → pays Q if barrier H is ever reached during [0,T]
No-touch           → pays Q at expiry if barrier H is never reached during [0,T]

All cash-or-nothing and asset-or-nothing formulas are analytical (BSM).
One-touch and no-touch formulas use the Reiner-Rubinstein barrier framework.
"""

import numpy as np
from scipy.stats import norm


# ── helpers ───────────────────────────────────────────────────────────────────

def _d1d2(S, K, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return d1, d1 - sigma * sqrt_T


# ── cash-or-nothing ───────────────────────────────────────────────────────────

def price_cash_or_nothing(S: float, K: float, T: float, r: float, sigma: float,
                          option_type: str = "call",
                          cash: float = 1.0) -> dict:
    """
    European cash-or-nothing (binary) option.

    Pays `cash` at expiry T if:
      call → S_T > K
      put  → S_T < K

    Formula (BSM):
      call : cash · e^{-rT} · N(d2)
      put  : cash · e^{-rT} · N(-d2)

    This is the Q-probability of finishing ITM, discounted:
    the d2 in BSM is exactly P_Q(S_T > K).
    """
    if T <= 0:
        raise ValueError("T must be positive")
    _, d2 = _d1d2(S, K, T, r, sigma)
    df    = np.exp(-r * T)

    if option_type == "call":
        px = cash * df * norm.cdf(d2)
        prob = float(norm.cdf(d2))
    else:
        px = cash * df * norm.cdf(-d2)
        prob = float(norm.cdf(-d2))

    return {"price": float(px), "prob_itm": prob, "d2": float(d2)}


# ── asset-or-nothing ──────────────────────────────────────────────────────────

def price_asset_or_nothing(S: float, K: float, T: float, r: float, sigma: float,
                           option_type: str = "call") -> dict:
    """
    European asset-or-nothing option.

    Pays S_T at expiry if:
      call → S_T > K
      put  → S_T < K

    Formula (BSM):
      call : S · N(d1)
      put  : S · N(-d1)

    Note: vanilla BSM = asset-or-nothing − cash-or-nothing·K·e^{-rT}
    """
    if T <= 0:
        raise ValueError("T must be positive")
    d1, _ = _d1d2(S, K, T, r, sigma)

    if option_type == "call":
        px   = S * norm.cdf(d1)
        prob = float(norm.cdf(d1))   # risk-neutral prob (not real-world)
    else:
        px   = S * norm.cdf(-d1)
        prob = float(norm.cdf(-d1))

    return {"price": float(px), "prob": prob, "d1": float(d1)}


# ── one-touch ─────────────────────────────────────────────────────────────────

def price_one_touch(S: float, T: float, r: float, sigma: float,
                    H: float,
                    touch_type: str = "down",
                    payout: float = 1.0,
                    payout_at: str = "touch") -> dict:
    """
    One-touch option: pays `payout` if barrier H is ever touched.

    Parameters
    ----------
    touch_type  : 'down' (H < S) or 'up' (H > S)
    payout      : cash amount paid on touch (default 1.0)
    payout_at   : 'touch' (at the moment of touching) or 'expiry'

    Formula
    -------
    For payout at touch time — uses the Reiner-Rubinstein F-term:
        OT = payout · [(H/S)^{μ+λ} · N(η·z) + (H/S)^{μ-λ} · N(η·(z − 2λσ√T))]

    where
        μ = (r − σ²/2) / σ²
        λ = √(μ² + 2r/σ²)
        z = ln(H/S) / (σ√T) + λσ√T
        η = +1 for down, −1 for up
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if touch_type not in ("down", "up"):
        raise ValueError("touch_type must be 'down' or 'up'")
    if (touch_type == "down" and H >= S) or (touch_type == "up" and H <= S):
        raise ValueError("Barrier H must be below S for 'down' and above S for 'up'")

    sqrt_T = np.sqrt(T)
    mu  = (r - 0.5 * sigma ** 2) / sigma ** 2
    lam = np.sqrt(mu ** 2 + 2.0 * r / sigma ** 2)
    eta = 1.0 if touch_type == "down" else -1.0

    z = np.log(H / S) / (sigma * sqrt_T) + lam * sigma * sqrt_T

    # F term (payout at touch time)
    HS = H / S
    ot = payout * (
        HS ** (mu + lam) * norm.cdf(eta * z)
        + HS ** (mu - lam) * norm.cdf(eta * (z - 2 * lam * sigma * sqrt_T))
    )

    if payout_at == "expiry":
        # The Reiner-Rubinstein E-term = PV of payout at T if barrier NOT touched (no-touch).
        # So: one-touch at T = payout * e^{-rT} - E_term
        x2 = np.log(S / H) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
        y2 = np.log(H / S) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
        e_no_touch = payout * np.exp(-r * T) * (
            norm.cdf(eta * (x2 - sigma * sqrt_T))
            - HS ** (2 * mu) * norm.cdf(eta * (y2 - sigma * sqrt_T))
        )
        px = payout * np.exp(-r * T) - e_no_touch
    else:
        px = ot

    # Implied probability of touch (discounted present value / payout)
    prob_touch = float(ot / payout)

    return {"price": float(px), "prob_touch": prob_touch}


# ── no-touch ──────────────────────────────────────────────────────────────────

def price_no_touch(S: float, T: float, r: float, sigma: float,
                   H: float,
                   touch_type: str = "down",
                   payout: float = 1.0) -> dict:
    """
    No-touch option: pays `payout` at expiry T if barrier H is NEVER reached.

    No-touch (pays at T) + one-touch (pays at T) = e^{-rT} · payout
    (since exactly one of the two events happens)

    This function prices the no-touch paying at expiry.
    """
    ot_expiry = price_one_touch(S, T, r, sigma, H, touch_type, payout, payout_at="expiry")
    nt_price  = payout * np.exp(-r * T) - ot_expiry["price"]
    return {"price": float(max(nt_price, 0.0)),
            "prob_no_touch": 1.0 - ot_expiry["prob_touch"]}


# ── Monte Carlo (for all digital types) ───────────────────────────────────────

def mc_digital(S: float, K: float, T: float, r: float, sigma: float,
               digital_type: str = "cash-or-nothing",
               option_type: str = "call",
               H: float = None,
               cash: float = 1.0,
               n_sims: int = 100_000,
               n_steps: int = 252,
               seed: int = None) -> dict:
    """
    Monte Carlo pricing for digital options.

    digital_type : 'cash-or-nothing', 'asset-or-nothing', 'one-touch', 'no-touch'
    H            : barrier level (required for 'one-touch' and 'no-touch')
    """
    rng    = np.random.default_rng(seed)
    dt     = T / n_steps
    drift  = (r - 0.5 * sigma ** 2) * dt
    vol    = sigma * np.sqrt(dt)

    Z       = rng.standard_normal((n_sims, n_steps))
    log_ret = drift + vol * Z
    log_S   = np.log(S) + np.hstack([np.zeros((n_sims, 1)), np.cumsum(log_ret, axis=1)])
    paths   = np.exp(log_S)
    S_T     = paths[:, -1]

    if digital_type == "cash-or-nothing":
        itm = (S_T > K) if option_type == "call" else (S_T < K)
        payoffs = cash * itm.astype(float)
    elif digital_type == "asset-or-nothing":
        itm = (S_T > K) if option_type == "call" else (S_T < K)
        payoffs = S_T * itm.astype(float)
    elif digital_type in ("one-touch", "no-touch"):
        if H is None:
            raise ValueError("H is required for one-touch / no-touch")
        touched = (np.any(paths <= H, axis=1) if H < S
                   else np.any(paths >= H, axis=1))
        payoffs = (cash * touched.astype(float) if digital_type == "one-touch"
                   else cash * (~touched).astype(float))
    else:
        raise ValueError(f"Unknown digital_type: {digital_type}")

    disc   = np.exp(-r * T) * payoffs
    price  = float(disc.mean())
    stderr = float(disc.std() / np.sqrt(n_sims))
    return {
        "price":      price,
        "std_error":  stderr,
        "conf_95_lo": price - 1.96 * stderr,
        "conf_95_hi": price + 1.96 * stderr,
    }
