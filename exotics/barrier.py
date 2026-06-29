"""
exotics/barrier.py
Barrier option pricing: closed-form (Reiner-Rubinstein 1991) + Monte Carlo.

Eight standard types
--------------------
down-out call   down-in call    up-out call   up-in call
down-out put    down-in put     up-out put    up-in put

Continuous monitoring → closed-form formula.
Discrete monitoring   → Monte Carlo.

Both respect parity:   price_in + price_out = BS vanilla price.
"""

import numpy as np
from scipy.stats import norm


# ── helpers ───────────────────────────────────────────────────────────────────

def _d1d2(S, K, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return d1, d1 - sigma * sqrt_T


def _bs_vanilla(S, K, T, r, sigma, phi):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return phi * S * norm.cdf(phi * d1) - phi * K * np.exp(-r * T) * norm.cdf(phi * d2)


def _rr_blocks(S, K, T, r, sigma, H, phi, eta, rebate):
    """
    Reiner-Rubinstein (1991) building blocks A B C D E F.
    phi : +1 = call, −1 = put
    eta : +1 = down barrier, −1 = up barrier
    """
    sqrt_T = np.sqrt(T)
    mu  = (r - 0.5 * sigma ** 2) / sigma ** 2
    lam = np.sqrt(mu ** 2 + 2.0 * r / sigma ** 2)

    x1 = np.log(S / K)          / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
    x2 = np.log(S / H)          / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
    y1 = np.log(H ** 2 / (S*K)) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
    y2 = np.log(H / S)          / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
    z  = np.log(H / S)          / (sigma * sqrt_T) + lam * sigma * sqrt_T

    def term(s_factor, k_factor, n_arg):
        return phi * s_factor * norm.cdf(phi * n_arg) \
             - phi * K * np.exp(-r * T) * k_factor * norm.cdf(phi * (n_arg - sigma * sqrt_T))

    HS_2mu1 = (H / S) ** (2 * (mu + 1))
    HS_2mu  = (H / S) ** (2 * mu)

    A = phi * S * norm.cdf(phi * x1) \
      - phi * K * np.exp(-r * T) * norm.cdf(phi * (x1 - sigma * sqrt_T))
    B = phi * S * norm.cdf(phi * x2) \
      - phi * K * np.exp(-r * T) * norm.cdf(phi * (x2 - sigma * sqrt_T))
    C = phi * S * HS_2mu1 * norm.cdf(eta * y1) \
      - phi * K * np.exp(-r * T) * HS_2mu * norm.cdf(eta * (y1 - sigma * sqrt_T))
    D = phi * S * HS_2mu1 * norm.cdf(eta * y2) \
      - phi * K * np.exp(-r * T) * HS_2mu * norm.cdf(eta * (y2 - sigma * sqrt_T))
    # E: rebate paid at expiry if knocked out
    E = rebate * np.exp(-r * T) * (
        norm.cdf(eta * (x2 - sigma * sqrt_T))
        - HS_2mu * norm.cdf(eta * (y2 - sigma * sqrt_T))
    )
    # F: rebate paid at touch time (undiscounted in the formula — already PV)
    F = rebate * (
        (H / S) ** (mu + lam) * norm.cdf(eta * z)
        + (H / S) ** (mu - lam) * norm.cdf(eta * (z - 2 * lam * sigma * sqrt_T))
    )
    return A, B, C, D, E, F


# ── closed-form pricing ───────────────────────────────────────────────────────

def price_barrier(S: float, K: float, T: float, r: float, sigma: float,
                  H: float,
                  option_type: str = "call",
                  barrier_type: str = "down-out",
                  rebate: float = 0.0) -> dict:
    """
    Closed-form barrier option price (continuous monitoring).
    Uses Reiner-Rubinstein (1991). Knock-in uses parity with knock-out.

    Parameters
    ----------
    S           : spot price
    K           : strike price
    T           : time to expiry (years)
    r           : continuous risk-free rate
    sigma       : annualised volatility
    H           : barrier level
    option_type : 'call' or 'put'
    barrier_type: 'down-out', 'down-in', 'up-out', 'up-in'
    rebate      : amount paid at barrier event (default 0)

    Returns
    -------
    dict with 'price', 'vanilla', 'discount' (early_exercise_premium or barrier discount)
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if barrier_type not in ("down-out", "down-in", "up-out", "up-in"):
        raise ValueError("barrier_type must be 'down-out', 'down-in', 'up-out', 'up-in'")
    if T <= 0:
        raise ValueError("T must be positive")

    phi  = 1.0 if option_type == "call" else -1.0
    eta  = 1.0 if "down" in barrier_type else -1.0
    down = "down" in barrier_type
    out  = "out"  in barrier_type
    call = option_type == "call"

    vanilla = _bs_vanilla(S, K, T, r, sigma, phi)

    # ── immediate boundary conditions ─────────────────────────────────────────
    if down and S <= H:
        px = rebate if out else vanilla
        return {"price": max(px, 0.0), "vanilla": vanilla, "discount": vanilla - max(px, 0.0)}
    if (not down) and S >= H:
        px = rebate if out else vanilla
        return {"price": max(px, 0.0), "vanilla": vanilla, "discount": vanilla - max(px, 0.0)}

    # ── closed-form knock-out ─────────────────────────────────────────────────
    A, B, C, D, E, F = _rr_blocks(S, K, T, r, sigma, H, phi, eta, rebate)

    if out:
        if down and call:
            px = (A - C + E) if K >= H else (B - D + E)
        elif down and not call:  # down-out put
            px = (A - B + C - D + E) if K > H else E   # K ≤ H → payoff impossible → 0+rebate
        elif not down and call:  # up-out call
            px = (A - B + C - D + E) if K < H else E   # K ≥ H → 0+rebate
        else:                   # up-out put
            px = (A - C + E) if K <= H else (B - D + E)
    else:
        # knock-in = vanilla − knock-out  (rebate=0 for the out leg; F handles the in-rebate)
        out_px_no_rebate = price_barrier(
            S, K, T, r, sigma, H, option_type,
            barrier_type.replace("in", "out"), rebate=0.0
        )["price"]
        px = vanilla - out_px_no_rebate + F  # F = PV of rebate paid at knock-in

    return {"price": max(float(px), 0.0), "vanilla": vanilla,
            "discount": vanilla - max(float(px), 0.0)}


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def mc_barrier(S: float, K: float, T: float, r: float, sigma: float,
               H: float,
               option_type: str = "call",
               barrier_type: str = "down-out",
               rebate: float = 0.0,
               n_sims: int = 100_000,
               n_steps: int = 252,
               seed: int = None) -> dict:
    """
    Monte Carlo barrier option pricing with discrete barrier monitoring.

    Useful for daily-checked barriers (which differ from continuous
    closed-form due to discretisation gap).

    Parameters are the same as price_barrier(). Returns the same dict
    plus 'std_error'.
    """
    rng    = np.random.default_rng(seed)
    dt     = T / n_steps
    drift  = (r - 0.5 * sigma ** 2) * dt
    vol    = sigma * np.sqrt(dt)

    # log-returns: shape (n_sims, n_steps)
    Z       = rng.standard_normal((n_sims, n_steps))
    log_ret = drift + vol * Z
    # paths: shape (n_sims, n_steps+1)
    log_S   = np.log(S) + np.hstack([np.zeros((n_sims, 1)), np.cumsum(log_ret, axis=1)])
    paths   = np.exp(log_S)

    S_T = paths[:, -1]

    # barrier check across all time steps
    if "down" in barrier_type:
        touched = np.any(paths <= H, axis=1)
    else:
        touched = np.any(paths >= H, axis=1)

    # payoff
    if option_type == "call":
        intrinsic = np.maximum(S_T - K, 0.0)
    else:
        intrinsic = np.maximum(K - S_T, 0.0)

    if "out" in barrier_type:
        payoffs = np.where(touched, rebate, intrinsic)
    else:
        payoffs = np.where(touched, intrinsic, rebate)

    disc    = np.exp(-r * T) * payoffs
    price   = float(disc.mean())
    stderr  = float(disc.std() / np.sqrt(n_sims))
    return {
        "price":      price,
        "std_error":  stderr,
        "conf_95_lo": price - 1.96 * stderr,
        "conf_95_hi": price + 1.96 * stderr,
        "vanilla":    _bs_vanilla(S, K, T, r, sigma, 1.0 if option_type == "call" else -1.0),
    }


# ── Greeks (numerical) ────────────────────────────────────────────────────────

def greeks_barrier(S: float, K: float, T: float, r: float, sigma: float,
                   H: float, option_type: str = "call",
                   barrier_type: str = "down-out",
                   rebate: float = 0.0) -> dict:
    """First-order Greeks via central finite differences."""
    eps_S     = S     * 1e-4
    eps_sigma = sigma * 1e-4
    eps_T     = 1.0 / 365

    def px(s=S, t=T, v=sigma):
        return price_barrier(s, K, t, r, v, H, option_type, barrier_type, rebate)["price"]

    delta = (px(S + eps_S) - px(S - eps_S)) / (2 * eps_S)
    gamma = (px(S + eps_S) - 2 * px() + px(S - eps_S)) / eps_S ** 2
    theta = (px(t=T - eps_T) - px()) / eps_T          # per year → /365 for per day
    vega  = (px(v=sigma + eps_sigma) - px(v=sigma - eps_sigma)) / (2 * eps_sigma)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta_year": theta,
        "theta_day":  theta / 365,
        "vega_1pct":  vega / 100,
    }
