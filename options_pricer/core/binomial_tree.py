"""
core/binomial_tree.py
Cox-Ross-Rubinstein (CRR) binomial tree.

Supports European and American options (call and put).
American options allow early exercise at every node via backward induction.
"""

import numpy as np


# ── CRR parameters ────────────────────────────────────────────────────────────

def _crr_params(T: float, r: float, sigma: float,
                n_steps: int) -> tuple[float, float, float, float]:
    """
    Compute CRR up/down factors and risk-neutral probability.

    Returns (dt, u, d, p) where:
      dt : time step size
      u  : up-factor   = exp(sigma * sqrt(dt))
      d  : down-factor = 1/u
      p  : risk-neutral probability of up move
    """
    dt = T / n_steps
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp(r * dt) - d) / (u - d)

    if not (0 < p < 1):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} is outside (0,1). "
            "Increase n_steps or check inputs."
        )
    return dt, u, d, p


# ── terminal payoffs ──────────────────────────────────────────────────────────

def _terminal_payoffs(S: float, K: float, u: float, d: float,
                      n_steps: int, option: str) -> np.ndarray:
    """Compute intrinsic payoffs at all terminal nodes."""
    j      = np.arange(n_steps + 1)           # 0 = all-down, n_steps = all-up
    S_T    = S * (u ** j) * (d ** (n_steps - j))
    if option == "call":
        return np.maximum(S_T - K, 0)
    else:
        return np.maximum(K - S_T, 0)


# ── backward induction ────────────────────────────────────────────────────────

def _backward(S: float, K: float, T: float, r: float,
              u: float, d: float, p: float, dt: float,
              n_steps: int, option: str, american: bool) -> float:
    """
    Backward induction through the tree.

    At each node, option value = max(continuation, intrinsic) for American,
    or just continuation for European.
    """
    disc    = np.exp(-r * dt)
    q       = 1.0 - p
    values  = _terminal_payoffs(S, K, u, d, n_steps, option)

    for step in range(n_steps - 1, -1, -1):
        # continuation value (one step earlier)
        values = disc * (p * values[1:] + q * values[:-1])

        if american:
            # intrinsic at each node at this step
            j        = np.arange(step + 1)
            S_nodes  = S * (u ** j) * (d ** (step - j))
            if option == "call":
                intrinsic = np.maximum(S_nodes - K, 0)
            else:
                intrinsic = np.maximum(K - S_nodes, 0)
            values = np.maximum(values, intrinsic)

    return float(values[0])


# ── public API ────────────────────────────────────────────────────────────────

def binomial_price(S: float, K: float, T: float, r: float, sigma: float,
                   option: str = "call",
                   style: str = "european",
                   n_steps: int = 500) -> dict:
    """
    Price an option using the CRR binomial tree.

    Parameters
    ----------
    S       : underlying price
    K       : strike
    T       : time to expiry (years)
    r       : risk-free rate
    sigma   : volatility
    option  : 'call' or 'put'
    style   : 'european' or 'american'
    n_steps : number of time steps (more = more accurate, slower)

    Returns
    -------
    dict with keys:
      price          : option price
      early_exercise : estimated early-exercise premium (American - European)
      n_steps        : steps used
      style          : 'american' or 'european'
    """
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'.")
    if style not in ("european", "american"):
        raise ValueError("style must be 'european' or 'american'.")

    dt, u, d, p = _crr_params(T, r, sigma, n_steps)
    american    = style == "american"
    price_val   = _backward(S, K, T, r, u, d, p, dt, n_steps, option, american)

    # early-exercise premium (only meaningful for American)
    if american:
        euro_val = _backward(S, K, T, r, u, d, p, dt, n_steps, option, False)
        premium  = price_val - euro_val
    else:
        premium  = 0.0

    return {
        "price":          price_val,
        "early_exercise": float(premium),
        "n_steps":        n_steps,
        "style":          style,
    }
