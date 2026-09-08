"""
volsurface/american.py
Cox-Ross-Rubinstein binomial tree, and the early-exercise bias it measures.

Why a lattice belongs in a project about *European* implied vol: listed
single-stock and ETF options -- SPY included -- are **American**. Their quoted
prices contain an early-exercise premium, and inverting a European
Black-Scholes formula on them attributes that premium to volatility. The
implied vol that comes out is biased upward, systematically, and not uniformly:
the premium lives in in-the-money puts and grows with maturity and with rates.

`early_exercise_bias` quantifies that bias on an actual chain, in basis points
of implied vol, so the pipeline's choice to fit out-of-the-money quotes only can
be justified with a number instead of an assertion. On short-dated OTM options
it is negligible, which is exactly what the chain filter relies on.

The tree is also the independent check on the analytic Black-Scholes formula
that the whole project rests on: `tests/test_numerical_pricers.py` pins its
convergence against the closed form.
"""

from __future__ import annotations

import numpy as np


# ── CRR parameters ────────────────────────────────────────────────────────────

def _crr_params(T: float, r: float, sigma: float,
                n_steps: int, q: float = 0.0) -> tuple[float, float, float, float]:
    """
    Compute CRR up/down factors and risk-neutral probability.

    Returns (dt, u, d, p) where:
      dt : time step size
      u  : up-factor   = exp(sigma * sqrt(dt))
      d  : down-factor = 1/u
      p  : risk-neutral probability of up move

    The dividend yield enters the *drift*, ``p = (e^{(r-q)dt} - d)/(u - d)``,
    while discounting stays at ``r``. It must not be folded into the spot as
    ``S*e^{-qT}``: that shortcut reproduces the European price exactly and gets
    the American one wrong, because it moves the whole lattice rather than its
    drift and so shifts the early-exercise boundary.
    """
    dt = T / n_steps
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp((r - q) * dt) - d) / (u - d)

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
    j      = np.arange(n_steps + 1)
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
        values = disc * (p * values[1:] + q * values[:-1])

        if american:
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
                   n_steps: int = 500,
                   q: float = 0.0) -> dict:
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
    q       : continuous dividend yield

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

    dt, u, d, p = _crr_params(T, r, sigma, n_steps, q)
    american    = style == "american"
    price_val   = _backward(S, K, T, r, u, d, p, dt, n_steps, option, american)

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


# ── de-Americanization ───────────────────────────────────────────────────────

def de_americanised_iv(spot: float, strike: float, T: float, r: float, q: float,
                       forward: float, discount: float, market_price: float,
                       option: str, sigma_european: float,
                       n_steps: int = 150, max_iter: int = 4,
                       tol: float = 1e-5) -> tuple[float, float]:
    """The volatility an American quote implies, with the early-exercise premium removed.

    The quote is an American option; inverting a European formula on it has
    nowhere to put the early-exercise premium and charges it to volatility
    instead. The premium is priced on a lattice and subtracted from the quote
    before the European inversion runs:

        sigma <- IV_BS( market_price - [CRR_american(sigma) - CRR_european(sigma)] )

    iterated to a fixed point. Three details make this the right construction:

    * **The premium is a difference of two prices off the same lattice at the
      same sigma**, so the tree's discretisation error cancels almost exactly.
      What survives is the premium, the only quantity being asked for -- which
      is why 150 steps is enough here when pricing to the same accuracy would
      need far more.
    * **The final inversion is the exact Black-Scholes one**, not a linearised
      vega step. A single Newton step from the European solution is fine for the
      few-basis-point premia of short-dated calls and badly wrong for a
      long-dated put, where the premium can exceed a whole volatility point.
    * **It converges in two or three iterations**, because the premium is a slow
      function of sigma even where it is large.

    The result drops straight into a European surface fit: the surface stays
    European and arbitrage-free while the data feeding it stops being biased.

    Parameters
    ----------
    forward, discount : the chain's own fitted F and DF for this expiry, so the
        lattice and the surface cannot disagree about the forward.
    sigma_european : the European inversion, used as the starting point.

    Returns
    -------
    ``(sigma, european_equivalent_price)``. The second value is the quote minus
    its early-exercise premium: the price a European option with the same strike
    and expiry would trade at. `volsurface.data` needs it because put-call
    parity -- which it uses to fit the forward -- holds for European options and
    *not* for American ones, so the parity regression has to run on these prices
    rather than on the raw mids.

    Falls back to ``(sigma_european, market_price)`` if the iteration cannot
    produce a valid inversion: a quote that resists this is a quote to drop, not
    to guess at.
    """
    from .impliedvol import implied_vol

    sigma = sigma_european
    for _ in range(max_iter):
        euro = binomial_price(spot, strike, T, r, sigma, option,
                              style="european", n_steps=n_steps, q=q)["price"]
        amer = binomial_price(spot, strike, T, r, sigma, option,
                              style="american", n_steps=n_steps, q=q)["price"]
        premium = max(amer - euro, 0.0)
        if premium <= 0.0:
            return sigma_european, market_price

        european_price = market_price - premium
        try:
            sigma_new = implied_vol(forward, strike, T, 0.0,
                                    european_price / discount, option)
        except ValueError:
            return sigma_european, market_price

        converged = abs(sigma_new - sigma) < tol
        sigma = sigma_new
        if converged:
            break

    return sigma, european_price


def carry_from_forward(spot: float, forward: float, discount: float,
                       T: float) -> tuple[float, float]:
    """Recover ``(r, q)`` from the chain's own fitted forward and discount factor.

    ``r`` comes from ``DF = e^{-rT}`` and ``q`` from ``F = S e^{(r-q)T}``. Using
    the numbers the chain implied, rather than an external rate curve and a
    dividend estimate, keeps the lattice consistent with the surface it is
    correcting -- the two cannot disagree about the forward.
    """
    r = -np.log(discount) / T
    q = r - np.log(forward / spot) / T
    return float(r), float(q)


def american_implied_vol(S: float, K: float, T: float, r: float, q: float,
                         market_price: float, option: str = "call",
                         n_steps: int = 120,
                         sigma_lo: float = 0.01, sigma_hi: float = 3.0) -> float:
    """Invert an American quote on the lattice instead of on Black-Scholes.

    This is de-Americanization, and it is what the whole diagnostic above argues
    for. The quote is an American option; solving

        CRR_american(sigma) = market_price

    returns the volatility the market is actually quoting, with the
    early-exercise premium accounted for by the lattice rather than absorbed
    into sigma. Repricing a *European* option at that sigma and inverting it
    with Black-Scholes gives back the same number, so the result drops straight
    into a European surface fit -- which is the point: the surface stays
    European and arbitrage-free while the data feeding it stops being biased.

    Raises
    ------
    ValueError
        If the price lies outside the lattice's reachable range, which means the
        quote is stale or violates a no-arbitrage bound.
    """
    from scipy.optimize import brentq

    def gap(sigma: float) -> float:
        return binomial_price(S, K, T, r, sigma, option, style="american",
                              n_steps=n_steps, q=q)["price"] - market_price

    lo, hi = gap(sigma_lo), gap(sigma_hi)
    if lo > 0:
        raise ValueError(
            f"price {market_price:.4f} is below the lattice value at "
            f"{sigma_lo:.0%} vol -- stale or below intrinsic"
        )
    if hi < 0:
        raise ValueError(
            f"price {market_price:.4f} exceeds the lattice value at "
            f"{sigma_hi:.0%} vol"
        )
    return float(brentq(gap, sigma_lo, sigma_hi, xtol=1e-6, maxiter=100))
