"""
volsurface/data/forward.py
The forward and the discount factor, implied from the market rather than assumed.

Put-call parity gives ``C(K) - P(K) = DF * (F - K)``: a straight line in ``K``
with slope ``-DF`` and intercept ``DF * F``, so a weighted regression over the
matched liquid strikes returns both unknowns at once. Assuming instead that
``F = S * exp((r - q) T)`` with a guessed dividend yield puts a systematic tilt
through the entire fitted skew.

The complication is that parity is a theorem about *European* options and listed
equity options are American, so `fit_forward` runs the parity fit and the
de-Americanisation as a short fixed point -- see its docstring.
"""

from __future__ import annotations

import numpy as np

from volsurface.pricing.american import carry_from_forward, de_americanised_iv
from volsurface.pricing.impliedvol import implied_vol


def implied_forward(strikes: np.ndarray, calls: np.ndarray, puts: np.ndarray,
                    weights: np.ndarray | None = None) -> tuple[float, float]:
    """Fit ``C(K) - P(K) = DF * (F - K)`` by weighted least squares.

    Parameters
    ----------
    strikes, calls, puts : matched arrays -- same strike, same expiry.
    weights              : optional per-strike weights (use inverse spread).

    Returns
    -------
    ``(forward, discount_factor)``

    Raises
    ------
    ValueError
        If there are fewer than three matched strikes, or the fit implies a
        discount factor outside a plausible range -- which means the quotes were
        too stale to trust and the caller should fall back to a rate assumption.
    """
    if len(strikes) < 3:
        raise ValueError("need at least 3 matched call/put strikes for a parity fit")

    strikes = np.asarray(strikes, dtype=float)
    y = np.asarray(calls, dtype=float) - np.asarray(puts, dtype=float)
    wts = np.ones_like(strikes) if weights is None else np.asarray(weights, dtype=float)
    wts = wts / wts.sum()

    # Weighted linear regression y = intercept + slope * K.
    kbar = float(np.sum(wts * strikes))
    ybar = float(np.sum(wts * y))
    cov = float(np.sum(wts * (strikes - kbar) * (y - ybar)))
    var = float(np.sum(wts * (strikes - kbar) ** 2))
    if var <= 0:
        raise ValueError("degenerate strike grid in parity fit")

    slope = cov / var
    intercept = ybar - slope * kbar

    df = -slope
    if not (0.80 < df <= 1.02):
        raise ValueError(f"parity fit implied an implausible discount factor {df:.4f}")
    forward = intercept / df
    if forward <= 0:
        raise ValueError("parity fit implied a non-positive forward")
    return float(forward), float(df)


def fit_forward(spot: float, strikes: np.ndarray,
                call_mid: np.ndarray, put_mid: np.ndarray,
                call_spread: np.ndarray, put_spread: np.ndarray,
                T: float, *, de_americanize: bool = True,
                lattice_steps: int = 150, fallback_rate: float = 0.04,
                n_passes: int = 2) -> tuple[float, float]:
    """Forward and discount factor from parity, corrected for American exercise.

    `implied_forward` fits ``C(K) - P(K) = DF * (F - K)``. That identity is a
    theorem about *European* options. Listed equity and ETF options are
    American, their early-exercise premia differ between the call and the put
    and vary with strike, so ``C - P`` is not a straight line in ``K`` and the
    regression returns a biased slope -- a biased discount factor, and through
    it a biased forward. On an eighteen-month SPY expiry the error is close to a
    full percent of the forward, which is a systematic tilt through the entire
    fitted skew.

    The fix is a short fixed point, because the two unknowns are circular: the
    forward is needed to de-Americanise a quote, and de-Americanised quotes are
    needed to fit the forward.

        1. fit (F, DF) from the raw American mids
        2. strip the early-exercise premium off every matched call and put
        3. refit (F, DF) on the European-equivalent prices
        4. repeat once

    Two passes are enough: the premium depends on the forward only weakly, so
    the second correction is already an order of magnitude smaller than the
    first.
    """
    weights = 1.0 / (call_spread + put_spread + 1e-6)
    F, DF = implied_forward(strikes, call_mid, put_mid, weights=weights)
    if not de_americanize:
        return F, DF

    for _ in range(n_passes):
        r, q = carry_from_forward(spot, F, DF, T)
        euro_c, euro_p = [], []
        for i, K in enumerate(strikes):
            euro_c.append(_european_equivalent(spot, float(K), T, r, q, F, DF,
                                               float(call_mid[i]), "call", lattice_steps))
            euro_p.append(_european_equivalent(spot, float(K), T, r, q, F, DF,
                                               float(put_mid[i]), "put", lattice_steps))
        try:
            F, DF = implied_forward(strikes, np.array(euro_c), np.array(euro_p),
                                    weights=weights)
        except ValueError:
            break                      # keep the last plausible fit
    return F, DF


def _european_equivalent(spot, K, T, r, q, F, DF, mid, option, lattice_steps) -> float:
    """Quote minus its early-exercise premium; the raw mid if that is not computable."""
    try:
        sigma_e = implied_vol(F, K, T, 0.0, mid / DF, option)
    except ValueError:
        return mid                     # unfittable quote: leave it alone
    _, euro = de_americanised_iv(spot, K, T, r, q, F, DF, mid, option,
                                 sigma_e, n_steps=lattice_steps)
    return euro
