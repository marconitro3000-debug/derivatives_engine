"""
volsurface.data
===============

Where the quotes come from and what is done to them before any model sees them.

The quality of an implied-vol surface is decided in this package, not in the
network. Five things matter and each is done here explicitly rather than
delegated to the data vendor:

1. **Liquidity filtering** -- two-sided quotes only, positive size, and a cap on
   the relative bid-ask spread.
2. **Forward and discount factor implied from the market**, from put-call parity
   across matched strikes, instead of assuming ``F = S * exp((r - q) T)`` with a
   guessed dividend yield. This is what removes the systematic skew tilt a wrong
   dividend assumption produces.
3. **OTM-only selection** -- calls above the forward, puts below it.
4. **In-house IV inversion**, from ``mid / DF`` in the forward measure, so the
   vols are consistent with the forward fitted in step 2.
5. **De-Americanisation** -- listed equity and ETF options are American, and a
   European inversion charges the early-exercise premium to volatility. On SPY
   that is 13bp of vol on average and 25bp beyond a year, almost entirely in the
   puts.

Module map
----------
    snapshot     `ChainSnapshot` -- the cleaned chain, and the only structure
                 that leaves this package
    conventions  day-count conversions (ACT/365F is what everything uses)
    forward      the forward and discount factor, from parity, as a fixed point
                 against the de-Americanised prices
    clean        raw quote frames -> `ChainSnapshot`; every filter lives here
    fetch        the live Yahoo Finance chain
    synthetic    the offline generator: a chain from a known arbitrage-free
                 SSVI surface, which is what the tests fit
"""

from .clean import build_snapshot
from .conventions import year_fraction
from .fetch import fetch_chain
from .forward import fit_forward, implied_forward
from .snapshot import ChainSnapshot
from .synthetic import synthetic_snapshot

__all__ = [
    "ChainSnapshot",
    "build_snapshot",
    "fetch_chain",
    "fit_forward",
    "implied_forward",
    "synthetic_snapshot",
    "year_fraction",
]
