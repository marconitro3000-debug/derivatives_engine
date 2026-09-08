"""
volsurface.pricing
==================

Closed-form and numerical option pricing: the primitives.

These are primitives: they take numbers and return numbers, and depend on
nothing else in the library. To price a *named* option off a fitted surface --
strike, expiry date, side -- use `volsurface.price_option`, which lives one
level up because it consumes every other subpackage.

Module map
----------
    blackscholes  closed-form price and Greeks
    impliedvol    inversion, with a guard for the region where implied vol is
                  not identifiable from a float64 price
    american      binomial lattice, and the de-Americanisation it makes possible
    montecarlo    an independent numerical check on the analytic formula
"""

from .american import (
    american_implied_vol,
    binomial_price,
    carry_from_forward,
    de_americanised_iv,
)
from .blackscholes import greeks, price, put_call_parity_check
from .impliedvol import implied_vol
from .montecarlo import mc_price

__all__ = [
    "american_implied_vol",
    "binomial_price",
    "carry_from_forward",
    "de_americanised_iv",
    "greeks",
    "implied_vol",
    "mc_price",
    "price",
    "put_call_parity_check",
]
