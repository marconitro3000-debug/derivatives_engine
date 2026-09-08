"""
volsurface.pricing
==================

Closed-form and numerical option pricing, and the one function that prices a
*named* option off a fitted surface.

Everything else in the library works in ``(k, T)`` -- log-moneyness against the
fitted forward, and a year fraction. That is the right coordinate for fitting a
surface and the wrong one for using it, so `option.price_option` is the
translation layer, and deliberately the only place the translation happens.

Module map
----------
    blackscholes  closed-form price and Greeks
    impliedvol    inversion, with a guard for the region where implied vol is
                  not identifiable from a float64 price
    american      binomial lattice, and the de-Americanisation it makes possible
    montecarlo    an independent numerical check on the analytic formula
    option        strike + expiry date + side -> price, Greeks, and the caveats
                  needed to judge the number
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
from .option import MAX_YEARS, OptionQuote, price_option, resolve_maturity

__all__ = [
    "MAX_YEARS",
    "OptionQuote",
    "american_implied_vol",
    "binomial_price",
    "carry_from_forward",
    "de_americanised_iv",
    "greeks",
    "implied_vol",
    "mc_price",
    "price",
    "price_option",
    "put_call_parity_check",
    "resolve_maturity",
]
