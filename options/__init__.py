"""
options/
Black-Scholes pricing primitives and implied-volatility inversion.

These are the *ground truth* of the project: the neural surface is trained on
implied vols that this package extracts from market quotes, and it is scored by
re-pricing the chain through `black_scholes.price`.
"""

from .black_scholes import price, greeks, put_call_parity_check
from .implied_vol import implied_vol, iv_surface
from .binomial_tree import binomial_price
from .monte_carlo import mc_price

__all__ = [
    "price", "greeks", "put_call_parity_check",
    "implied_vol", "iv_surface",
    "binomial_price", "mc_price",
]
