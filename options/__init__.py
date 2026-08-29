"""
options/
Core pricing primitives: Black-Scholes and implied vol extraction.

Extended modules (Monte Carlo, binomial tree, vol surface interpolation)
are in lib/options/.
"""

from .black_scholes import price, greeks, put_call_parity_check
from .implied_vol import implied_vol, iv_surface

__all__ = [
    "price", "greeks", "put_call_parity_check",
    "implied_vol", "iv_surface",
]
