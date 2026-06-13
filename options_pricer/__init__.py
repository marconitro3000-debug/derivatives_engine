"""
options_pricer — derivatives pricing & volatility calibration engine.
"""

from .core.black_scholes import price, greeks, put_call_parity_check
from .core.implied_vol import implied_vol, iv_surface
from .core.monte_carlo import mc_price
from .core.binomial_tree import binomial_price
from .surface.vol_surface import VolSurface, from_iv_dict

__all__ = [
    "price", "greeks", "put_call_parity_check",
    "implied_vol", "iv_surface",
    "mc_price",
    "binomial_price",
    "VolSurface", "from_iv_dict",
]
