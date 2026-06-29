"""
exotics/
Exotic option pricing beyond vanilla calls and puts.

Modules
-------
barrier  : down-out/in, up-out/in calls and puts (Reiner-Rubinstein + MC)
asian    : geometric closed-form, arithmetic MC, Kemna-Vorst approximation
lookback : floating-strike (Goldman-Sosin-Gatto) and fixed-strike (MC)
digital  : cash-or-nothing, asset-or-nothing, one-touch, no-touch
charts   : payoff diagrams, path visualisation, exotic comparison
"""

from .barrier  import price_barrier, mc_barrier, greeks_barrier
from .asian    import price_asian, price_asian_geo, price_asian_kv, mc_asian_arith
from .lookback import price_lookback, price_lookback_float, mc_lookback
from .digital  import (price_cash_or_nothing, price_asset_or_nothing,
                        price_one_touch, price_no_touch, mc_digital)
