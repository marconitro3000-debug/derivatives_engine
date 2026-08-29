"""
credit/
Credit derivatives pricing: CDS, risky bonds, CVA/DVA, credit curves.

Modules
-------
hazard_rate : HazardCurve — piecewise-constant default intensity, bootstrap
cds         : CDS pricing — protection/premium legs, par spread, CS01, IR01
bond        : Risky bond — Duffie-Singleton, Z-spread, asset swap spread
cva         : CVA/DVA/BCVA — expected exposure, option CVA
charts      : Academic figures — survival curve, CDS legs, CVA profile
"""

from .hazard_rate import HazardCurve
from .cds  import (protection_leg, risky_pv01, par_spread,
                   cds_value, cs01, ir01)
from .bond import (risky_bond_price, z_spread, asset_swap_spread, bond_summary)
from .cva  import (cva, cva_option, dva, bilateral_cva)
