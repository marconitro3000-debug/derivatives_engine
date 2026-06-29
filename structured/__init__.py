"""Structured products: CDO (Gaussian copula), autocallables, MBS/PSA."""

from .cdo import (
    expected_tranche_loss,
    tranche_fair_spread,
    cdo_structure,
    loss_distribution,
)
from .autocall import (
    AutocallResult,
    price_autocall,
    autocall_greeks,
)
from .mbs import (
    psa_smm,
    psa_schedule,
    mbs_cashflows,
    weighted_average_life,
    mbs_price,
    mbs_yield,
    oas,
    mbs_summary,
)
