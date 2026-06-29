"""Volatility analysis: realized estimators, GARCH(1,1), variance swaps, VRP."""

from .realized import (
    close_to_close,
    parkinson,
    garman_klass,
    rogers_satchell,
    yang_zhang,
    ewma,
    all_estimators,
    realized_variance,
)
from .garch import (
    GARCHResult,
    fit as garch_fit,
    forecast as garch_forecast,
    simulate as garch_simulate,
)
from .variance_swap import (
    fair_variance_strike_bs,
    fair_variance_strike_mf,
    variance_swap_pv,
    vrp,
    vrp_summary,
    realized_variance_from_prices,
)
