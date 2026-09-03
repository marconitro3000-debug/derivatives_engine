"""
volsurface
==========

An arbitrage-penalised neural implied-volatility surface, fitted to real option
chains and benchmarked against the parametric surfaces a desk actually uses.

The whole library is one idea. A volatility surface is defined by its total
implied variance ``w(k, T) = sigma(k, T)^2 * T`` over log-moneyness and
maturity, every model implements the same `surface.VolSurface` interface, and
every model is therefore judged by the same two questions:

* how closely does it reproduce the quotes  (`diagnostics.fit_report`)
* does the surface it defines *between* them admit static arbitrage
  (`diagnostics.scan_arbitrage`)

Reporting only the first is how a model that prices negative probabilities wins
a comparison.

Typical use
-----------
    from volsurface import fetch_chain, train_surface, compare, comparison_table

    chain  = fetch_chain("SPY")
    result = train_surface(chain)
    print(comparison_table(compare(chain, result)))

Module map
----------
    conventions   day-count conversions
    blackscholes  closed-form price and Greeks
    impliedvol    inversion, with an identifiability guard
    chain         option chain -> clean (k, T, IV) cloud
    surface       the VolSurface interface
    diagnostics   arbitrage scans and fit reports
    svi           raw SVI and joint SSVI baselines
    neural        the network: prior, model, penalties, training
    report        scorecards and figures
    registry      archive of trained runs, for comparing which model to use
    american      binomial tree, and the de-Americanisation it makes possible
    montecarlo    independent numerical check on the analytic formula
"""

from .american import (
    american_implied_vol,
    binomial_price,
    carry_from_forward,
    de_americanised_iv,
)
from .blackscholes import greeks, price, put_call_parity_check
from .chain import ChainSnapshot, build_snapshot, fetch_chain, synthetic_snapshot
from .diagnostics import ArbitrageReport, FitReport, fit_report, scan_arbitrage
from .impliedvol import implied_vol
from .montecarlo import mc_price
from .neural import (
    ModelConfig,
    NeuralVolSurface,
    PenaltyWeights,
    TrainConfig,
    TrainResult,
    train_surface,
)
from .report import (
    compare,
    comparison_table,
    evaluate_surface,
    plot_arbitrage_map,
    plot_fit,
    plot_model_comparison,
    plot_quote,
    plot_training,
)
from .registry import RunRecord, list_runs, load_history, load_run, runs_table, save_run
from .surface import VolSurface
from .svi import SSVIParams, SSVISurface, SVIParams, SVISliceSurface

__version__ = "1.0.0"

__all__ = [
    # market data
    "ChainSnapshot", "fetch_chain", "build_snapshot", "synthetic_snapshot",
    # pricing primitives
    "price", "greeks", "put_call_parity_check", "implied_vol",
    "binomial_price", "de_americanised_iv", "american_implied_vol",
    "carry_from_forward", "mc_price",
    # surfaces
    "VolSurface", "SVIParams", "SVISliceSurface", "SSVIParams", "SSVISurface",
    "NeuralVolSurface", "ModelConfig",
    # fitting
    "TrainConfig", "TrainResult", "PenaltyWeights", "train_surface",
    # scoring
    "FitReport", "ArbitrageReport", "fit_report", "scan_arbitrage",
    "evaluate_surface", "compare", "comparison_table",
    "plot_fit", "plot_arbitrage_map", "plot_training", "plot_quote", "plot_model_comparison",
    # registry
    "RunRecord", "save_run", "list_runs", "load_run", "load_history", "runs_table",
]
