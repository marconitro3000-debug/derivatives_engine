"""
volsurface
==========

An arbitrage-penalised neural implied-volatility surface, fitted to real option
chains and benchmarked against the parametric surfaces a desk actually uses.

The whole library is one idea. A volatility surface is defined by its total
implied variance ``w(k, T) = sigma(k, T)^2 * T`` over log-moneyness and
maturity, every model implements the same `surfaces.base.VolSurface` interface,
and every model is therefore judged by the same two questions:

* how closely does it reproduce the quotes
  (`evaluation.diagnostics.fit_report`)
* does the surface it defines *between* them admit static arbitrage
  (`evaluation.diagnostics.scan_arbitrage`)

Reporting only the first is how a model that prices negative probabilities wins
a comparison.

Typical use
-----------
    from volsurface import fetch_chain, train_surface, compare, comparison_table

    chain  = fetch_chain("SPY")
    result = train_surface(chain)
    print(comparison_table(compare(chain, result)))

    from volsurface import price_option
    q = price_option(result.model, chain, strike=780, expiry="2026-12-19", kind="put")

Where things are
----------------
Four packages, each named after the question it answers. Every name below is
importable from `volsurface` directly as well -- this module re-exports the
whole public surface -- so the layout is for reading the code, not a tax on
using it.

``volsurface.data``       **where the quotes come from.**
                          `fetch_chain` (live), `synthetic_snapshot` (the
                          offline generator the tests fit), `build_snapshot`
                          (raw frames -> clean chain, every filter), `forward`
                          (the forward and discount from put-call parity),
                          `ChainSnapshot`, day counts.

``volsurface.pricing``    **what a price is.** Black-Scholes closed form,
                          implied-vol inversion, the American lattice and the
                          de-Americanisation it enables, Monte Carlo as an
                          independent check. Primitives: numbers in, numbers
                          out, no dependency on anything else here.

``volsurface.surfaces``   **the models.** `base` is the interface, `svi` the
                          parametric baselines (raw SVI per expiry, joint
                          SSVI), and **`neural` is the network** -- prior,
                          model, penalties, dataset, training loop.

``volsurface.evaluation`` **how a surface is judged.** Fit reports, the
                          dense-grid arbitrage scan, the scorecard and figures,
                          and the archive of trained runs.

``volsurface.quote``      **using a fitted surface.** `price_option` -- strike,
                          expiry date and side in, price, Greeks and the
                          caveats needed to judge the number out. It sits above
                          the four packages because it consumes all of them.

The dependencies run strictly one way, ``pricing -> surfaces -> data ->
evaluation -> quote``, and `tests/test_layout.py` fails if that stops being
true.
"""

from .data import (
    ChainSnapshot,
    build_snapshot,
    fetch_chain,
    fit_forward,
    implied_forward,
    synthetic_snapshot,
    year_fraction,
)
from .evaluation import (
    ArbitrageReport,
    FitReport,
    RunRecord,
    butterfly_g,
    capacity_table,
    compare,
    comparison_table,
    evaluate_surface,
    fit_report,
    list_runs,
    load_history,
    load_run,
    plot_arbitrage_map,
    plot_capacity,
    plot_fit,
    plot_model_comparison,
    plot_quote,
    plot_run_overlay,
    plot_training,
    runs_table,
    save_run,
    scan_arbitrage,
    worst_quote_notes,
)
from .pricing import (
    american_implied_vol,
    binomial_price,
    carry_from_forward,
    de_americanised_iv,
    greeks,
    implied_vol,
    mc_price,
    price,
    put_call_parity_check,
)
from .quote import MAX_YEARS, OptionQuote, price_option, resolve_maturity
from .surfaces import (
    ModelConfig,
    NeuralVolSurface,
    PenaltyWeights,
    SSVIParams,
    SSVISurface,
    SVIParams,
    SVISliceSurface,
    TrainConfig,
    TrainResult,
    VolSurface,
    train_surface,
)

__version__ = "1.1.0"

__all__ = [
    # market data
    "ChainSnapshot", "fetch_chain", "build_snapshot", "synthetic_snapshot",
    "implied_forward", "fit_forward", "year_fraction",
    # pricing primitives
    "price", "greeks", "put_call_parity_check", "implied_vol",
    "binomial_price", "de_americanised_iv", "american_implied_vol",
    "carry_from_forward", "mc_price",
    # pricing one named option off a fitted surface
    "OptionQuote", "price_option", "resolve_maturity",
    # surfaces
    "VolSurface", "SVIParams", "SVISliceSurface", "SSVIParams", "SSVISurface",
    "NeuralVolSurface", "ModelConfig",
    # fitting
    "TrainConfig", "TrainResult", "PenaltyWeights", "train_surface",
    # scoring
    "FitReport", "ArbitrageReport", "fit_report", "scan_arbitrage", "butterfly_g",
    "evaluate_surface", "compare", "comparison_table", "worst_quote_notes",
    "capacity_table", "plot_fit", "plot_arbitrage_map", "plot_training", "plot_quote",
    "plot_model_comparison", "plot_capacity", "plot_run_overlay",
    # registry
    "RunRecord", "save_run", "list_runs", "load_run", "load_history", "runs_table",
]
