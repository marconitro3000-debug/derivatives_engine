"""
volsurface.evaluation
=====================

How a surface is judged, and the archive of what has been judged.

Every model in `volsurface.surfaces` implements the same interface, so every
model passes through the same two questions here:

* how closely does it reproduce the quotes  (`diagnostics.fit_report`)
* does the surface it defines *between* them admit static arbitrage
  (`diagnostics.scan_arbitrage`)

Reporting only the first is how a model that prices negative probabilities wins
a comparison.

Module map
----------
    diagnostics  fit reports and the dense-grid arbitrage scan
    report       the scorecard, the training curves, the smiles, the
                 arbitrage maps
    registry     the archive of trained runs -- which checkpoint to actually
                 use, ranked by validation error and generalisation gap
"""

from .diagnostics import (
    ArbitrageReport,
    FitReport,
    butterfly_g,
    fit_report,
    scan_arbitrage,
)
from .registry import RunRecord, list_runs, load_history, load_run, runs_table, save_run
from .report import (
    compare,
    comparison_table,
    evaluate_surface,
    plot_arbitrage_map,
    plot_fit,
    plot_model_comparison,
    plot_quote,
    plot_training,
    worst_quote_notes,
)

__all__ = [
    "ArbitrageReport",
    "FitReport",
    "RunRecord",
    "butterfly_g",
    "compare",
    "comparison_table",
    "evaluate_surface",
    "fit_report",
    "list_runs",
    "load_history",
    "load_run",
    "plot_arbitrage_map",
    "plot_fit",
    "plot_model_comparison",
    "plot_quote",
    "plot_training",
    "runs_table",
    "save_run",
    "scan_arbitrage",
    "worst_quote_notes",
]
