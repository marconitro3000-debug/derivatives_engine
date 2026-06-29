"""
portfolio/charts.py
Academic-style risk visualisations.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors

_BLUE   = "#0072B2"
_ORANGE = "#E69F00"
_GREEN  = "#009E73"
_RED    = "#D55E00"
_PURPLE = "#CC79A7"

STYLE = {
    "font.family": "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
}


def plot_pnl_distribution(var_result, title: str = "P&L Distribution",
                           ax: plt.Axes | None = None) -> plt.Figure:
    """Histogram of simulated P&L with VaR and CVaR markers."""
    plt.rcParams.update(STYLE)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    pnl = var_result.pnl_series
    ax.hist(pnl, bins=100, color=_BLUE, alpha=0.7, density=True,
            label="P&L")
    ax.axvline(-var_result.var, color=_RED, lw=2,
               label=f"VaR {var_result.confidence*100:.0f}% = {var_result.var:,.0f}")
    ax.axvline(-var_result.cvar, color=_ORANGE, lw=2, ls="--",
               label=f"CVaR = {var_result.cvar:,.0f}")
    ax.axvline(0, color="black", lw=0.8, ls=":")

    ax.set(title=f"{title}  [{var_result.method}]",
           xlabel="P&L ($)", ylabel="Density")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    if standalone:
        plt.tight_layout()
    return fig


def plot_var_comparison(results_dict: dict, ax: plt.Axes | None = None) -> plt.Figure:
    """Bar chart comparing VaR and CVaR across methods."""
    plt.rcParams.update(STYLE)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    methods = list(results_dict.keys())
    vars_   = [r.var  for r in results_dict.values()]
    cvars   = [r.cvar for r in results_dict.values()]
    x       = np.arange(len(methods))
    w       = 0.35

    ax.bar(x - w/2, vars_,  w, color=_RED,    alpha=0.85, label="VaR")
    ax.bar(x + w/2, cvars,  w, color=_ORANGE, alpha=0.85, label="CVaR")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in methods])
    ax.set(title="VaR / CVaR by Method", ylabel="Loss ($)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend()
    if standalone:
        plt.tight_layout()
    return fig


def plot_greeks_bar(portfolio, ax: plt.Axes | None = None) -> plt.Figure:
    """Grouped bar chart: per-position Greeks."""
    plt.rcParams.update(STYLE)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.get_figure()

    rows = portfolio.greeks_by_position()
    labels  = [r["label"]  for r in rows]
    deltas  = [r["delta"]  for r in rows]
    gammas  = [r["gamma"]  for r in rows]
    vegas   = [r["vega"]   for r in rows]
    thetas  = [r["theta"]  for r in rows]

    x = np.arange(len(labels))
    w = 0.2

    ax.bar(x - 1.5*w, deltas, w, color=_BLUE,   alpha=0.85, label="Delta")
    ax.bar(x - 0.5*w, gammas, w, color=_ORANGE, alpha=0.85, label="Gamma")
    ax.bar(x + 0.5*w, vegas,  w, color=_GREEN,  alpha=0.85, label="Vega")
    ax.bar(x + 1.5*w, thetas, w, color=_RED,    alpha=0.85, label="Theta")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set(title="Greeks by Position", ylabel="$ sensitivity")
    ax.legend(fontsize=9)
    if standalone:
        plt.tight_layout()
    return fig


def plot_stress_results(stress_results, n_top: int = 10,
                         ax: plt.Axes | None = None) -> plt.Figure:
    """Horizontal bar chart of scenario P&L, top N by magnitude."""
    plt.rcParams.update(STYLE)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    # Sort by abs PnL
    top = sorted(stress_results, key=lambda r: abs(r.pnl), reverse=True)[:n_top]
    top = top[::-1]   # bottom-to-top for readability

    names = [r.scenario.name for r in top]
    pnls  = [r.pnl           for r in top]
    cols  = [_RED if p < 0 else _GREEN for p in pnls]

    ax.barh(names, pnls, color=cols, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title=f"Stress Test P&L (Top {n_top} scenarios)",
           xlabel="P&L ($)")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    if standalone:
        plt.tight_layout()
    return fig


def plot_spot_vol_grid(grid: np.ndarray, spot_shocks: np.ndarray,
                        vol_shocks: np.ndarray,
                        ax: plt.Axes | None = None) -> plt.Figure:
    """Heatmap of PnL over spot × vol scenario grid."""
    plt.rcParams.update(STYLE)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.get_figure()

    # Diverging colormap centred on 0
    vmax = np.abs(grid).max()
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(grid, cmap="RdYlGn", norm=norm, aspect="auto", origin="lower")
    plt.colorbar(im, ax=ax, label="P&L ($)")

    xs = [f"{v:+.0f}%" for v in vol_shocks * 100]
    ys = [f"{s:+.0f}%" for s in spot_shocks * 100]
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys, fontsize=7)
    ax.set(title="P&L Grid: Spot × Volatility Shocks",
           xlabel="Vol shock", ylabel="Spot shock")

    # Contour at zero
    ax.contour(grid, levels=[0], colors=["black"], linewidths=1.5)
    if standalone:
        plt.tight_layout()
    return fig
