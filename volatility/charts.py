"""
volatility/charts.py
Academic volatility charts (Okabe-Ito palette, DejaVu Serif).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

_BLUE   = "#0072B2"
_ORANGE = "#E69F00"
_GREEN  = "#009E73"
_RED    = "#D55E00"
_PURPLE = "#CC79A7"
_GRAY   = "#888888"

_COLORS = [_BLUE, _ORANGE, _GREEN, _RED, _PURPLE, _GRAY]

ACADEMIC_STYLE = {
    "font.family":       "DejaVu Serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.28,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        110,
}


def _apply():
    plt.rcParams.update(ACADEMIC_STYLE)


def _pct_fmt(ax, axis="y"):
    fmt = mtick.FuncFormatter(lambda y, _: f"{y*100:.0f}%")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def _date_fmt(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


# ── 1. Realized vol estimators overlaid on price ──────────────────────────────

def plot_realized_vol(price: pd.Series, vol_df: pd.DataFrame,
                      out_dir: str = ".", ticker: str = "TICKER") -> str:
    """
    Top panel  : price history
    Bottom panel: all realized vol estimators (CC, Parkinson, GK, RS, YZ, EWMA)
    Saves: realized_vol.png
    """
    _apply()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 2]})
    fig.suptitle(f"{ticker} — Realized Volatility Estimators  "
                 f"({len(price)} days)", fontsize=12)

    # Price
    ax = axes[0]
    ax.plot(price.index, price.values, color=_BLUE, lw=1.2)
    ax.set(ylabel="Price ($)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y:,.0f}"))

    # Vol estimators
    ax = axes[1]
    for col, color in zip(vol_df.columns, _COLORS):
        ax.plot(vol_df.index, vol_df[col], lw=1.2, color=color,
                alpha=0.85, label=col)
    ax.set(xlabel="Date", ylabel="Annualized Vol")
    _pct_fmt(ax)
    ax.legend(loc="upper right", ncol=2)
    _date_fmt(ax)

    plt.tight_layout()
    fname = "realized_vol.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname


# ── 2. GARCH fitted conditional vol + forecast cone ──────────────────────────

def plot_garch_forecast(result, fcast_df: pd.DataFrame,
                        price: pd.Series = None,
                        out_dir: str = ".", ticker: str = "TICKER") -> str:
    """
    Top (optional) : price history
    Bottom         : GARCH conditional vol (fitted) + h-day forecast cone
    Saves: garch_forecast.png
    """
    _apply()
    has_price = price is not None
    n_rows    = 2 if has_price else 1
    ratios    = [1, 2] if has_price else [1]
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 6 if has_price else 4),
                             sharex=False, squeeze=False,
                             gridspec_kw={"height_ratios": ratios})
    fig.suptitle(f"{ticker} — GARCH(1,1)  "
                 f"α={result.alpha:.4f}  β={result.beta:.4f}  "
                 f"α+β={result.persistence:.4f}  "
                 f"σ_∞={result.long_run_vol:.1%}", fontsize=11)

    if has_price:
        ax = axes[0, 0]
        ax.plot(price.index, price.values, color=_BLUE, lw=1.1)
        ax.set(ylabel="Price ($)")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y:,.0f}"))
        _date_fmt(ax)

    ax = axes[-1, 0]
    cond = result.conditional_vol
    ax.plot(cond.index, cond.values, color=_BLUE, lw=1.2, label="GARCH fitted vol")
    ax.axhline(result.long_run_vol, color=_GRAY, ls="--", lw=1,
               label=f"Long-run σ = {result.long_run_vol:.1%}")

    # Forecast
    last_date = cond.index[-1]
    try:
        fut_dates = pd.bdate_range(last_date, periods=len(fcast_df) + 1)[1:]
    except Exception:
        fut_dates = pd.date_range(last_date, periods=len(fcast_df) + 1,
                                  freq="B")[1:]

    fv  = fcast_df["forecast_vol"].values
    lb  = fcast_df["vol_lb_95"].values
    ub  = fcast_df["vol_ub_95"].values
    ax.plot(fut_dates, fv, color=_ORANGE, lw=2, label="Forecast")
    ax.fill_between(fut_dates, lb, ub, alpha=0.2, color=_ORANGE, label="95% CI")

    _pct_fmt(ax)
    ax.set(xlabel="Date", ylabel="Annualized Vol")
    ax.legend(loc="upper right")
    _date_fmt(ax)

    plt.tight_layout()
    fname = "garch_forecast.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname


# ── 3. VRP — implied vs realized, premium shading ────────────────────────────

def plot_vrp(implied_vol: pd.Series, realized_vol: pd.Series,
             vrp_series: pd.Series,
             out_dir: str = ".", ticker: str = "TICKER") -> str:
    """
    Top panel   : IV (dashed) vs RV (solid)
    Bottom panel: VRP = IV − RV with shading (green=IV>RV, red=RV>IV)
    Saves: vrp.png
    """
    _apply()
    idx  = implied_vol.index.intersection(realized_vol.index)
    IV   = implied_vol.reindex(idx)
    RV   = realized_vol.reindex(idx)
    VRP  = vrp_series.reindex(idx)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"{ticker} — Volatility Risk Premium (VRP = IV − RV)", fontsize=12)

    ax = axes[0]
    ax.plot(idx, IV.values, color=_ORANGE, lw=1.5, ls="--", label="Implied Vol (IV)")
    ax.plot(idx, RV.values, color=_BLUE,   lw=1.5,           label="Realized Vol (RV)")
    ax.fill_between(idx, IV.values, RV.values,
                    where=IV.values >= RV.values, alpha=0.12,
                    color=_GREEN, label="IV > RV (seller benefits)")
    ax.fill_between(idx, IV.values, RV.values,
                    where=IV.values < RV.values, alpha=0.12, color=_RED)
    _pct_fmt(ax)
    ax.set(ylabel="Annualized Vol")
    ax.legend()
    _date_fmt(ax)

    ax = axes[1]
    v = VRP.values
    ax.bar(idx, v, color=np.where(v >= 0, _GREEN, _RED),
           alpha=0.75, width=1.5)
    ax.axhline(0, color="black", lw=0.7)
    mean_vrp = float(np.nanmean(v))
    ax.axhline(mean_vrp, color=_GRAY, ls="--", lw=1,
               label=f"Mean VRP = {mean_vrp*100:.1f}%")
    _pct_fmt(ax)
    ax.set(xlabel="Date", ylabel="VRP")
    ax.legend()
    _date_fmt(ax)

    plt.tight_layout()
    fname = "vrp.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname


# ── 4. GARCH residuals diagnostic ────────────────────────────────────────────

def plot_garch_diagnostics(result, out_dir: str = ".",
                           ticker: str = "TICKER") -> str:
    """
    2×2 diagnostic panel for GARCH(1,1) fit:
      [0,0] Standardized residuals time series
      [0,1] ACF of residuals²  (should be near zero if GARCH fits well)
      [1,0] QQ-plot vs N(0,1)
      [1,1] Conditional vol distribution (histogram)
    Saves: garch_diagnostics.png
    """
    from scipy.stats import probplot

    _apply()
    z    = result.residuals.dropna().values
    cvol = result.conditional_vol.dropna().values

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(f"{ticker} — GARCH(1,1) Diagnostics", fontsize=12)

    # Residuals time series
    ax = axes[0, 0]
    ax.plot(result.residuals.index, z, color=_BLUE, lw=0.6, alpha=0.7)
    ax.axhline(0,  color="black", lw=0.5)
    ax.axhline(2,  color=_RED,    ls="--", lw=0.8)
    ax.axhline(-2, color=_RED,    ls="--", lw=0.8)
    ax.set(title="Standardized Residuals", ylabel="z_t")
    _date_fmt(ax)

    # ACF of z²
    ax = axes[0, 1]
    max_lag = min(20, len(z) // 5)
    lags_   = np.arange(1, max_lag + 1)
    z2      = z ** 2 - z.mean() ** 2
    acfs    = [np.corrcoef(z2[:-k], z2[k:])[0, 1] for k in lags_]
    ci      = 1.96 / np.sqrt(len(z))
    ax.bar(lags_, acfs, color=_BLUE, alpha=0.7)
    ax.axhline(ci,  color=_RED, ls="--", lw=1, label="95% CI")
    ax.axhline(-ci, color=_RED, ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set(title="ACF of Squared Residuals", xlabel="Lag", ylabel="Autocorrelation")
    ax.legend()

    # QQ plot
    ax = axes[1, 0]
    pp = probplot(z, dist="norm")
    ax.scatter(pp[0][0], pp[0][1], s=4, color=_BLUE, alpha=0.5)
    xmin, xmax = pp[0][0].min(), pp[0][0].max()
    slope, intercept = pp[1][0], pp[1][1]
    ax.plot([xmin, xmax],
            [slope * xmin + intercept, slope * xmax + intercept],
            color=_RED, lw=1.5)
    ax.set(title="QQ-Plot vs N(0,1)",
           xlabel="Theoretical quantile", ylabel="Sample quantile")

    # Conditional vol histogram
    ax = axes[1, 1]
    ax.hist(cvol, bins=40, color=_ORANGE, alpha=0.75, edgecolor="white")
    ax.axvline(np.mean(cvol), color=_RED, lw=1.5, ls="--",
               label=f"Mean = {np.mean(cvol):.1%}")
    ax.axvline(result.long_run_vol, color=_BLUE, lw=1.5, ls=":",
               label=f"σ_∞ = {result.long_run_vol:.1%}")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.set(title="Conditional Vol Distribution", xlabel="Annualized Vol")
    ax.legend()

    plt.tight_layout()
    fname = "garch_diagnostics.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname
