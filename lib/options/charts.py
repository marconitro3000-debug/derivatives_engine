"""
options/charts.py
Academic-quality chart suite for single-option analysis.

Four chart families
-------------------
plot_history      : 52-week price history + realized vs implied vol panel
plot_option_value : BS value curves for multiple time horizons (shows time decay)
plot_greeks       : 2×2 Greek sensitivities vs spot (Δ, Γ, Θ, Vega)
plot_pnl          : Profit/loss at expiry diagram for long position

All functions save a PNG to out_dir and return the filename.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from .black_scholes import price as bs_price, greeks as bs_greeks


# ── color palette (Okabe-Ito — colorblind safe) ───────────────────────────────

_BLUE   = "#0072B2"
_ORANGE = "#E69F00"
_GREEN  = "#009E73"
_RED    = "#D55E00"
_PURPLE = "#CC79A7"
_GRAY   = "#6C757D"
_LBLUE  = "#56B4E9"

# blue gradient for time-decay curves (dark = long time, light = short time)
_BLUES = ["#08306b", "#2171b5", "#6baed6", "#bdd7e7"]

ACADEMIC_STYLE = {
    "font.family":        ["DejaVu Serif", "serif"],
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     ":",
    "grid.alpha":         0.35,
    "lines.linewidth":    2.0,
    "figure.dpi":         150,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "legend.framealpha":  0.93,
    "legend.edgecolor":   "#cccccc",
    "legend.fontsize":    9.5,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _apply():
    plt.rcParams.update(ACADEMIC_STYLE)


def _save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return name


def _spot_grid(K: float, S: float, n: int = 300) -> np.ndarray:
    lo = min(K * 0.65, S * 0.68)
    hi = max(K * 1.42, S * 1.38)
    return np.linspace(lo, hi, n)


def _time_steps(T: float) -> list[tuple[float, str]]:
    """Return (T_value, label) pairs from current T down to payoff."""
    fracs  = [1.0, 0.60, 0.30, 0.10]
    labels = ["now", None, None, None]
    steps  = []
    for f, lbl in zip(fracs, labels):
        t = T * f
        if t < 0.5 / 365:
            break
        tag = f"T = {t * 365:.0f}d" + ("  (now)" if lbl else "")
        steps.append((t, tag))
    steps.append((1e-5, "Payoff  (T → 0)"))
    return steps


# ── 1. 52-week history + option setup ────────────────────────────────────────

def plot_history(hist_df, S: float, K: float, T: float, sigma: float,
                 option_type: str, ticker: str, out_dir: str) -> str | None:
    """
    Two-panel chart:
      Top    — 52-week close price with strike, current spot, and option-life shading.
      Bottom — Rolling 21-day realised vol vs ATM implied vol.

    Parameters
    ----------
    hist_df     : pd.DataFrame with 'Close' column and DatetimeIndex
    S           : current spot
    K           : strike
    T           : time to expiry in years
    sigma       : ATM implied volatility
    option_type : 'call' or 'put'
    ticker      : label for title
    out_dir     : output directory
    """
    import pandas as pd

    _apply()

    # --- data prep -----------------------------------------------------------
    close = hist_df["Close"].dropna()
    if hasattr(close.columns if hasattr(close, "columns") else [], "__len__"):
        pass  # already a Series
    if len(close) < 15:
        return None

    dates   = close.index
    log_ret = np.log(close / close.shift(1))
    rvol_21 = log_ret.rolling(21).std() * np.sqrt(252) * 100   # ann. %

    last_date   = pd.Timestamp(dates[-1]).tz_localize(None) if dates[-1].tzinfo else pd.Timestamp(dates[-1])
    expiry_date = last_date + pd.Timedelta(days=max(int(T * 365), 1))

    # make dates tz-naive for plotting
    plot_dates = [pd.Timestamp(d).tz_localize(None) if hasattr(d, "tzinfo") and d.tzinfo else pd.Timestamp(d)
                  for d in dates]

    # --- figure layout -------------------------------------------------------
    fig = plt.figure(figsize=(13, 6.5))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ── top panel: price ──────────────────────────────────────────────────────
    cv = close.values
    ax1.plot(plot_dates, cv, color=_BLUE, lw=1.6, label="Close price", zorder=3)
    ax1.fill_between(plot_dates, cv.min() * 0.994, cv,
                     color=_BLUE, alpha=0.07, zorder=1)

    # strike line
    ax1.axhline(K, color=_ORANGE, ls="--", lw=1.8, zorder=2,
                label=f"Strike  K = ${K:,.2f}")

    # option-life shading (from today onward)
    ax1.axvspan(last_date, expiry_date, color=_ORANGE, alpha=0.11, zorder=1,
                label=f"Option life  ({T * 365:.0f} d)")
    ax1.axvline(last_date,   color=_GRAY,   lw=0.9, ls="-",  zorder=2)
    ax1.axvline(expiry_date, color=_ORANGE, lw=1.0, ls=":",  zorder=2)
    ax1.text(expiry_date, cv.max() * 1.003, "  Expiry",
             fontsize=8.5, color=_ORANGE, va="bottom")

    # current spot dot + annotation
    ax1.scatter([last_date], [S], color=_BLUE, s=75, zorder=6)
    ax1.annotate(
        f"  S = ${S:,.2f}",
        xy=(last_date, S), xytext=(6, 8), textcoords="offset points",
        fontsize=9.5, color=_BLUE, fontweight="bold",
    )

    # moneyness badge
    pct = (S / K - 1) * 100
    itm = (option_type == "call" and S > K) or (option_type == "put" and S < K)
    badge = ("ATM" if abs(pct) < 2
             else f"{abs(pct):.1f}% {'ITM' if itm else 'OTM'}")
    badge_color = _GREEN if itm else _PURPLE
    ax1.text(0.012, 0.035, badge, transform=ax1.transAxes, fontsize=10,
             color=badge_color, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=badge_color, alpha=0.85))

    ax1.set_ylabel("Price ($)", labelpad=6)
    ax1.set_title(
        f"{ticker}  —  52-Week Price History  ·  "
        f"{option_type.capitalize()}  K = ${K:,.0f}  |  "
        f"T = {T * 365:.0f} d  |  σ = {sigma:.1%}",
        pad=9, fontsize=12,
    )
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax1.legend(loc="upper left", ncol=3, fontsize=9)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── bottom panel: realised vs implied vol ─────────────────────────────────
    rv = rvol_21.dropna()
    rv_dates = [pd.Timestamp(d).tz_localize(None) if hasattr(d, "tzinfo") and d.tzinfo
                else pd.Timestamp(d) for d in rv.index]

    ax2.plot(rv_dates, rv.values, color=_GRAY, lw=1.3, label="Realised vol (21 d)")
    ax2.fill_between(rv_dates, 0, rv.values, color=_GRAY, alpha=0.14)
    ax2.axhline(sigma * 100, color=_RED, ls="--", lw=1.8,
                label=f"Implied vol  σ = {sigma:.1%}")

    # IV label at right edge
    ax2.text(rv_dates[-1], sigma * 100, f"  IV = {sigma:.1%}",
             va="center", fontsize=8.5, color=_RED)

    ax2.set_ylabel("Ann. vol (%)", labelpad=6)
    ax2.set_ylim(bottom=0, top=max(rv.max() * 1.15, sigma * 155))
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.legend(loc="upper left", fontsize=9)

    fig.autofmt_xdate(rotation=25, ha="right")
    plt.tight_layout()
    return _save(fig, out_dir, "history.png")


# ── 2. Option value curves ────────────────────────────────────────────────────

def plot_option_value(S: float, K: float, T: float, r: float, sigma: float,
                      option_type: str, out_dir: str,
                      style: str = "european",
                      premium: float = None,
                      ticker: str = "") -> str:
    """
    Option value as a function of spot for multiple time horizons.

    Shows:
      • A family of value curves from T = now down to T → 0 (payoff / intrinsic value).
      • Time-value shading between the current curve and intrinsic.
      • Current spot, strike, and option price markers.
    """
    _apply()

    S_arr  = _spot_grid(K, S)
    tsteps = _time_steps(T)

    # colours: dark blue (long T) → light blue → red (intrinsic)
    n_mid = len(tsteps) - 1
    mid_colors = plt.cm.Blues_r(np.linspace(0.08, 0.68, max(n_mid, 1)))
    colors = list(mid_colors) + [_RED]
    lws    = [2.6] + [1.7] * (n_mid - 1) + [1.7]
    lss    = ["solid"] * n_mid + ["dashed"]

    fig, ax = plt.subplots(figsize=(10, 6))

    first_curve = None
    for i, ((t_val, t_label), col, lw, ls) in enumerate(
            zip(tsteps, colors, lws, lss)):
        vals = []
        for s in S_arr:
            try:
                if t_val < 1 / 365:
                    v = max(s - K, 0) if option_type == "call" else max(K - s, 0)
                else:
                    v = bs_price(s, K, t_val, r, sigma, option_type)
            except Exception:
                v = 0.0
            vals.append(v)
        vals = np.array(vals)
        ax.plot(S_arr, vals, color=col, lw=lw, ls=ls, label=t_label, zorder=3)
        if i == 0:
            first_curve = vals

    # time-value shading: between intrinsic and current T curve
    intrinsic = np.array([
        max(s - K, 0) if option_type == "call" else max(K - s, 0)
        for s in S_arr
    ])
    if first_curve is not None:
        ax.fill_between(S_arr, intrinsic, first_curve,
                        where=(first_curve >= intrinsic),
                        color=mid_colors[0], alpha=0.10, zorder=1,
                        label="Time value (now)")

    # key vertical lines
    ax.axvline(K, color=_ORANGE, ls="--", lw=1.5, alpha=0.85,
               label=f"Strike  K = ${K:,.2f}")
    ax.axvline(S, color=_LBLUE,  ls=":",  lw=1.5, alpha=0.85,
               label=f"Spot  S = ${S:,.2f}")

    # current option price marker
    if premium is not None:
        ax.scatter([S], [premium], color=_BLUE, s=110, zorder=6,
                   label=f"Current price  ${premium:.4f}")
        ax.annotate(
            f"  ${premium:.2f}",
            xy=(S, premium), xytext=(6, 0), textcoords="offset points",
            fontsize=9.5, color=_BLUE,
        )

    # note for American puts (BS curves are European lower bounds)
    if style == "american" and option_type == "put":
        ax.text(0.02, 0.97,
                "American put has early-exercise premium\n"
                "(curves shown are European lower bounds)",
                transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round", fc="lightyellow",
                          ec="#ccaa00", alpha=0.9))

    ax.set_xlabel("Spot price  S ($)", labelpad=8)
    ax.set_ylabel("Option value ($)", labelpad=8)
    ax.set_title(
        f"{option_type.capitalize()} Option Value vs Spot  "
        f"[{ticker}  K = ${K:,.0f}  |  σ = {sigma:.1%}  |  r = {r:.2%}]",
        pad=10,
    )
    if first_curve is not None:
        ax.set_ylim(bottom=-0.04 * float(first_curve.max()))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y:,.2f}"))
    ax.legend(loc="upper left", ncol=2, fontsize=9)

    plt.tight_layout()
    return _save(fig, out_dir, "option_value.png")


# ── 3. Greeks ─────────────────────────────────────────────────────────────────

def plot_greeks(S: float, K: float, T: float, r: float, sigma: float,
                option_type: str, out_dir: str, ticker: str = "") -> str:
    """
    2×2 panel of option Greeks vs spot price.

    Panels
    ------
    Δ Delta   — sensitivity to spot (sigmoid for call, negative sigmoid for put)
    Γ Gamma   — rate of change of Delta (bell-shaped, same for call & put)
    Θ Theta   — time-value decay per calendar day (negative)
    Vega      — sensitivity to σ per 1% change (bell-shaped, same for call & put)
    """
    _apply()

    S_arr = _spot_grid(K, S)

    # vectorised greek computation
    deltas, gammas, thetas, vegas = [], [], [], []
    for s in S_arr:
        try:
            g = bs_greeks(s, K, T, r, sigma)
            deltas.append(g[f"delta_{option_type}"])
            gammas.append(g["gamma"])
            thetas.append(g[f"theta_{option_type}"])
            vegas.append(g["vega"])
        except Exception:
            deltas.append(np.nan)
            gammas.append(np.nan)
            thetas.append(np.nan)
            vegas.append(np.nan)

    greeks_now = bs_greeks(S, K, T, r, sigma)

    panels = [
        ("Δ  Delta",          np.array(deltas), _BLUE,   greeks_now[f"delta_{option_type}"]),
        ("Γ  Gamma",          np.array(gammas), _GREEN,  greeks_now["gamma"]),
        ("Θ  Theta (per day)",np.array(thetas), _RED,    greeks_now[f"theta_{option_type}"]),
        ("Vega (per 1% σ)",   np.array(vegas),  _PURPLE, greeks_now["vega"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Option Greeks vs Spot  [{ticker}  K = ${K:,.0f}  |  "
        f"T = {T * 365:.0f} d  |  σ = {sigma:.1%}  |  {option_type.capitalize()}]",
        fontsize=12, y=1.01,
    )

    for ax, (label, vals, color, val_now) in zip(axes.flat, panels):
        ax.plot(S_arr, vals, color=color, lw=2.3, zorder=3)

        # reference lines
        ax.axvline(K, color=_ORANGE, ls="--", lw=1.3, alpha=0.75,
                   label=f"K = ${K:,.0f}")
        ax.axvline(S, color=_LBLUE,  ls=":",  lw=1.3, alpha=0.75,
                   label=f"S = ${S:,.2f}")
        ax.axhline(0, color=_GRAY,   ls="-",  lw=0.7, alpha=0.50)

        # current value dot + annotation
        ax.scatter([S], [val_now], color=color, s=80, zorder=5)
        ax.annotate(
            f"  {val_now:.4f}",
            xy=(S, val_now), xytext=(5, 5), textcoords="offset points",
            fontsize=8.5, color=color,
        )

        ax.set_xlabel("Spot  S ($)", labelpad=5)
        ax.set_ylabel(label, labelpad=5)
        ax.xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(fontsize=8.5, loc="best")

    # tighter delta y-axis
    if option_type == "call":
        axes[0, 0].set_ylim(-0.06, 1.06)
    else:
        axes[0, 0].set_ylim(-1.06, 0.06)

    plt.tight_layout()
    return _save(fig, out_dir, "greeks.png")


# ── 4. P&L at expiry ──────────────────────────────────────────────────────────

def plot_pnl(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str, premium: float, out_dir: str,
             ticker: str = "") -> str:
    """
    Profit/loss diagram at expiry for the long option position.

    Shows payoff minus premium paid, with break-even, max-loss, and shaded
    profit/loss regions.
    """
    _apply()

    S_arr = _spot_grid(K, S, n=400)

    if option_type == "call":
        intrinsic = np.maximum(S_arr - K, 0.0)
        be        = K + premium
    else:
        intrinsic = np.maximum(K - S_arr, 0.0)
        be        = K - premium

    pnl_long  =  intrinsic - premium   # buyer
    pnl_short = -intrinsic + premium   # seller (writer)

    fig, ax = plt.subplots(figsize=(10, 6))

    # P&L curves
    ax.plot(S_arr, pnl_long,  color=_BLUE,   lw=2.5, label="Long (buyer)",  zorder=4)
    ax.plot(S_arr, pnl_short, color=_PURPLE,  lw=1.8, ls="--",
            label="Short (writer)", zorder=3, alpha=0.75)

    # shaded regions for long position
    ax.fill_between(S_arr, pnl_long, 0,
                    where=(pnl_long > 0),  color=_GREEN, alpha=0.17, zorder=2)
    ax.fill_between(S_arr, pnl_long, 0,
                    where=(pnl_long <= 0), color=_RED,   alpha=0.17, zorder=2)

    # reference lines
    ax.axhline(0,        color=_GRAY,   lw=0.9, ls="-",  alpha=0.6, zorder=2)
    ax.axvline(K,        color=_ORANGE, lw=1.5, ls="--", alpha=0.8, zorder=3,
               label=f"Strike  K = ${K:,.2f}")
    ax.axvline(S,        color=_LBLUE,  lw=1.5, ls=":",  alpha=0.8, zorder=3,
               label=f"Spot now  S = ${S:,.2f}")
    ax.axvline(be,       color=_GREEN,  lw=1.5, ls="-.", alpha=0.8, zorder=3,
               label=f"Break-even  ${be:,.2f}")
    ax.axhline(-premium, color=_RED,    lw=0.9, ls=":",  alpha=0.6, zorder=2)

    # annotations
    ax.annotate(
        f"Max loss = −${premium:.2f}",
        xy=(S_arr[20], -premium),
        xytext=(0, -18), textcoords="offset points",
        fontsize=9, color=_RED, ha="left",
    )
    ax.annotate(
        f"Break-even\n${be:,.2f}",
        xy=(be, 0), xytext=(0, 18), textcoords="offset points",
        fontsize=9, color=_GREEN, ha="center",
        arrowprops=dict(arrowstyle="->", color=_GREEN, lw=1.0),
    )

    # current intrinsic P&L dot
    if option_type == "call":
        cur_intr = max(S - K, 0.0)
    else:
        cur_intr = max(K - S, 0.0)
    cur_pnl = cur_intr - premium
    ax.scatter([S], [cur_pnl], color=_BLUE, s=90, zorder=6,
               label=f"At current S:  ${cur_pnl:+.2f}")

    # profit/loss region labels
    if option_type == "call":
        ax.text(S_arr[-30], pnl_long[-30] * 0.5,
                "Profit\n(unlimited)", fontsize=8.5, color=_GREEN,
                ha="right", va="center")
    else:
        ax.text(S_arr[20], pnl_long[20] * 0.5,
                "Profit\n(capped at K−P)", fontsize=8.5, color=_GREEN,
                ha="left", va="center")
    ax.text(
        S_arr[len(S_arr) // 2],
        -premium * 0.55,
        f"Max loss = −${premium:.2f}",
        fontsize=8.5, color=_RED, ha="center",
    )

    ax.set_xlabel("Spot at expiry  $S_T$  ($)", labelpad=8)
    ax.set_ylabel("Profit / Loss ($)", labelpad=8)
    ax.set_title(
        f"Long {option_type.capitalize()} P&L at Expiry  "
        f"[{ticker}  K = ${K:,.0f}  |  Premium = ${premium:.2f}  |  "
        f"T = {T * 365:.0f} d]",
        pad=10,
    )
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y:+.2f}"))
    legend_loc = "upper left" if option_type == "call" else "upper right"
    ax.legend(loc=legend_loc, ncol=2, fontsize=9)

    plt.tight_layout()
    return _save(fig, out_dir, "pnl.png")
