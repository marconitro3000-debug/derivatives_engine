"""
exotics/charts.py
Academic-quality chart suite for exotic option analysis.

plot_barrier_payoff    : payoff diagram + barrier line (vs vanilla)
plot_asian_paths       : sample paths + arithmetic vs geometric average shading
plot_lookback_paths    : sample paths + running min/max bands
plot_digital_payoff    : step-function payoff for cash-or-nothing
plot_exotic_comparison : price of multiple exotics vs spot (spider chart)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

try:
    from options.charts import ACADEMIC_STYLE
except ImportError:
    ACADEMIC_STYLE = {"figure.dpi": 150, "figure.facecolor": "white"}

_BLUE   = "#0072B2"
_ORANGE = "#E69F00"
_GREEN  = "#009E73"
_RED    = "#D55E00"
_PURPLE = "#CC79A7"
_GRAY   = "#6C757D"


def _apply():
    plt.rcParams.update(ACADEMIC_STYLE)


def _save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return name


# ── 1. Barrier payoff diagram ─────────────────────────────────────────────────

def plot_barrier_payoff(S: float, K: float, T: float, r: float, sigma: float,
                        H: float,
                        option_type: str = "call",
                        barrier_type: str = "down-out",
                        out_dir: str = ".",
                        ticker: str = "") -> str:
    """
    Payoff at expiry for a barrier option vs its vanilla equivalent.
    Shows the knockout/knock-in zone and the barrier level.
    """
    from exotics.barrier import price_barrier

    _apply()
    lo = min(K * 0.60, H * 0.85, S * 0.65)
    hi = max(K * 1.45, H * 1.15, S * 1.40)
    S_arr = np.linspace(lo, hi, 400)

    # Payoff at expiry (T→0): barrier already determined by path, not S_T alone.
    # For the diagram we show the intrinsic payoff in each zone.
    if option_type == "call":
        vanilla_payoff = np.maximum(S_arr - K, 0.0)
    else:
        vanilla_payoff = np.maximum(K - S_arr, 0.0)

    # Knocked-out zone: where S_T is on the wrong side of H
    if "down" in barrier_type:
        knocked = S_arr <= H
    else:
        knocked = S_arr >= H

    barrier_payoff = np.where(
        knocked,
        0.0 if "out" in barrier_type else vanilla_payoff,
        vanilla_payoff if "out" in barrier_type else 0.0,
    )

    # Price curves across spot (at inception, not expiry payoff)
    prices_vanilla  = [price_barrier(s, K, T, r, sigma, H * 0,  # move barrier far away
                                     option_type, "down-out", 0.0)["vanilla"]
                       if False else 0.0
                       for s in S_arr]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: payoff at expiry
    ax = axes[0]
    ax.plot(S_arr, vanilla_payoff,  color=_GRAY,   lw=1.5, ls="--", label="Vanilla payoff")
    ax.plot(S_arr, barrier_payoff,  color=_BLUE,   lw=2.2, label=f"{barrier_type} payoff")
    ax.fill_between(S_arr, 0, barrier_payoff, color=_BLUE, alpha=0.12)

    # Knocked-out zone shading
    if "out" in barrier_type:
        ax.fill_between(S_arr, -0.5, max(vanilla_payoff) * 1.1, where=knocked,
                        color=_RED, alpha=0.08, label="Knocked-out zone")
    else:
        ax.fill_between(S_arr, -0.5, max(vanilla_payoff) * 1.1, where=~knocked,
                        color=_GRAY, alpha=0.08, label="Barrier not yet activated")

    ax.axvline(K, color=_ORANGE, ls="--", lw=1.5, label=f"Strike K=${K:,.0f}")
    ax.axvline(H, color=_RED,    ls=":",  lw=1.8, label=f"Barrier H=${H:,.0f}")
    ax.axvline(S, color=_BLUE,   ls=":",  lw=1.2, alpha=0.7, label=f"Spot S=${S:,.2f}")
    ax.set(xlabel="Spot at expiry ($)", ylabel="Payoff ($)",
           title=f"{barrier_type.replace('-', ' ').title()} {option_type.capitalize()}"
                 f"  [{'↓' if 'down' in barrier_type else '↑'}H=${H:,.0f}]")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y:,.2f}"))
    ax.set_ylim(bottom=-0.02 * (vanilla_payoff.max() + 1))
    ax.legend(fontsize=9, loc="upper left")

    # Right: price curves across spot TODAY
    ax2 = axes[1]
    spot_range = np.linspace(lo * 1.05, hi * 0.95, 200)
    bar_prices = []
    van_prices = []
    for s in spot_range:
        if ("down" in barrier_type and s <= H) or ("up" in barrier_type and s >= H):
            bar_prices.append(0.0)
        else:
            bar_prices.append(price_barrier(s, K, T, r, sigma, H,
                                            option_type, barrier_type)["price"])
        van_prices.append(price_barrier(s, K, T, r, sigma, H,
                                        option_type, barrier_type)["vanilla"])

    ax2.plot(spot_range, van_prices, color=_GRAY,   lw=1.5, ls="--", label="Vanilla")
    ax2.plot(spot_range, bar_prices, color=_BLUE,   lw=2.2, label=f"{barrier_type}")
    ax2.fill_between(spot_range, bar_prices, van_prices,
                     color=_RED, alpha=0.10, label="Barrier discount")
    ax2.axvline(K, color=_ORANGE, ls="--", lw=1.3, label=f"K=${K:,.0f}")
    ax2.axvline(H, color=_RED,    ls=":",  lw=1.8, label=f"H=${H:,.0f}")
    ax2.axvline(S, color=_BLUE,   ls=":",  lw=1.2, alpha=0.7, label=f"S=${S:,.2f}")
    ax2.set(xlabel="Spot today ($)", ylabel="Option price ($)",
           title=f"Value Today vs Spot  [T={T*365:.0f}d  σ={sigma:.0%}]")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y:,.2f}"))
    ax2.legend(fontsize=9, loc="upper left")

    plt.suptitle(f"Barrier Option Analysis  [{ticker}  K=${K:,.0f}  H=${H:,.0f}]",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return _save(fig, out_dir, "barrier_analysis.png")


# ── 2. Asian option paths ─────────────────────────────────────────────────────

def plot_asian_paths(S: float, K: float, T: float, r: float, sigma: float,
                     option_type: str = "call",
                     out_dir: str = ".",
                     ticker: str = "",
                     n_paths: int = 50,
                     n_steps: int = 252,
                     seed: int = 42) -> str:
    """
    Visualise GBM paths and how the arithmetic average compares to S_T.
    Shows why Asian options cost less than vanilla.
    """
    _apply()
    rng     = np.random.default_rng(seed)
    dt      = T / n_steps
    Z       = rng.standard_normal((n_paths, n_steps))
    lr      = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_S   = np.log(S) + np.hstack([np.zeros((n_paths, 1)), np.cumsum(lr, axis=1)])
    paths   = np.exp(log_S)
    t_ax    = np.linspace(0, T * 252, n_steps + 1)

    S_avg = paths[:, 1:].mean(axis=1)   # arithmetic average (daily)
    S_T   = paths[:, -1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for path in paths:
        ax.plot(t_ax, path, alpha=0.15, lw=0.7, color=_BLUE)
    ax.plot(t_ax, paths.mean(axis=0), color="k", lw=2, label="Mean path")
    ax.axhline(K, color=_ORANGE, ls="--", lw=1.5, label=f"Strike K=${K:,.0f}")
    ax.scatter([t_ax[-1]] * n_paths, S_avg, color=_GREEN, s=15, alpha=0.4, zorder=5,
               label="Arithmetic avg")
    ax.scatter([t_ax[-1]] * n_paths, S_T,   color=_RED,   s=15, alpha=0.4, zorder=5,
               label="Final S_T")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.set(xlabel="Trading days", ylabel="Price ($)",
           title=f"GBM Paths  [σ={sigma:.0%}  T={T*365:.0f}d]")
    ax.legend(fontsize=9)

    # Right: distribution of S_T vs S_avg
    ax2 = axes[1]
    bins = np.linspace(min(S_T.min(), S_avg.min()) * 0.95,
                       max(S_T.max(), S_avg.max()) * 1.05, 50)
    ax2.hist(S_T,   bins=bins, alpha=0.5, color=_RED,   label="S_T (vanilla)")
    ax2.hist(S_avg, bins=bins, alpha=0.5, color=_GREEN, label="S_avg (Asian)")
    ax2.axvline(K, color=_ORANGE, ls="--", lw=2, label=f"K=${K:,.0f}")
    ax2.axvline(S_T.mean(),   color=_RED,   ls=":", lw=1.5)
    ax2.axvline(S_avg.mean(), color=_GREEN, ls=":", lw=1.5)

    # ITM fractions
    itm_van  = (S_T   > K).mean() if option_type == "call" else (S_T   < K).mean()
    itm_asia = (S_avg > K).mean() if option_type == "call" else (S_avg < K).mean()
    ax2.text(0.02, 0.97, f"ITM fraction\n  Vanilla: {itm_van:.1%}\n  Asian:   {itm_asia:.1%}",
             transform=ax2.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))

    ax2.set(xlabel="Price ($)", ylabel="Frequency",
            title="Distribution: S_T vs Arithmetic Average")
    ax2.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.legend(fontsize=9)

    plt.suptitle(f"Asian Option — Path Analysis  [{ticker}  K=${K:,.0f}]",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return _save(fig, out_dir, "asian_paths.png")


# ── 3. Lookback paths ─────────────────────────────────────────────────────────

def plot_lookback_paths(S: float, T: float, r: float, sigma: float,
                        option_type: str = "call",
                        out_dir: str = ".",
                        ticker: str = "",
                        n_paths: int = 5,
                        n_steps: int = 252,
                        seed: int = 42) -> str:
    """
    Show a few paths with their running min/max and highlight the lookback payoff.
    """
    _apply()
    rng   = np.random.default_rng(seed)
    dt    = T / n_steps
    Z     = rng.standard_normal((n_paths, n_steps))
    lr    = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.log(S) + np.hstack([np.zeros((n_paths, 1)), np.cumsum(lr, axis=1)])
    paths = np.exp(log_S)
    t_ax  = np.linspace(0, T * 252, n_steps + 1)

    colors = [_BLUE, _ORANGE, _GREEN, _RED, _PURPLE]

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (path, col) in enumerate(zip(paths, colors)):
        ax.plot(t_ax, path, color=col, lw=1.5, alpha=0.85, label=f"Path {i+1}")
        if option_type == "call":
            ext = path.min()
            payoff = path[-1] - ext
            ax.axhline(ext, color=col, ls=":", lw=0.8, alpha=0.5)
            ax.annotate(f"  min=${ext:.0f}  →  payoff=${payoff:.1f}",
                        xy=(t_ax[-1], path[-1]), fontsize=7.5, color=col)
        else:
            ext = path.max()
            payoff = ext - path[-1]
            ax.axhline(ext, color=col, ls=":", lw=0.8, alpha=0.5)
            ax.annotate(f"  max=${ext:.0f}  →  payoff=${payoff:.1f}",
                        xy=(t_ax[-1], path[-1]), fontsize=7.5, color=col)

    lbl = "min (call strike)" if option_type == "call" else "max (put strike)"
    ax.text(0.02, 0.04, f"Dotted lines = running {lbl}",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    ax.axhline(S, color="k", ls="--", lw=0.8, alpha=0.5, label=f"S₀=${S:.0f}")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.set(xlabel="Trading days",
           ylabel="Price ($)",
           title=f"Lookback {option_type.capitalize()} — 5 Sample Paths  "
                 f"[{ticker}  σ={sigma:.0%}  T={T*365:.0f}d]")
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    return _save(fig, out_dir, "lookback_paths.png")


# ── 4. Digital payoff ─────────────────────────────────────────────────────────

def plot_digital_payoff(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = "call",
                        cash: float = 1.0,
                        out_dir: str = ".",
                        ticker: str = "") -> str:
    """
    Payoff and price curves for cash-or-nothing and asset-or-nothing digitals.
    """
    from exotics.digital import price_cash_or_nothing, price_asset_or_nothing

    _apply()
    lo = K * 0.60
    hi = K * 1.45
    S_arr = np.linspace(lo, hi, 300)

    # Payoff at expiry
    if option_type == "call":
        payoff_con = cash * (S_arr > K).astype(float)
        payoff_aon = S_arr * (S_arr > K).astype(float)
    else:
        payoff_con = cash * (S_arr < K).astype(float)
        payoff_aon = S_arr * (S_arr < K).astype(float)

    # Price curves today
    con_prices = [price_cash_or_nothing(s, K, T, r, sigma, option_type, cash)["price"]
                  for s in S_arr]
    aon_prices = [price_asset_or_nothing(s, K, T, r, sigma, option_type)["price"]
                  for s in S_arr]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Payoff diagram
    ax = axes[0]
    ax.step(S_arr, payoff_con, color=_BLUE,   lw=2.2, where="post", label=f"Cash-or-Nothing (${cash:.0f})")
    ax.plot(S_arr, payoff_aon, color=_ORANGE, lw=1.8, ls="--",      label="Asset-or-Nothing (S_T)")
    ax.axvline(K, color=_RED,    ls="--", lw=1.5, label=f"Strike K=${K:,.0f}")
    ax.axvline(S, color=_BLUE,   ls=":",  lw=1.2, alpha=0.7, label=f"Spot S=${S:,.2f}")
    ax.set(xlabel="Spot at expiry ($)", ylabel="Payoff",
           title=f"Digital Payoff at Expiry  [{option_type.capitalize()}]")
    ax.legend(fontsize=9)

    # Price curves
    ax2 = axes[1]
    ax2.plot(S_arr, con_prices, color=_BLUE,   lw=2.2, label=f"Cash-or-Nothing price")
    ax2.plot(S_arr, aon_prices, color=_ORANGE, lw=1.8, ls="--", label="Asset-or-Nothing price")
    ax2.axvline(K, color=_RED,   ls="--", lw=1.5, label=f"K=${K:,.0f}")
    ax2.axvline(S, color=_BLUE,  ls=":",  lw=1.2, alpha=0.7, label=f"S=${S:,.2f}")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y:.3f}"))
    ax2.set(xlabel="Spot today ($)", ylabel="Price ($)",
            title=f"Digital Price vs Spot  [T={T*365:.0f}d  σ={sigma:.0%}]")
    ax2.legend(fontsize=9)

    plt.suptitle(f"Digital Options  [{ticker}  K=${K:,.0f}]", fontsize=12, y=1.02)
    plt.tight_layout()
    return _save(fig, out_dir, "digital_payoff.png")


# ── 5. Exotic comparison spider ───────────────────────────────────────────────

def plot_exotic_comparison(S: float, K: float, T: float, r: float, sigma: float,
                           H_down: float, H_up: float,
                           option_type: str = "call",
                           out_dir: str = ".",
                           ticker: str = "") -> str:
    """
    Price of multiple exotic types vs vanilla, plotted across a spot range.
    Useful for seeing how each exotic 'discounts' the vanilla price.
    """
    from exotics.barrier import price_barrier
    from exotics.asian   import price_asian_geo
    from exotics.digital import price_cash_or_nothing

    _apply()
    lo    = min(K * 0.75, H_down * 1.02, S * 0.78)
    hi    = max(K * 1.30, H_up   * 0.98, S * 1.25)
    S_arr = np.linspace(lo, hi, 150)

    results = {"Vanilla": [], "Down-Out": [], "Up-Out": [],
               "Geo Asian": [], "Cash-or-Nothing × K": []}

    for s in S_arr:
        from options.black_scholes import price as bs_price
        results["Vanilla"].append(bs_price(s, K, T, r, sigma, option_type))

        try:
            results["Down-Out"].append(
                price_barrier(s, K, T, r, sigma, H_down, option_type, "down-out")["price"])
        except Exception:
            results["Down-Out"].append(np.nan)

        try:
            results["Up-Out"].append(
                price_barrier(s, K, T, r, sigma, H_up, option_type, "up-out")["price"])
        except Exception:
            results["Up-Out"].append(np.nan)

        results["Geo Asian"].append(
            price_asian_geo(s, K, T, r, sigma, option_type)["price"])

        results["Cash-or-Nothing × K"].append(
            price_cash_or_nothing(s, K, T, r, sigma, option_type, cash=K)["price"])

    colors = [_GRAY, _BLUE, _ORANGE, _GREEN, _PURPLE]
    lws    = [2.0, 2.0, 2.0, 1.8, 1.8]
    lss    = ["--", "-", "-", "-", "-."]

    fig, ax = plt.subplots(figsize=(11, 5))
    for (lbl, vals), col, lw, ls in zip(results.items(), colors, lws, lss):
        ax.plot(S_arr, vals, color=col, lw=lw, ls=ls, label=lbl)

    ax.axvline(K,     color=_ORANGE, ls=":", lw=1.3, alpha=0.7, label=f"K=${K:,.0f}")
    ax.axvline(H_down,color=_BLUE,   ls=":", lw=1,   alpha=0.5, label=f"H↓=${H_down:,.0f}")
    ax.axvline(H_up,  color=_RED,    ls=":", lw=1,   alpha=0.5, label=f"H↑=${H_up:,.0f}")
    ax.axvline(S,     color="k",     ls=":", lw=0.8, alpha=0.4)

    ax.set(xlabel="Spot today ($)", ylabel="Option price ($)",
           title=f"Exotic vs Vanilla Price  [{ticker}  {option_type.capitalize()}"
                 f"  K=${K:,.0f}  T={T*365:.0f}d  σ={sigma:.0%}]")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y:,.3f}"))
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    return _save(fig, out_dir, "exotic_comparison.png")
