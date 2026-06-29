"""
structured/charts.py
Academic charts for structured products (Okabe-Ito, DejaVu Serif).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

_BLUE   = "#0072B2"
_ORANGE = "#E69F00"
_GREEN  = "#009E73"
_RED    = "#D55E00"
_PURPLE = "#CC79A7"
_GRAY   = "#888888"

_TRANCHE_COLORS = [_RED, _ORANGE, _GREEN, _BLUE, _PURPLE]

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


def _bps_fmt(ax):
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f} bps"))


# ── 1. CDO: loss distribution + tranche attachment waterfall ─────────────────

def plot_cdo_structure(tranches: list[dict],
                        loss_L: np.ndarray, loss_density: np.ndarray,
                        pd_1y: float, rho: float,
                        out_dir: str = ".", ticker: str = "CDO") -> str:
    """
    Left panel : portfolio loss distribution (PDF) + tranche boundaries
    Right panel: fair spread per tranche (bar chart)
    Saves: cdo_structure.png
    """
    _apply()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{ticker}  — Gaussian Copula CDO  "
        f"(PD={pd_1y*100:.1f}%  ρ={rho*100:.0f}%)",
        fontsize=12,
    )

    # Left: loss distribution
    ax = axes[0]
    ax.plot(loss_L * 100, loss_density, color=_BLUE, lw=1.5, label="Loss PDF")
    ax.set(xlabel="Portfolio Loss (%)", ylabel="Density",
           title="Portfolio Loss Distribution")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Shade tranches
    for i, tr in enumerate(tranches):
        A = tr["attachment"] * 100
        D = tr["detachment"] * 100
        color = _TRANCHE_COLORS[i % len(_TRANCHE_COLORS)]
        ax.axvspan(A, D, alpha=0.18, color=color, label=tr["name"])
        ax.axvline(A, color=color, ls="--", lw=0.8, alpha=0.7)

    ax.legend(loc="upper right", fontsize=8)

    # Right: fair spreads
    ax = axes[1]
    names   = [tr["name"]          for tr in tranches]
    spreads = [tr["fair_spread_bps"] for tr in tranches]
    colors  = [_TRANCHE_COLORS[i % len(_TRANCHE_COLORS)] for i in range(len(tranches))]
    bars    = ax.bar(names, spreads, color=colors, alpha=0.85, edgecolor="white")
    ax.set(title="Fair Tranche Spread", ylabel="Spread (bps)")
    _bps_fmt(ax)

    for bar, s in zip(bars, spreads):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(spreads) * 0.01,
                f"{s:.0f}", ha="center", va="bottom", fontsize=9)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    plt.tight_layout()

    fname = "cdo_structure.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname


def plot_cdo_sensitivity(attachment_points: list[float],
                          pd_range: np.ndarray,
                          rho: float,
                          discount_curve,
                          recovery: float,
                          maturity: float,
                          out_dir: str = ".", ticker: str = "CDO") -> str:
    """
    Fair spread vs PD for each tranche.
    Saves: cdo_sensitivity.png
    """
    from structured.cdo import cdo_structure

    _apply()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set(title=f"{ticker} — Tranche Spreads vs PD  (ρ={rho*100:.0f}%)",
           xlabel="1-Year PD (%)", ylabel="Fair Spread (bps)")

    n_tranches = len(attachment_points) - 1
    tranche_names = []
    for i in range(n_tranches):
        A, D = attachment_points[i], attachment_points[i + 1]
        if i == 0:
            tranche_names.append("Equity")
        elif i == n_tranches - 1:
            tranche_names.append("Senior")
        else:
            tranche_names.append(f"Mezz {i}")

    for i, name in enumerate(tranche_names):
        spreads = []
        for pd_1y in pd_range:
            try:
                res = cdo_structure(attachment_points, pd_1y, rho, discount_curve,
                                     recovery, maturity)
                spreads.append(res[i]["fair_spread_bps"])
            except Exception:
                spreads.append(np.nan)
        ax.plot(pd_range * 100, spreads,
                color=_TRANCHE_COLORS[i % len(_TRANCHE_COLORS)],
                lw=1.8, label=name)

    ax.legend()
    _bps_fmt(ax)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.tight_layout()
    fname = "cdo_sensitivity.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname


# ── 2. Autocall: payoff diagram + call probability ───────────────────────────

def plot_autocall(result, S: float, ki_barrier: float, autocall_level: float,
                  obs_dates: list[float], out_dir: str = ".",
                  ticker: str = "TICKER") -> str:
    """
    3-panel autocall chart:
      Left  : payoff at maturity (vs S_T/S₀) + barriers
      Center: call probability by date
      Right : payoff distribution histogram
    Saves: autocall.png
    """
    _apply()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"{ticker} Autocallable Note  "
        f"(price={result.price*100:.2f}%  E[life]={result.expected_life:.2f}y  "
        f"P(KI)={result.prob_ki_loss:.1%})",
        fontsize=11,
    )

    notional = result.notional

    # Left: payoff at maturity
    ax = axes[0]
    x  = np.linspace(0, 2, 300)
    # At maturity (if not called): pay notional if S_T/S0 >= KI_barrier else notional*S_T/S0
    payoff_mat = np.where(x >= ki_barrier, notional, notional * x / 1.0)
    ax.plot(x, payoff_mat / notional * 100, color=_BLUE, lw=2, label="Maturity payoff")
    ax.axvline(ki_barrier,     color=_RED,    ls="--", lw=1.5,
               label=f"KI barrier = {ki_barrier*100:.0f}%")
    ax.axvline(autocall_level, color=_GREEN,  ls="--", lw=1.5,
               label=f"Autocall = {autocall_level*100:.0f}%")
    ax.axhline(100, color=_GRAY, ls=":", lw=1)
    ax.set(xlabel="S_T / S₀", ylabel="Payoff (% of notional)",
           title="Payoff at Maturity (if not called)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}×"))
    ax.legend(fontsize=8)

    # Center: call probabilities
    ax = axes[1]
    pc = result.prob_call
    labels = [f"t={t:.2f}" for t in obs_dates]
    colors = [_GREEN if p > 0.05 else _GRAY for p in pc]
    bars   = ax.bar(labels, pc * 100, color=colors, alpha=0.85, edgecolor="white")
    ax.set(xlabel="Observation date", ylabel="Probability (%)",
           title="Call Probability by Date")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    # Annotate survival
    ax.text(0.97, 0.97, f"P(no call): {result.prob_no_call:.1%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=_RED)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Right: payoff histogram
    ax = axes[2]
    payoffs_pct = result.payoff_hist / notional * 100
    ax.hist(payoffs_pct, bins=60, color=_BLUE, alpha=0.75, edgecolor="white",
            density=True)
    ax.axvline(100, color=_GRAY, ls=":", lw=1)
    ax.axvline(np.mean(payoffs_pct), color=_RED, ls="--", lw=1.5,
               label=f"Mean = {np.mean(payoffs_pct):.1f}%")
    ax.set(xlabel="Discounted Payoff (% of notional)", ylabel="Density",
           title="Payoff Distribution")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=8)

    plt.tight_layout()
    fname = "autocall.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname


# ── 3. MBS: cash flow waterfall + WAL vs PSA ─────────────────────────────────

def plot_mbs_cashflows(cf_df: pd.DataFrame, psa_speed: float = 100.0,
                        out_dir: str = ".", ticker: str = "MBS") -> str:
    """
    Top panel  : stacked bar — interest vs principal vs prepayment
    Bottom panel: outstanding balance
    Saves: mbs_cashflows.png
    """
    _apply()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"{ticker} — MBS Cash Flows  (PSA {psa_speed:.0f}%)", fontsize=12)

    months   = cf_df["month"].values
    net_int  = cf_df["net_interest"].values
    sched_p  = cf_df["sched_principal"].values
    prepay   = cf_df["prepayment"].values
    face     = float(cf_df["balance_beg"].iloc[0])

    # Scale to % of face
    ax = axes[0]
    ax.bar(months, net_int   / face * 100, color=_BLUE,   alpha=0.85, label="Net Interest")
    ax.bar(months, sched_p   / face * 100, color=_GREEN,  alpha=0.85, label="Sched. Principal",
           bottom=net_int / face * 100)
    ax.bar(months, prepay    / face * 100, color=_ORANGE, alpha=0.85, label="Prepayment",
           bottom=(net_int + sched_p) / face * 100)
    ax.set(ylabel="% of Original Face", title="Monthly Cash Flows")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.2f}%"))
    ax.legend(loc="upper right")

    # Outstanding balance
    ax = axes[1]
    ax.fill_between(months, cf_df["balance_end"].values / face * 100,
                    color=_BLUE, alpha=0.30)
    ax.plot(months, cf_df["balance_end"].values / face * 100, color=_BLUE, lw=1.5)
    ax.set(xlabel="Month", ylabel="Balance (% of Face)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    plt.tight_layout()
    fname = "mbs_cashflows.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname


def plot_mbs_wal_sensitivity(face: float, wac: float, wam: int,
                              psa_range: np.ndarray,
                              out_dir: str = ".", ticker: str = "MBS") -> str:
    """
    WAL vs PSA speed.
    Saves: mbs_wal_sensitivity.png
    """
    from structured.mbs import mbs_cashflows, weighted_average_life

    _apply()
    wals = []
    for psa in psa_range:
        cf_df = mbs_cashflows(face, wac, wam, psa)
        wals.append(weighted_average_life(cf_df))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(psa_range, wals, color=_BLUE, lw=2)
    ax.axhline(wam / 12, color=_GRAY, ls=":", lw=1, label=f"WAM = {wam/12:.0f}y (0 PSA)")
    ax.set(title=f"{ticker} — WAL Sensitivity to PSA Speed",
           xlabel="PSA Speed (%)", ylabel="WAL (years)")
    ax.legend()
    plt.tight_layout()

    fname = "mbs_wal_sensitivity.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname
