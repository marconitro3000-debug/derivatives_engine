"""
credit/charts.py
Academic charts for the credit module (Okabe-Ito palette, DejaVu Serif).
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

_BLUE   = "#0072B2"
_ORANGE = "#E69F00"
_GREEN  = "#009E73"
_RED    = "#D55E00"
_PURPLE = "#CC79A7"

ACADEMIC_STYLE = {
    "font.family":       "DejaVu Serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.30,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        110,
}


def _apply():
    plt.rcParams.update(ACADEMIC_STYLE)


# ── 1. Survival curve + credit spread term structure ─────────────────────────

def plot_survival_curve(hazard_curve, out_dir: str = ".",
                        ticker: str = "CREDIT",
                        max_tenor: float = 10.0) -> str:
    """
    Left panel : Q(τ>T) and PD(T) vs tenor
    Right panel: implied credit spread (bps) term structure
    Saves survival_curve.png
    """
    _apply()
    t     = np.linspace(0.01, max_tenor, 500)
    Q     = np.array([hazard_curve.survival_prob(ti) for ti in t])
    PD    = 1.0 - Q
    sprd  = np.array([hazard_curve.credit_spread(ti) * 10_000 for ti in t])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{ticker} — Credit Curve  (R = {hazard_curve.recovery:.0%})",
                 fontsize=12)

    ax = axes[0]
    ax.plot(t, Q  * 100, color=_BLUE,  lw=2, label="Q(τ>T) Survival")
    ax.fill_between(t, Q * 100, alpha=0.12, color=_BLUE)
    ax.plot(t, PD * 100, color=_RED,   lw=2, ls="--", label="PD(T) Default prob")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set(xlabel="Tenor (years)", ylabel="Probability (%)",
           title="Survival & Default Probability")
    ax.legend()

    ax = axes[1]
    ax.plot(t, sprd, color=_ORANGE, lw=2)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}bp"))
    ax.set(xlabel="Tenor (years)", ylabel="Credit Spread (bps)",
           title="Implied Credit Spread Term Structure")

    plt.tight_layout()
    fname = "survival_curve.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname


# ── 2. CDS leg decomposition + MtM vs spread ─────────────────────────────────

def plot_cds_legs(hazard_curve, discount_curve, maturity: float,
                  contract_spread_bps: float,
                  recovery: float = None,
                  out_dir: str = ".", ticker: str = "CREDIT") -> str:
    """
    Left panel : bar chart of protection vs premium leg at current spread
    Right panel: CDS MtM (buyer) across range of market spreads
    Saves cds_legs.png
    """
    from .cds import protection_leg, risky_pv01, cds_value, par_spread

    _apply()
    R       = recovery if recovery is not None else hazard_curve.recovery
    s_par   = par_spread(hazard_curve, discount_curve, maturity, R) * 10_000

    # MtM across spread range
    lo = max(1.0,      s_par * 0.2)
    hi = max(s_par * 3, 500.0)
    spreads = np.linspace(lo, hi, 200)
    values  = [
        cds_value(hazard_curve, discount_curve, maturity,
                  s / 10_000, R, "buyer", 1.0)["value"]
        for s in spreads
    ]

    res = cds_value(hazard_curve, discount_curve, maturity,
                    contract_spread_bps / 10_000, R, "buyer", 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{ticker} — CDS  T={maturity:.1f}Y  R={R:.0%}  "
                 f"par={s_par:.0f}bps  contract={contract_spread_bps:.0f}bps",
                 fontsize=11)

    ax = axes[0]
    bars = ax.bar(["Protection\nLeg", "Premium\nLeg"],
                  [res["protection_leg"], res["premium_leg"]],
                  color=[_BLUE, _ORANGE], alpha=0.85, width=0.4)
    ax.bar_label(bars, fmt="${:.4f}", padding=4, fontsize=9)
    ax.axhline(0, color="black", lw=0.7)
    net_c = _GREEN if res["value"] >= 0 else _RED
    ax.annotate(f'Net (buyer) = ${res["value"]:.4f}',
                xy=(0.5, 0.95), xycoords="axes fraction",
                ha="center", fontsize=10, color=net_c, fontweight="bold")
    ax.set(ylabel="PV (per unit notional)",
           title=f"CDS Legs at {contract_spread_bps:.0f}bps")

    ax = axes[1]
    vals = np.array(values)
    ax.plot(spreads, vals, color=_BLUE, lw=2)
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(contract_spread_bps, color=_ORANGE, ls="--", lw=1.5,
               label=f"Contract {contract_spread_bps:.0f}bps")
    ax.axvline(s_par, color=_RED, ls=":", lw=1.2,
               label=f"Par {s_par:.0f}bps")
    ax.fill_between(spreads, vals, 0,
                    where=vals > 0, alpha=0.12, color=_GREEN, label="Buyer profit")
    ax.fill_between(spreads, vals, 0,
                    where=vals < 0, alpha=0.12, color=_RED, label="Buyer loss")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}bp"))
    ax.set(xlabel="Market CDS Spread", ylabel="CDS Value (buyer, per unit)",
           title="CDS Mark-to-Market vs Market Spread")
    ax.legend()

    plt.tight_layout()
    fname = "cds_legs.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname


# ── 3. CVA profile (EE + contributions + cumulative) ─────────────────────────

def plot_cva_profile(cva_result: dict, out_dir: str = ".",
                     ticker: str = "CREDIT") -> str:
    """
    Three panels: EE profile | CVA contributions by period | cumulative CVA
    Saves cva_profile.png
    """
    _apply()
    times    = cva_result["exposure_times"]
    contribs = cva_result["contributions"]
    total    = cva_result["cva"]
    EE       = cva_result.get("exposure_profile")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{ticker} — CVA Profile  (total = ${total:.4f})", fontsize=12)

    # Panel 1: expected exposure
    ax = axes[0]
    if EE is not None:
        ax.plot(times, EE, color=_BLUE, lw=2)
        ax.fill_between(times, EE, alpha=0.15, color=_BLUE)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"${y:.2f}"))
        ax.set(xlabel="Time (years)", ylabel="Expected Exposure",
               title="Expected Positive Exposure (EE)")
    else:
        ax.set_visible(False)

    # Panel 2: per-period CVA contributions
    ax = axes[1]
    pct = contribs / total * 100 if abs(total) > 1e-12 else contribs
    widths = np.diff(np.concatenate([[0.0], times]))
    ax.bar(times, pct, width=widths, color=_RED, alpha=0.75, align="edge")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax.set(xlabel="Time (years)", ylabel="CVA contribution (%)",
           title="CVA Contribution by Period")

    # Panel 3: cumulative CVA
    ax = axes[2]
    cum = np.cumsum(contribs)
    cum_pct = cum / total * 100 if abs(total) > 1e-12 else cum
    ax.plot(times, cum_pct, color=_PURPLE, lw=2)
    ax.fill_between(times, cum_pct, alpha=0.15, color=_PURPLE)
    ax.axhline(100 if abs(total) > 1e-12 else 0,
               color="gray", ls="--", lw=1)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set(xlabel="Time (years)", ylabel="Cumulative CVA (%)",
           title="Cumulative CVA Build-up")

    plt.tight_layout()
    fname = "cva_profile.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname


# ── 4. Hazard rate structure + spread vs recovery ────────────────────────────

def plot_hazard_structure(hazard_curve, discount_curve, maturity: float,
                          out_dir: str = ".",
                          ticker: str = "CREDIT") -> str:
    """
    Left panel : step-function hazard rate h(t) in bps/yr
    Right panel: par CDS spread vs recovery rate at the given maturity
    Saves hazard_structure.png
    """
    from .cds import par_spread
    from .hazard_rate import HazardCurve

    _apply()
    max_t  = max(float(hazard_curve.tenors[-1]), maturity)
    t_fine = np.linspace(0.01, max_t, 500)
    h_fine = np.array([hazard_curve.hazard_rate_at(ti) * 10_000 for ti in t_fine])

    recoveries = np.linspace(0.0, 0.80, 120)
    spreads    = []
    for R in recoveries:
        hc = HazardCurve(hazard_curve.tenors, hazard_curve.hazard_rates, R)
        spreads.append(par_spread(hc, discount_curve, maturity, R) * 10_000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{ticker} — Hazard Rate Structure", fontsize=12)

    ax = axes[0]
    ax.step(t_fine, h_fine, color=_RED, lw=2, where="post")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}bp"))
    ax.set(xlabel="Tenor (years)", ylabel="Hazard Rate (bps/yr)",
           title="Piecewise-Constant Hazard Rate h(t)")

    ax = axes[1]
    ax.plot(recoveries * 100, spreads, color=_BLUE, lw=2)
    ax.axvline(hazard_curve.recovery * 100, color=_ORANGE, ls="--", lw=1.5,
               label=f"R = {hazard_curve.recovery:.0%}")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}bp"))
    ax.set(xlabel="Recovery Rate (%)", ylabel="Par CDS Spread (bps)",
           title=f"Par Spread vs Recovery  (T={maturity:.1f}Y)")
    ax.legend()

    plt.tight_layout()
    fname = "hazard_structure.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    return fname
