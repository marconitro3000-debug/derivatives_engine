"""
scripts/analyze_rates_advanced.py
Advanced rates CLI: swaptions, caps/floors, SABR smile, Nelson-Siegel.

Usage
-----
# Swaption pricing + vol surface
python scripts/analyze_rates_advanced.py swaption --expiry 1 --tenor 5 --strike 0.05 --vol 0.20

# Interest rate cap / floor
python scripts/analyze_rates_advanced.py capfloor --maturity 3 --strike 0.05 --vol 0.20

# SABR smile calibration
python scripts/analyze_rates_advanced.py sabr --forward 0.05 --expiry 1 --beta 0.5

# Nelson-Siegel / Svensson curve fitting
python scripts/analyze_rates_advanced.py ns

Output: output/<cmd_YYYYMMDD_HHMMSS>_rates/
  swaption_surface.png  — vol surface expiry × tenor
  capfloor_schedule.png — caplet/floorlet breakdown
  sabr_smile.png        — SABR fitted smile vs market
  ns_fit.png            — NS/Svensson curve fit
  results.txt           — full text log
"""

import argparse
import io
import os
import sys
from datetime import datetime

# Force UTF-8 on Windows so Greek/subscript chars don't crash cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from rates import DiscountCurve
from rates.swaption import (forward_swap_rate, price_swaption_black,
                              price_swaption_bachelier)
from rates.capfloor import (cap, floor, cap_floor_parity_check,
                              forward_libor)
from rates.sabr import (SABRParams, implied_vol_sabr, implied_vol_grid,
                          calibrate as calibrate_sabr)
from rates.nelson_siegel import (fit_ns, fit_svensson, ns_yield,
                                  svensson_yield, fit_summary)

_BLUE   = "#0072B2"
_ORANGE = "#E69F00"
_GREEN  = "#009E73"
_RED    = "#D55E00"
_GRAY   = "#888888"

STYLE = {
    "font.family": "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
}


class _Tee:
    def __init__(self):
        self._buf = io.StringIO()
        self._out = sys.stdout
    def write(self, msg):
        self._out.write(msg); self._buf.write(msg)
    def flush(self):
        self._out.flush()
    def getvalue(self):
        return self._buf.getvalue()


def _make_outdir(name: str) -> str:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join("output", f"{name}_{ts}_rates")
    os.makedirs(out, exist_ok=True)
    return out


def _build_curve(rate: float = 0.05) -> DiscountCurve:
    return DiscountCurve.flat(rate, 30.0)


# ── swaption ──────────────────────────────────────────────────────────────────

def cmd_swaption(args, out_dir: str):
    dc = _build_curve(args.rate)

    print(f"\nSWAPTION PRICING  (Black's model)")
    print(f"  Rate curve : flat {args.rate*100:.2f}%")
    print(f"  Expiry     : {args.expiry}y")
    print(f"  Tenor      : {args.tenor}y")
    print(f"  Strike     : {args.strike*100:.2f}%")
    print(f"  Black vol  : {args.vol*100:.1f}%")
    print()

    F, A = forward_swap_rate(dc, args.expiry, args.tenor)
    print(f"  Forward swap rate : {F*100:.4f}%")
    print(f"  Annuity (PV01)    : {A:.6f}")
    print()

    for stype in ("payer", "receiver"):
        res = price_swaption_black(dc, args.expiry, args.tenor,
                                    args.strike, args.vol, stype)
        print(f"  {stype.capitalize():<9}: ${res['price']:>12,.2f}  "
              f"d1={res['d1']:.4f}  vega=${res['vega']:.0f}")

    # Bachelier comparison
    sig_n = args.vol * F   # rough normal vol conversion
    for stype in ("payer", "receiver"):
        res = price_swaption_bachelier(dc, args.expiry, args.tenor,
                                        args.strike, sig_n, stype)
        print(f"  Bachelier {stype:<9}: ${res['price']:>12,.2f}")

    # Vol surface
    print("\nSWAPTION VOL SURFACE")
    expiries = [0.25, 0.5, 1, 2, 3, 5]
    tenors   = [1, 2, 3, 5, 7, 10]
    print(f"  {'Expiry':<8}", end="")
    for t in tenors:
        print(f" {'T='+str(t)+'y':>8}", end="")
    print()
    for T_exp in expiries:
        print(f"  {T_exp:<8.2f}", end="")
        for tenor in tenors:
            px = price_swaption_black(dc, T_exp, tenor, args.strike, args.vol)
            print(f" {px['price']/1000:>7.1f}k", end="")
        print()

    # Chart: vol surface
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))
    for tenor in tenors:
        prices = [price_swaption_black(dc, T_exp, tenor, args.strike, args.vol)["price"] / 1000
                  for T_exp in expiries]
        ax.plot(expiries, prices, lw=1.8, marker="o", ms=4, label=f"Tenor {tenor}y")
    ax.set(xlabel="Option Expiry (y)", ylabel="Price ($k)", title="Swaption Prices vs Expiry")
    ax.legend(fontsize=8)
    fname = "swaption_surface.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  {fname}")


# ── cap / floor ───────────────────────────────────────────────────────────────

def cmd_capfloor(args, out_dir: str):
    dc = _build_curve(args.rate)

    print(f"\nCAP / FLOOR  (Black's model)")
    print(f"  Maturity : {args.maturity}y  Strike: {args.strike*100:.2f}%  Vol: {args.vol*100:.1f}%")
    print()

    cap_res   = cap(dc, args.maturity, args.strike, args.vol)
    floor_res = floor(dc, args.maturity, args.strike, args.vol)
    parity    = cap_floor_parity_check(dc, args.maturity, args.strike, args.vol)

    print(f"  Cap    : ${cap_res['price']:>12,.2f}  ({cap_res['n_caplets']} caplets)")
    print(f"  Floor  : ${floor_res['price']:>12,.2f}  ({floor_res['n_floorlets']} floorlets)")
    print(f"  Cap−Floor        = ${cap_res['price'] - floor_res['price']:>12,.2f}")
    print(f"  Swap PV (parity) = ${parity['swap_pv']:>12,.2f}")
    print(f"  Parity error     = ${parity['parity_error']:>8.2f}")
    print()
    print(f"  Caplet breakdown:")
    for i, (sch, cp, fr) in enumerate(zip(cap_res["schedule"],
                                           cap_res["caplet_prices"],
                                           cap_res["forward_rates"])):
        print(f"    [{sch[0]:.2f}–{sch[1]:.2f}]  Fwd={fr*100:.3f}%  Caplet=${cp:>8,.2f}")

    # Chart
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Cap/Floor  K={args.strike*100:.1f}%  σ={args.vol*100:.0f}%  T={args.maturity}y",
                 fontsize=11)

    # Caplet/Floorlet breakdown
    ax = axes[0]
    labels = [f"{s[0]:.2f}" for s in cap_res["schedule"]]
    ax.bar(labels, cap_res["caplet_prices"], color=_BLUE, alpha=0.8, label="Caplets")
    ax.bar(labels, floor_res["floorlet_prices"], color=_RED, alpha=0.5, label="Floorlets")
    ax.set(title="Caplet / Floorlet Prices", xlabel="Reset date", ylabel="$")
    ax.legend(); plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Forward curve
    ax = axes[1]
    ax.plot(labels, [f * 100 for f in cap_res["forward_rates"]],
            color=_BLUE, lw=2, marker="o", ms=4, label="Forward rate")
    ax.axhline(args.strike * 100, color=_RED, ls="--", lw=1.5,
               label=f"Strike = {args.strike*100:.1f}%")
    ax.set(title="Forward Rate Schedule", xlabel="Reset date", ylabel="%")
    ax.legend(); plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fname = "capfloor_schedule.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  {fname}")


# ── SABR ──────────────────────────────────────────────────────────────────────

def cmd_sabr(args, out_dir: str):
    F = args.forward
    T = args.expiry

    # Synthetic market smile
    true_params = SABRParams(alpha=args.alpha, beta=args.beta,
                              rho=args.rho, nu=args.nu)
    atm_strike  = F
    strikes_pct = np.array([-200, -150, -100, -50, -25, 0, 25, 50, 100, 150, 200])
    strikes     = F + strikes_pct / 10_000.0
    strikes     = strikes[strikes > 0]
    mkt_vols    = np.array([implied_vol_sabr(F, K, T, true_params) for K in strikes])
    # Add small noise
    rng = np.random.default_rng(42)
    noisy_vols = mkt_vols + rng.normal(0, 0.001, len(mkt_vols))

    print(f"\nSABR MODEL  (Hagan 2002)")
    print(f"  Forward  : {F*100:.4f}%")
    print(f"  Expiry   : {T}y")
    print(f"  True params: α={args.alpha:.4f}  β={args.beta:.2f}  ρ={args.rho:.2f}  ν={args.nu:.2f}")
    print()

    # Calibrate
    fitted = calibrate_sabr(F, T, strikes, noisy_vols, beta=args.beta)
    print(f"  Fitted params: α={fitted.alpha:.4f}  β={fitted.beta:.2f}  "
          f"ρ={fitted.rho:.4f}  ν={fitted.nu:.4f}")

    fitted_vols = np.array([implied_vol_sabr(F, K, T, fitted) for K in strikes])
    rmse = np.sqrt(np.mean((fitted_vols - noisy_vols) ** 2)) * 10_000
    print(f"  Fit RMSE: {rmse:.3f} bps")
    print()
    print(f"  {'Strike':>10}  {'Mkt Vol':>10}  {'SABR Vol':>10}  {'Error (bps)':>12}")
    for K, mv, fv in zip(strikes, noisy_vols, fitted_vols):
        print(f"  {K*100:>9.4f}%  {mv*100:>9.4f}%  {fv*100:>9.4f}%  {(fv-mv)*10000:>10.2f}")

    # Chart
    plt.rcParams.update(STYLE)
    K_dense = np.linspace(strikes[0] * 0.8, strikes[-1] * 1.2, 200)
    v_dense = implied_vol_grid(F, K_dense, T, fitted)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(K_dense * 100, v_dense * 100, color=_BLUE, lw=2, label="SABR fitted")
    ax.scatter(strikes * 100, noisy_vols * 100, color=_ORANGE, zorder=5,
               s=40, label="Market vols")
    ax.axvline(F * 100, color=_GRAY, ls=":", lw=1, label=f"ATM = {F*100:.2f}%")
    ax.set(title=f"SABR Smile  α={fitted.alpha:.4f}  β={fitted.beta:.2f}  "
                 f"ρ={fitted.rho:.2f}  ν={fitted.nu:.2f}",
           xlabel="Strike (%)", ylabel="Black Vol (%)")
    ax.legend()
    fname = "sabr_smile.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  {fname}")


# ── Nelson-Siegel ─────────────────────────────────────────────────────────────

def cmd_ns(args, out_dir: str):
    # US Treasury-style curve
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    zero_rates  = np.array([0.053, 0.054, 0.052, 0.049, 0.048,
                             0.047, 0.048, 0.049, 0.051, 0.052])

    print(f"\nNELSON-SIEGEL / SVENSSON CURVE FITTING")
    print(f"\n  Market zero rates:")
    for t, r in zip(maturities, zero_rates):
        print(f"    {t:>5.2f}y : {r*100:.4f}%")

    ns_fitted = fit_ns(maturities, zero_rates)
    sv_fitted = fit_svensson(maturities, zero_rates)

    print(f"\n{fit_summary(maturities, zero_rates, ns_fitted, 'NS')}")
    print(f"\n{fit_summary(maturities, zero_rates, sv_fitted, 'SV')}")

    # Chart
    plt.rcParams.update(STYLE)
    T_dense = np.linspace(0.1, 35, 500)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(maturities, zero_rates * 100, color="black", zorder=5,
               s=40, label="Market")
    ax.plot(T_dense, ns_yield(T_dense, ns_fitted) * 100, color=_BLUE, lw=2,
            label=f"Nelson-Siegel  β₀={ns_fitted.beta0*100:.2f}%")
    ax.plot(T_dense, svensson_yield(T_dense, sv_fitted) * 100, color=_ORANGE,
            lw=2, ls="--", label="Svensson")
    ax.set(title="Yield Curve Fitting: Nelson-Siegel vs Svensson",
           xlabel="Maturity (years)", ylabel="Zero Rate (%)")
    ax.legend()
    fname = "ns_fit.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  {fname}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="analyze_rates_advanced.py",
        description="Advanced rates: swaptions, caps/floors, SABR, Nelson-Siegel",
    )
    sub = parser.add_subparsers(dest="cmd")

    # swaption
    p = sub.add_parser("swaption")
    p.add_argument("--expiry",  type=float, default=1.0)
    p.add_argument("--tenor",   type=float, default=5.0)
    p.add_argument("--strike",  type=float, default=0.05)
    p.add_argument("--vol",     type=float, default=0.20)
    p.add_argument("--rate",    type=float, default=0.05)

    # capfloor
    p = sub.add_parser("capfloor")
    p.add_argument("--maturity", type=float, default=3.0)
    p.add_argument("--strike",   type=float, default=0.05)
    p.add_argument("--vol",      type=float, default=0.20)
    p.add_argument("--rate",     type=float, default=0.05)

    # sabr
    p = sub.add_parser("sabr")
    p.add_argument("--forward", type=float, default=0.05)
    p.add_argument("--expiry",  type=float, default=1.0)
    p.add_argument("--alpha",   type=float, default=0.04)
    p.add_argument("--beta",    type=float, default=0.50)
    p.add_argument("--rho",     type=float, default=-0.30)
    p.add_argument("--nu",      type=float, default=0.40)

    # nelson-siegel
    sub.add_parser("ns")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    out_dir = _make_outdir(args.cmd)
    tee = _Tee(); sys.stdout = tee

    print("=" * 60)
    print(f"  RATES ADVANCED — {args.cmd.upper()}")
    print("=" * 60)

    if   args.cmd == "swaption":  cmd_swaption(args, out_dir)
    elif args.cmd == "capfloor":  cmd_capfloor(args, out_dir)
    elif args.cmd == "sabr":      cmd_sabr(args, out_dir)
    elif args.cmd == "ns":        cmd_ns(args, out_dir)

    sys.stdout = tee._out
    with open(os.path.join(out_dir, "results.txt"), "w") as fh:
        fh.write(tee.getvalue())
    print(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    main()
