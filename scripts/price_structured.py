"""
scripts/price_structured.py
Structured products CLI: CDO, autocall, MBS.

Usage
-----
# CDO: Gaussian copula tranche pricing
python scripts/price_structured.py cdo --pd 0.02 --rho 0.20 --recovery 0.40 --maturity 5

# Autocallable note (live spot from Yahoo Finance)
python scripts/price_structured.py autocall AAPL --coupon 0.08 --ki-barrier 0.70

# MBS: cash flow analysis + WAL
python scripts/price_structured.py mbs --face 1000000 --wac 6.5 --wam 360 --psa 100

Output: output/<PRODUCT_YYYYMMDD_HHMMSS>_structured/
  cdo_structure.png       — loss distribution + tranche spread bars
  cdo_sensitivity.png     — fair spreads vs PD
  autocall.png            — payoff, call probabilities, distribution
  mbs_cashflows.png       — cash flow waterfall + balance
  mbs_wal_sensitivity.png — WAL vs PSA speed
  results.txt             — full text summary
"""

import argparse
import io
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yfinance as yf

from rates import DiscountCurve
from structured.cdo      import (cdo_structure, loss_distribution,
                                   expected_tranche_loss)
from structured.autocall import price_autocall
from structured.mbs      import (mbs_cashflows, weighted_average_life,
                                   mbs_price, mbs_yield, oas, mbs_summary)
from structured.charts   import (plot_cdo_structure, plot_cdo_sensitivity,
                                   plot_autocall, plot_mbs_cashflows,
                                   plot_mbs_wal_sensitivity)


# ── helpers ───────────────────────────────────────────────────────────────────

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
    out = os.path.join("output", f"{name}_{ts}_structured")
    os.makedirs(out, exist_ok=True)
    return out


def _flat_dc(r: float, max_t: float = 15.0) -> DiscountCurve:
    return DiscountCurve.flat(r, max_t)


def _fetch_vol_and_rate(ticker: str):
    """Return (sigma, spot, risk_free_rate)."""
    import yfinance as yf, numpy as np
    tk     = yf.Ticker(ticker)
    hist   = tk.history(period="1y", auto_adjust=True)
    close  = hist["Close"]
    sigma  = float(np.log(close / close.shift(1)).std() * np.sqrt(252))
    spot   = float(close.iloc[-1])
    try:
        tnx    = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1]
        r      = float(tnx) / 100.0
    except Exception:
        r = 0.045
    return sigma, spot, r


# ── subcommands ───────────────────────────────────────────────────────────────

def cmd_cdo(args, out_dir: str):
    dc = _flat_dc(args.rate)
    aps = [0.0, 0.03, 0.07, 0.12, 0.22, 1.0]

    print(f"\nCDO TRANCHE PRICING  (Gaussian Copula LHP)")
    print(f"  PD (1y)   = {args.pd*100:.2f}%")
    print(f"  ρ         = {args.rho*100:.0f}%")
    print(f"  Recovery  = {args.recovery*100:.0f}%")
    print(f"  Maturity  = {args.maturity}y")
    print(f"  Rate      = {args.rate*100:.2f}%")
    print()

    results = cdo_structure(aps, args.pd, args.rho, dc,
                             args.recovery, args.maturity)

    print(f"  {'Tranche':<14}  {'[A,D]':<12}  {'ETL':>8}  {'Spread':>12}")
    print("  " + "-" * 52)
    for r in results:
        A, D = r["attachment"], r["detachment"]
        print(f"  {r['name']:<14}  [{A*100:.0f}%–{D*100:.0f}%]{'':<4}  "
              f"{r['etl_at_maturity']*100:>6.2f}%  "
              f"{r['fair_spread_bps']:>8.1f} bps")

    # Loss distribution
    L, dens = loss_distribution(args.pd, args.rho, args.recovery)

    print("\nCHARTS")
    f1 = plot_cdo_structure(results, L, dens, args.pd, args.rho, out_dir, "CDO")
    print(f"  {f1}")
    f2 = plot_cdo_sensitivity(aps, np.linspace(0.005, 0.10, 20),
                               args.rho, dc, args.recovery, args.maturity,
                               out_dir, "CDO")
    print(f"  {f2}")


def cmd_autocall(args, out_dir: str):
    ticker = args.ticker.upper()
    print(f"\nFetching live data for {ticker}...")
    sigma, spot, r = _fetch_vol_and_rate(ticker)
    if args.vol:   sigma = args.vol
    if args.rate:  r     = args.rate
    if args.spot:  spot  = args.spot

    obs_dates = [round((i + 1) * args.maturity / (args.obs * args.maturity), 6)
                 for i in range(int(args.obs * args.maturity))]

    print(f"\nAUTOCALLABLE NOTE  — {ticker}")
    print(f"  Spot       = ${spot:.2f}")
    print(f"  Sigma      = {sigma:.2%}")
    print(f"  r          = {r:.2%}")
    print(f"  Maturity   = {args.maturity}y")
    print(f"  Autocall   = {args.autocall_level*100:.0f}% of initial")
    print(f"  KI barrier = {args.ki_barrier*100:.0f}% of initial")
    print(f"  Coupon     = {args.coupon*100:.1f}% p.a.")
    print(f"  Obs dates  : {obs_dates}")
    print()

    result = price_autocall(
        S=spot, r=r, sigma=sigma, T=args.maturity,
        obs_dates=obs_dates,
        autocall_level=args.autocall_level,
        ki_barrier=args.ki_barrier,
        coupon_rate=args.coupon,
        n_sims=50_000, seed=42,
    )

    print(f"  Price      = {result.price*100:.2f}% of notional  "
          f"(${result.price_notional:.2f})")
    print(f"  E[life]    = {result.expected_life:.2f}y")
    print(f"  P(KI loss) = {result.prob_ki_loss:.2%}")
    print(f"\n  Call probability by date:")
    for label, p in result.call_date_dist.items():
        print(f"    {label}: {p:.2%}")

    print("\nCHARTS")
    f1 = plot_autocall(result, spot, args.ki_barrier, args.autocall_level,
                        obs_dates, out_dir, ticker)
    print(f"  {f1}")


def cmd_mbs(args, out_dir: str):
    dc = _flat_dc(args.rate)

    print(f"\nMBS ANALYSIS  (PSA {args.psa}%)")
    print(f"  Face      = ${args.face:,.0f}")
    print(f"  WAC       = {args.wac:.2f}%")
    print(f"  WAM       = {args.wam} months ({args.wam/12:.1f} years)")
    print(f"  PSA speed = {args.psa}%")
    print()

    wac = args.wac / 100.0
    cf_df = mbs_cashflows(args.face, wac, args.wam, args.psa)

    wal   = weighted_average_life(cf_df)
    price = mbs_price(cf_df, wac)        # priced at WAC (should be ~100)
    yield_ = mbs_yield(100.0, cf_df)     # yield at par

    print(f"  WAL               = {wal:.2f} years")
    print(f"  Price @ WAC       = {price:.3f}")
    print(f"  Yield @ par       = {yield_*100:.4f}%")

    z = oas(100.0, cf_df, dc, args.face)
    print(f"  OAS (vs flat {args.rate*100:.1f}%) = {z:.1f} bps")
    print()
    print(mbs_summary(cf_df, yield_=wac, psa_speed=args.psa, discount_curve=dc))

    print("\nCHARTS")
    f1 = plot_mbs_cashflows(cf_df, args.psa, out_dir, "MBS")
    print(f"  {f1}")
    f2 = plot_mbs_wal_sensitivity(args.face, wac, args.wam,
                                   np.arange(50, 601, 50), out_dir, "MBS")
    print(f"  {f2}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="price_structured.py",
        description="Structured products: CDO, autocall, MBS",
    )
    sub = parser.add_subparsers(dest="cmd")

    # CDO
    p_cdo = sub.add_parser("cdo", help="CDO tranche pricing (Gaussian copula)")
    p_cdo.add_argument("--pd",       type=float, default=0.02)
    p_cdo.add_argument("--rho",      type=float, default=0.20)
    p_cdo.add_argument("--recovery", type=float, default=0.40)
    p_cdo.add_argument("--maturity", type=float, default=5.0)
    p_cdo.add_argument("--rate",     type=float, default=0.05)

    # Autocall
    p_ac = sub.add_parser("autocall", help="Autocallable structured note")
    p_ac.add_argument("ticker")
    p_ac.add_argument("--maturity",       type=float, default=1.0)
    p_ac.add_argument("--autocall-level", type=float, default=1.0,
                      dest="autocall_level")
    p_ac.add_argument("--ki-barrier",     type=float, default=0.70,
                      dest="ki_barrier")
    p_ac.add_argument("--coupon",         type=float, default=0.08)
    p_ac.add_argument("--obs",            type=int,   default=4,
                      help="Observation dates per year")
    p_ac.add_argument("--vol",            type=float, default=None)
    p_ac.add_argument("--spot",           type=float, default=None)
    p_ac.add_argument("--rate",           type=float, default=None)

    # MBS
    p_mbs = sub.add_parser("mbs", help="MBS / PSA prepayment model")
    p_mbs.add_argument("--face",   type=float, default=1_000_000.0)
    p_mbs.add_argument("--wac",    type=float, default=6.5,
                       help="Weighted average coupon (%%, e.g. 6.5)")
    p_mbs.add_argument("--wam",    type=int,   default=360)
    p_mbs.add_argument("--psa",    type=float, default=100.0)
    p_mbs.add_argument("--rate",   type=float, default=0.05)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    name    = args.cmd
    out_dir = _make_outdir(name)
    tee     = _Tee(); sys.stdout = tee

    print("=" * 60)
    print(f"  STRUCTURED PRODUCTS ENGINE — {name.upper()}")
    print("=" * 60)

    if   args.cmd == "cdo":      cmd_cdo(args, out_dir)
    elif args.cmd == "autocall": cmd_autocall(args, out_dir)
    elif args.cmd == "mbs":      cmd_mbs(args, out_dir)

    sys.stdout = tee._out
    with open(os.path.join(out_dir, "results.txt"), "w") as fh:
        fh.write(tee.getvalue())
    print(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    main()
