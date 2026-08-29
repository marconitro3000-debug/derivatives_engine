"""
scripts/price_credit.py
Credit derivatives pricing CLI.

Usage
-----
# CDS: par spread + MtM at contract spread
python scripts/price_credit.py AAPL cds --spread 150 --maturity 5

# Bootstrap credit curve from market CDS quotes and plot it
python scripts/price_credit.py AAPL curve --quotes 1:50 2:80 3:100 5:130

# Risky bond: price, YTM, Z-spread, asset swap
python scripts/price_credit.py AAPL bond --coupon 0.04 --maturity 5 --spread 150

# CVA on a vanilla call option (counterparty credit risk)
python scripts/price_credit.py AAPL cva --strike 185 --expiry 1 --vol 0.28 --spread 200

Output: output/<TICKER_YYYYMMDD_HHMMSS>/  (PNG charts + results.txt)
"""

import argparse
import os
import sys
import io
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yfinance as yf
import numpy as np

from rates import DiscountCurve
from credit.hazard_rate import HazardCurve
from credit.cds    import cds_value, par_spread
from credit.bond   import risky_bond_price, z_spread, asset_swap_spread, bond_summary
from credit.cva    import cva_option
from credit.charts import (plot_survival_curve, plot_cds_legs,
                            plot_cva_profile, plot_hazard_structure)


# ── helpers ───────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self):
        self._buf = io.StringIO()
        self._out = sys.stdout

    def write(self, msg):
        self._out.write(msg)
        self._buf.write(msg)

    def flush(self):
        self._out.flush()

    def getvalue(self):
        return self._buf.getvalue()


def _make_outdir(ticker: str) -> str:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join("output", f"{ticker}_{ts}_credit")
    os.makedirs(out, exist_ok=True)
    return out


def _fetch_spot_rate(ticker: str) -> tuple[float, float]:
    """Return (spot, risk_free_rate). Falls back gracefully."""
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="5d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else 100.0
    except Exception:
        spot = 100.0

    try:
        tnx  = yf.Ticker("^TNX")
        rate = float(tnx.history(period="5d")["Close"].iloc[-1]) / 100
    except Exception:
        rate = 0.05

    return spot, rate


def _flat_curve(rate: float) -> DiscountCurve:
    return DiscountCurve.flat(rate, max_maturity=30.0)


def _hazard_from_spread(spread_bps: float, recovery: float) -> HazardCurve:
    return HazardCurve.from_spread(spread_bps, recovery, max_tenor=30.0)


def _print_header(ticker, product, spot, rate):
    print("=" * 60)
    print(f"  CREDIT ENGINE — {ticker}  ({product.upper()})")
    print(f"  Spot: ${spot:.2f}   Risk-free rate: {rate:.2%}")
    print("=" * 60)


# ── sub-commands ──────────────────────────────────────────────────────────────

def cmd_cds(args):
    spot, rate = _fetch_spot_rate(args.ticker)
    dc         = _flat_curve(rate)
    hc         = _hazard_from_spread(args.spread, args.recovery)
    out_dir    = _make_outdir(args.ticker)

    tee = _Tee(); sys.stdout = tee
    _print_header(args.ticker, "cds", spot, rate)

    T  = args.maturity
    R  = args.recovery
    N  = args.notional
    s_par = par_spread(hc, dc, T, R) * 10_000

    print(f"\nCREDIT CURVE  (flat hazard from {args.spread:.0f}bps spread)")
    print(hc.summary(max_tenor=T + 1))

    print(f"\nCDS PRICING   T={T:.1f}Y  R={R:.0%}  notional={N:,.0f}")
    print(f"  Par spread:       {s_par:.2f} bps")

    if args.spread:
        res = cds_value(hc, dc, T, args.spread / 10_000, R, "buyer", N)
        print(f"  Contract spread:  {args.spread:.0f} bps")
        print(f"  Protection leg:   ${res['protection_leg']:>14,.4f}")
        print(f"  Premium leg:      ${res['premium_leg']:>14,.4f}")
        print(f"  MtM (buyer):      ${res['value']:>14,.4f}")
        print(f"  RPV01:            ${res['rpv01']:>14,.4f}")
        print(f"  CS01 (1bp):       ${res['cs01']:>14,.4f}")

    print("\nCHARTS")
    f1 = plot_survival_curve(hc, out_dir, args.ticker, max_tenor=T + 2)
    f2 = plot_cds_legs(hc, dc, T, args.spread, R, out_dir, args.ticker)
    f3 = plot_hazard_structure(hc, dc, T, out_dir, args.ticker)
    for f in [f1, f2, f3]:
        print(f"  {f}")

    sys.stdout = tee._out
    with open(os.path.join(out_dir, "results.txt"), "w") as fh:
        fh.write(tee.getvalue())
    print(f"\nOutput: {out_dir}/")


def cmd_curve(args):
    """Bootstrap a credit curve from CDS quotes."""
    spot, rate = _fetch_spot_rate(args.ticker)
    dc         = _flat_curve(rate)
    out_dir    = _make_outdir(args.ticker)

    # Parse --quotes T1:S1 T2:S2 ...
    tenors, spreads = [], []
    for q in args.quotes:
        t_str, s_str = q.split(":")
        tenors.append(float(t_str))
        spreads.append(float(s_str))

    tee = _Tee(); sys.stdout = tee
    _print_header(args.ticker, "curve", spot, rate)

    print("\nMARKET CDS QUOTES")
    for t, s in zip(tenors, spreads):
        print(f"  T={t:.1f}Y   {s:.1f} bps")

    print("\nBOOTSTRAPPING...")
    hc = HazardCurve.bootstrap(tenors, spreads, dc, args.recovery)
    print(hc.summary())

    print("\nVERIFICATION (model par spread vs market)")
    for t, s_mkt in zip(tenors, spreads):
        s_mod = par_spread(hc, dc, t, args.recovery) * 10_000
        print(f"  T={t:.1f}Y  market={s_mkt:.1f}bps  model={s_mod:.2f}bps  "
              f"error={abs(s_mod - s_mkt):.4f}bps")

    max_t = max(tenors) + 2
    print("\nCHARTS")
    f1 = plot_survival_curve(hc, out_dir, args.ticker, max_tenor=max_t)
    f2 = plot_hazard_structure(hc, dc, tenors[-1], out_dir, args.ticker)
    for f in [f1, f2]:
        print(f"  {f}")

    sys.stdout = tee._out
    with open(os.path.join(out_dir, "results.txt"), "w") as fh:
        fh.write(tee.getvalue())
    print(f"\nOutput: {out_dir}/")


def cmd_bond(args):
    spot, rate = _fetch_spot_rate(args.ticker)
    dc         = _flat_curve(rate)
    hc         = _hazard_from_spread(args.spread, args.recovery)
    out_dir    = _make_outdir(args.ticker)

    tee = _Tee(); sys.stdout = tee
    _print_header(args.ticker, "bond", spot, rate)

    print(f"\nBOND ANALYSIS  face={args.face:,.0f}  coupon={args.coupon:.2%}  "
          f"T={args.maturity:.1f}Y  spread={args.spread:.0f}bps")

    print("\n" + bond_summary(args.face, args.coupon, args.maturity,
                               hc, dc, pay_freq=2, recovery=args.recovery))

    # Sensitivity: price vs spread
    print("\nPRICE vs CREDIT SPREAD  (bps → price % of par)")
    for s in [50, 100, 150, 200, 300, 500]:
        hc_s = _hazard_from_spread(s, args.recovery)
        res  = risky_bond_price(args.face, args.coupon, args.maturity,
                                hc_s, dc, pay_freq=2)
        print(f"  {s:>4}bps  →  {res['price_pct']:.4f}%  YTM={res['yield_to_maturity']:.3%}")

    print("\nCHARTS")
    f1 = plot_survival_curve(hc, out_dir, args.ticker, max_tenor=args.maturity + 2)
    f2 = plot_hazard_structure(hc, dc, args.maturity, out_dir, args.ticker)
    for f in [f1, f2]:
        print(f"  {f}")

    sys.stdout = tee._out
    with open(os.path.join(out_dir, "results.txt"), "w") as fh:
        fh.write(tee.getvalue())
    print(f"\nOutput: {out_dir}/")


def cmd_cva(args):
    spot, rate = _fetch_spot_rate(args.ticker)
    if args.spot:
        spot = args.spot
    if args.rate:
        rate = args.rate
    vol    = args.vol
    dc     = _flat_curve(rate)
    hc     = _hazard_from_spread(args.spread, args.recovery)
    out_dir = _make_outdir(args.ticker)

    tee = _Tee(); sys.stdout = tee
    _print_header(args.ticker, "cva", spot, rate)

    print(f"\nOPTION CVA  type={args.type}  S={spot:.2f}  K={args.strike:.2f}  "
          f"T={args.expiry:.2f}Y  σ={vol:.2%}  r={rate:.2%}")
    print(f"  Counterparty spread: {args.spread:.0f}bps  "
          f"R={args.recovery:.0%}")

    res = cva_option(spot, args.strike, args.expiry, rate, vol,
                     hc, dc, option_type=args.type,
                     recovery=args.recovery)

    print(f"\nRESULTS")
    print(f"  Vanilla {args.type}:       ${res['vanilla_price']:.4f}")
    print(f"  CVA:                   ${res['cva']:.4f}  "
          f"({res['cva_pct_of_vanilla']:.2f}% of vanilla)")
    print(f"  CVA-adjusted price:    ${res['cva_adjusted_price']:.4f}")
    print(f"  EE at T/4:             ${res['exposure_profile'][len(res['exposure_profile'])//4]:.4f}")
    print(f"  EE at T/2:             ${res['exposure_profile'][len(res['exposure_profile'])//2]:.4f}")

    print("\nCHARTS")
    f1 = plot_survival_curve(hc, out_dir, args.ticker, max_tenor=args.expiry + 1)
    f2 = plot_cva_profile(res, out_dir, args.ticker)
    for f in [f1, f2]:
        print(f"  {f}")

    sys.stdout = tee._out
    with open(os.path.join(out_dir, "results.txt"), "w") as fh:
        fh.write(tee.getvalue())
    print(f"\nOutput: {out_dir}/")


# ── argument parser ───────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="price_credit.py",
        description="Credit derivatives pricing engine",
    )
    p.add_argument("ticker", type=str, help="Underlying ticker (e.g. AAPL)")
    sub = p.add_subparsers(dest="product", required=True)

    # ── cds ──
    p_cds = sub.add_parser("cds", help="Price a CDS")
    p_cds.add_argument("--spread",   "-s", type=float, default=100.0,
                       help="Contract CDS spread (bps, default 100)")
    p_cds.add_argument("--maturity", "-T", type=float, default=5.0,
                       help="CDS maturity in years (default 5)")
    p_cds.add_argument("--recovery", "-R", type=float, default=0.40,
                       help="Recovery rate (default 0.40)")
    p_cds.add_argument("--notional", "-N", type=float, default=1_000_000,
                       help="Notional (default 1 000 000)")

    # ── curve ──
    p_curve = sub.add_parser("curve", help="Bootstrap credit curve from CDS quotes")
    p_curve.add_argument("--quotes", "-q", nargs="+", required=True,
                         metavar="T:S",
                         help='CDS quotes as "tenor:spread_bps", e.g. 1:50 5:130')
    p_curve.add_argument("--recovery", "-R", type=float, default=0.40)

    # ── bond ──
    p_bond = sub.add_parser("bond", help="Price a risky bond")
    p_bond.add_argument("--coupon",   "-c", type=float, default=0.05,
                        help="Annual coupon rate (default 0.05 = 5%)")
    p_bond.add_argument("--maturity", "-T", type=float, default=5.0)
    p_bond.add_argument("--face",     "-F", type=float, default=1000.0,
                        help="Face value (default 1000)")
    p_bond.add_argument("--spread",   "-s", type=float, default=150.0,
                        help="Flat credit spread in bps (default 150)")
    p_bond.add_argument("--recovery", "-R", type=float, default=0.40)

    # ── cva ──
    p_cva = sub.add_parser("cva", help="Compute CVA on a vanilla option")
    p_cva.add_argument("--strike",   "-K", type=float, required=True)
    p_cva.add_argument("--expiry",   "-T", type=float, required=True,
                       help="Option expiry in years")
    p_cva.add_argument("--type",           type=str,   default="call",
                       choices=["call", "put"])
    p_cva.add_argument("--vol",      "-v", type=float, default=0.25,
                       help="Implied vol (decimal, e.g. 0.25 = 25%)")
    p_cva.add_argument("--spread",   "-s", type=float, default=200.0,
                       help="Counterparty CDS spread (bps, default 200)")
    p_cva.add_argument("--recovery", "-R", type=float, default=0.40)
    p_cva.add_argument("--spot",     "-S", type=float, default=None,
                       help="Override spot price")
    p_cva.add_argument("--rate",     "-r", type=float, default=None,
                       help="Override risk-free rate (decimal)")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {"cds": cmd_cds, "curve": cmd_curve,
                "bond": cmd_bond, "cva": cmd_cva}
    dispatch[args.product](args)


if __name__ == "__main__":
    main()
