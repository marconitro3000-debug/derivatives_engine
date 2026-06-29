"""
scripts/analyze_fx.py
CLI for FX options analysis using Garman-Kohlhagen and smile tools.

Usage:
  python scripts/analyze_fx.py gk
  python scripts/analyze_fx.py smile
  python scripts/analyze_fx.py barrier
  python scripts/analyze_fx.py surface
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fx import (
    fx_forward, price_gk, implied_vol_gk, put_call_parity_check,
    delta_to_strike, strike_to_delta, atm_dns_strike, atm_forward_strike,
    FXSmileQuotes, build_smile, vanna_volga_price, price_fx_barrier_vv,
)


# ── GK pricing ────────────────────────────────────────────────────────────────

def cmd_gk(args):
    print("=" * 65)
    print("  GARMAN-KOHLHAGEN FX OPTION PRICER")
    print("=" * 65)

    params = dict(
        S=args.spot, K=args.strike, T=args.tenor,
        r_d=args.rd, r_f=args.rf, sigma=args.vol,
    )
    F = fx_forward(args.spot, args.rd, args.rf, args.tenor)

    print(f"\nSpot    : {args.spot:.4f}")
    print(f"Forward : {F:.4f}   (T={args.tenor}y, r_d={args.rd:.2%}, r_f={args.rf:.2%})")
    print(f"Strike  : {args.strike:.4f}")
    print(f"Vol     : {args.vol:.1%}")

    for otype in ("call", "put"):
        res = price_gk(**params, option_type=otype, notional=args.notional)
        print(f"\n  {'CALL' if otype=='call' else 'PUT '}")
        print(f"    Price   : {res['price']:>12,.2f}  ({args.notional/1e6:.0f}M notional)")
        print(f"    Unit px : {res['unit_px']:>12.6f}")
        print(f"    Delta   : {res['delta']:>12.4f}")
        print(f"    Gamma   : {res['gamma']:>12.6f}")
        print(f"    Vega    : {res['vega']:>12,.2f}  (per 1% vol)")
        print(f"    Theta   : {res['theta']:>12,.2f}  (per day)")
        print(f"    Vanna   : {res['vanna']:>12.6f}")
        print(f"    Volga   : {res['volga']:>12.6f}")

    pcp = put_call_parity_check(**params)
    print(f"\n  Put-Call Parity error: {pcp['error']:.2e}")

    # ATM conventions
    K_fwd = atm_forward_strike(args.spot, args.tenor, args.rd, args.rf)
    K_dns = atm_dns_strike(args.spot, args.tenor, args.rd, args.rf, args.vol)
    print(f"\n  ATM forward  : {K_fwd:.4f}")
    print(f"  ATM DNS      : {K_dns:.4f}")

    # Strike grid
    strikes = np.linspace(args.spot * 0.90, args.spot * 1.10, 9)
    print(f"\n  {'Strike':>8}  {'Delta':>8}  {'Call px':>12}  {'Put px':>12}")
    for K in strikes:
        c = price_gk(**{**params, 'K': K}, option_type="call", notional=1.0)
        p = price_gk(**{**params, 'K': K}, option_type="put",  notional=1.0)
        print(f"  {K:>8.4f}  {c['delta']:>8.4f}  {c['unit_px']:>12.6f}  {p['unit_px']:>12.6f}")


# ── FX smile ─────────────────────────────────────────────────────────────────

def cmd_smile(args):
    print("=" * 65)
    print("  FX VOLATILITY SMILE (25-delta / 10-delta)")
    print("=" * 65)

    q = FXSmileQuotes(
        S=args.spot, T=args.tenor, r_d=args.rd, r_f=args.rf,
        atm=args.atm, rr25=args.rr25, bf25=args.bf25,
        rr10=args.rr10, bf10=args.bf10,
    )

    smile = build_smile(q)

    print(f"\nSpot    : {args.spot:.4f}")
    F = fx_forward(args.spot, args.rd, args.rf, args.tenor)
    print(f"Forward : {F:.4f}   (T={args.tenor}y)")
    print(f"\nMarket quotes:")
    print(f"  ATM  vol : {args.atm:.2%}")
    print(f"  25-delta RR: {args.rr25:+.2%}  (25C - 25P)")
    print(f"  25-delta BF: {args.bf25:.2%}   (½(25C+25P) - ATM)")
    if args.rr10 or args.bf10:
        print(f"  10-delta RR: {args.rr10:+.2%}")
        print(f"  10-delta BF: {args.bf10:.2%}")

    print(f"\nReconstructed smile:")
    print(f"  {'Label':>6}  {'Strike':>10}  {'Vol':>8}")
    for label, K, v in zip(smile.labels, smile.strikes, smile.vols):
        print(f"  {label:>6}  {K:>10.4f}  {v:>8.2%}")

    # Vanna-Volga for ATM call
    K_atm  = atm_dns_strike(args.spot, args.tenor, args.rd, args.rf, args.atm)
    res_bs = price_gk(args.spot, K_atm, args.tenor, args.rd, args.rf, args.atm, "call", 1.0)
    vv_px  = vanna_volga_price(res_bs["unit_px"], res_bs["vanna"], res_bs["volga"], smile, "call")
    print(f"\n  ATM call BS price : {res_bs['unit_px']:.6f}")
    print(f"  ATM call VV price : {vv_px:.6f}  (smile adjusted)")


# ── Barrier option ────────────────────────────────────────────────────────────

def cmd_barrier(args):
    print("=" * 65)
    print("  FX BARRIER OPTION (Vanna-Volga)")
    print("=" * 65)

    q = FXSmileQuotes(
        S=args.spot, T=args.tenor, r_d=args.rd, r_f=args.rf,
        atm=args.atm, rr25=args.rr25, bf25=args.bf25,
    )
    smile = build_smile(q)

    F = fx_forward(args.spot, args.rd, args.rf, args.tenor)
    print(f"\nSpot    : {args.spot:.4f}")
    print(f"Forward : {F:.4f}   T={args.tenor}y")
    print(f"Strike  : {args.strike:.4f}")
    print(f"Barrier : {args.barrier:.4f}  ({args.btype})")
    print(f"Notional: {args.notional/1e6:.0f}M")

    res = price_fx_barrier_vv(
        args.spot, args.strike, args.barrier, args.tenor,
        args.rd, args.rf, smile,
        option_type="call", barrier_type=args.btype,
        notional=args.notional,
    )
    print(f"\n  BS price   : {res['price_bs']:>12,.2f}")
    print(f"  VV price   : {res['price']:>12,.2f}")
    print(f"  VV adj     : {res['vv_adj']:>12,.2f}")
    print(f"  Vol used   : {res['sigma_used']:.2%}")

    # Vanilla for comparison
    vanilla = price_gk(args.spot, args.strike, args.tenor, args.rd, args.rf,
                        smile.vol_at_strike(args.strike), "call", args.notional)
    print(f"\n  Vanilla call: {vanilla['price']:>12,.2f}")
    print(f"  Barrier/Vanilla: {res['price']/vanilla['price']:.2%}")


# ── vol surface ───────────────────────────────────────────────────────────────

def cmd_surface(args):
    print("=" * 65)
    print("  FX IMPLIED VOL SURFACE")
    print("=" * 65)

    tenors = [1/12, 3/12, 6/12, 1.0, 2.0]
    atms   = [0.065, 0.070, 0.075, 0.080, 0.085]
    rrs    = [0.008, 0.009, 0.010, 0.010, 0.011]
    bfs    = [0.002, 0.002, 0.003, 0.003, 0.004]

    print(f"\nSpot = {args.spot:.4f}")
    print(f"\nSmile surface (25P / ATM / 25C vols by tenor):")
    header = f"  {'Tenor':>7}  {'ATM':>7}  {'25P':>7}  {'25C':>7}  {'RR25':>7}  {'BF25':>7}"
    print(header)

    for T, atm, rr, bf in zip(tenors, atms, rrs, bfs):
        q = FXSmileQuotes(S=args.spot, T=T, r_d=args.rd, r_f=args.rf,
                           atm=atm, rr25=rr, bf25=bf)
        smile = build_smile(q)
        # After sort: [25P, ATM, 25C]
        sigma_25P = smile.vols[0]
        sigma_ATM = smile.vols[1]
        sigma_25C = smile.vols[2]
        label = f"{int(T*12):2d}M" if T < 1 else f"{int(T):2d}Y"
        print(f"  {label:>7}  {sigma_ATM:>7.2%}  {sigma_25P:>7.2%}  {sigma_25C:>7.2%}  {rr:>+7.2%}  {bf:>7.2%}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="FX Options Analysis")
    sub = p.add_subparsers(dest="cmd")

    # common
    def _add_common(sp):
        sp.add_argument("--spot",   type=float, default=1.08)
        sp.add_argument("--rd",     type=float, default=0.05, help="domestic rate")
        sp.add_argument("--rf",     type=float, default=0.03, help="foreign rate")
        sp.add_argument("--tenor",  type=float, default=1.0)

    # gk
    sp_gk = sub.add_parser("gk", help="Garman-Kohlhagen pricing")
    _add_common(sp_gk)
    sp_gk.add_argument("--strike",   type=float, default=1.08)
    sp_gk.add_argument("--vol",      type=float, default=0.08)
    sp_gk.add_argument("--notional", type=float, default=1_000_000)

    # smile
    sp_sm = sub.add_parser("smile", help="FX smile from RR/BF quotes")
    _add_common(sp_sm)
    sp_sm.add_argument("--atm",   type=float, default=0.080)
    sp_sm.add_argument("--rr25",  type=float, default=0.010)
    sp_sm.add_argument("--bf25",  type=float, default=0.003)
    sp_sm.add_argument("--rr10",  type=float, default=0.0)
    sp_sm.add_argument("--bf10",  type=float, default=0.0)

    # barrier
    sp_bar = sub.add_parser("barrier", help="FX barrier option (VV)")
    _add_common(sp_bar)
    sp_bar.add_argument("--strike",   type=float, default=1.10)
    sp_bar.add_argument("--barrier",  type=float, default=1.02)
    sp_bar.add_argument("--btype",    type=str,   default="down-out")
    sp_bar.add_argument("--notional", type=float, default=1_000_000)
    sp_bar.add_argument("--atm",  type=float, default=0.080)
    sp_bar.add_argument("--rr25", type=float, default=0.010)
    sp_bar.add_argument("--bf25", type=float, default=0.003)

    # surface
    sp_sur = sub.add_parser("surface", help="vol surface term structure")
    _add_common(sp_sur)

    args = p.parse_args()

    dispatch = {"gk": cmd_gk, "smile": cmd_smile,
                "barrier": cmd_barrier, "surface": cmd_surface}

    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
