"""
scripts/analyze_commodities.py
CLI for commodity derivatives analysis.

Usage:
  python scripts/analyze_commodities.py curve
  python scripts/analyze_commodities.py schwartz
  python scripts/analyze_commodities.py spread
  python scripts/analyze_commodities.py option
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from commodities import (
    FuturesCurve, crack_spread, spark_spread,
    implied_convenience_yield, futures_fair_price,
    convenience_yield_curve,
    spread_option_margrabe, spread_option_kirk,
    SchwartzParams, futures_price, calibrate_schwartz,
    simulate_schwartz, price_commodity_option, schwartz_fit_summary,
)


# ── Futures curve analysis ────────────────────────────────────────────────────

def cmd_curve(args):
    print("=" * 65)
    print("  COMMODITY FUTURES CURVE ANALYSIS")
    print("=" * 65)

    # WTI-like example curve (or oil curve)
    mats   = np.array([1/12, 3/12, 6/12, 1.0, 1.5, 2.0, 3.0])
    prices = np.array([args.spot * 1.000,
                       args.spot * 1.006,
                       args.spot * 1.015,
                       args.spot * 1.028,
                       args.spot * 1.038,
                       args.spot * 1.046,
                       args.spot * 1.058])

    if args.backwardation:
        prices = args.spot / (prices / args.spot)   # flip to backwardation

    curve = FuturesCurve(maturities=mats, futures_prices=prices,
                          spot=args.spot, commodity=args.commodity)

    shape = "Contango" if curve.is_contango() else "Backwardation"
    print(f"\n{curve.commodity}  |  Spot: {args.spot:.2f}  |  Structure: {shape}")
    print(curve.summary())

    # Convenience yields
    cy_curve = convenience_yield_curve(prices, mats, args.spot,
                                        r=args.r, storage_cost=args.storage)
    print(f"\nImplied Convenience Yields (r={args.r:.1%}, storage={args.storage:.1%}):")
    print(f"  {'Tenor':>7}  {'Futures':>10}  {'Cy yield':>10}")
    for T, F, cy in zip(mats, prices, cy_curve.yields):
        label = f"{int(T*12):2d}M" if T < 1 else f"{T:.1f}Y"
        print(f"  {label:>7}  {F:>10.3f}  {cy:>10.2%}")

    # Calendar spreads
    print(f"\nCalendar Spreads:")
    spread_pairs = [(1/12, 3/12), (3/12, 6/12), (6/12, 1.0), (1.0, 2.0)]
    for T1, T2 in spread_pairs:
        cs  = curve.calendar_spread(T1, T2)
        ry  = curve.roll_yield(T1, T2)
        l1  = f"{int(T1*12)}M" if T1 < 1 else f"{T1:.1f}Y"
        l2  = f"{int(T2*12)}M" if T2 < 1 else f"{T2:.1f}Y"
        print(f"  {l1}/{l2}: spread={cs:+.3f}  roll_yield={ry:+.2%}")

    # Crack/Spark spreads
    if args.commodity.upper() in ("WTI", "CRUDE", "OIL"):
        gasoline = args.spot * 1.25
        ho       = args.spot * 1.20
        cs = crack_spread(args.spot, gasoline, ho)
        print(f"\n  3-2-1 Crack Spread: ${cs:.2f}/bbl")
        ss = spark_spread(power_price=50, gas_price=args.spot / 20, heat_rate=7.5)
        print(f"  Spark Spread: ${ss:.2f}/MWh")


# ── Schwartz model ────────────────────────────────────────────────────────────

def cmd_schwartz(args):
    print("=" * 65)
    print("  SCHWARTZ (1997) ONE-FACTOR MEAN-REVERSION MODEL")
    print("=" * 65)

    # Build a synthetic market curve then calibrate
    true_p = SchwartzParams(kappa=args.kappa, mu_star=np.log(args.spot),
                             sigma=args.vol)
    T_list = np.array([1/12, 3/12, 6/12, 1.0, 1.5, 2.0, 3.0, 5.0])
    F_mkt  = futures_price(args.spot, T_list, true_p)

    # add small noise to simulate market
    rng   = np.random.default_rng(42)
    F_mkt = F_mkt * (1 + rng.normal(0, 0.002, len(T_list)))

    print(f"\nCalibrating to noisy futures curve (S0={args.spot:.2f})...")
    fitted, rmse = calibrate_schwartz(F_mkt, T_list, args.spot)

    print(f"\nTrue params  : kappa={args.kappa:.3f}  mu*=ln({args.spot:.1f})  sigma={args.vol:.3f}")
    print(f"Fitted params: kappa={fitted.kappa:.3f}  mu*={fitted.mu_star:.3f}  sigma={fitted.sigma:.3f}")
    print(f"  Half-life   : {fitted.half_life():.2f}y")
    print(f"  Long-run F  : {fitted.long_run_price():.3f}")
    print(f"  RMSE        : {rmse:.4f}")

    print(f"\n{schwartz_fit_summary(args.spot, T_list, F_mkt, fitted)}")

    # Option pricing
    print(f"\nOption Prices (ATM, MC 50k paths):")
    print(f"  {'Type':>5}  {'K':>8}  {'Price':>10}  {'95% CI':>20}")
    for otype in ("call", "put"):
        res = price_commodity_option(args.spot, args.spot, 1.0, args.r,
                                      fitted, otype, n_sims=50_000, n_steps=100)
        ci = f"[{res['conf_95_lo']:.3f}, {res['conf_95_hi']:.3f}]"
        print(f"  {otype:>5}  {args.spot:>8.2f}  {res['price']:>10.4f}  {ci:>20}")


# ── Spread options ────────────────────────────────────────────────────────────

def cmd_spread(args):
    print("=" * 65)
    print("  COMMODITY SPREAD OPTIONS")
    print("=" * 65)

    print(f"\nMargrabe exchange option (K=0):")
    print(f"  F1={args.f1:.2f}  F2={args.f2:.2f}  T={args.tenor}y")
    print(f"  sigma1={args.sig1:.1%}  sigma2={args.sig2:.1%}  rho={args.rho:.2f}")

    m = spread_option_margrabe(args.f1, args.f2, args.tenor,
                                args.sig1, args.sig2, args.rho, args.r)
    print(f"  Call: {m['price']:.4f}   spread_vol={m['spread_vol']:.2%}")

    m_put = spread_option_margrabe(args.f1, args.f2, args.tenor,
                                    args.sig1, args.sig2, args.rho, args.r, "put")
    print(f"  Put : {m_put['price']:.4f}")
    fwd_diff = (args.f1 - args.f2) * np.exp(-args.r * args.tenor)
    print(f"  Parity check C-P={(m['price']-m_put['price']):.4f}  F1-F2(disc)={fwd_diff:.4f}")

    print(f"\nKirk approximation (K=5.0):")
    for K in [0, 2, 5, 10]:
        k = spread_option_kirk(args.f1, args.f2, float(K), args.tenor,
                                args.sig1, args.sig2, args.rho, args.r)
        print(f"  K={K:>4.0f}  price={k['price']:.4f}")

    # Sensitivity to correlation
    print(f"\nSensitivity to correlation:")
    print(f"  {'rho':>6}  {'Exchange':>12}  {'Kirk(K=5)':>12}")
    for rho in [-0.5, 0.0, 0.3, 0.6, 0.9]:
        m_  = spread_option_margrabe(args.f1, args.f2, args.tenor,
                                      args.sig1, args.sig2, rho, args.r)
        k_  = spread_option_kirk(args.f1, args.f2, 5.0, args.tenor,
                                  args.sig1, args.sig2, rho, args.r)
        print(f"  {rho:>6.2f}  {m_['price']:>12.4f}  {k_['price']:>12.4f}")


# ── Commodity option (Schwartz MC) ────────────────────────────────────────────

def cmd_option(args):
    print("=" * 65)
    print("  COMMODITY OPTION PRICER (SCHWARTZ 1F + MC)")
    print("=" * 65)

    params = SchwartzParams(kappa=args.kappa, mu_star=np.log(args.spot),
                             sigma=args.vol)
    print(f"\n{params}")
    print(f"\n  {'K':>8}  {'Call':>10}  {'SE':>8}  {'Put':>10}  {'SE':>8}")
    strikes = np.linspace(args.spot * 0.85, args.spot * 1.15, 7)
    for K in strikes:
        c = price_commodity_option(args.spot, K, args.tenor, args.r, params,
                                    "call", n_sims=30_000, n_steps=100)
        p = price_commodity_option(args.spot, K, args.tenor, args.r, params,
                                    "put", n_sims=30_000, n_steps=100)
        print(f"  {K:>8.2f}  {c['price']:>10.4f}  {c['std_error']:>8.5f}  "
              f"{p['price']:>10.4f}  {p['std_error']:>8.5f}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Commodity Derivatives Analysis")
    sub = p.add_subparsers(dest="cmd")

    def _add_spot(sp, default=80.0):
        sp.add_argument("--spot",      type=float, default=default)
        sp.add_argument("--r",         type=float, default=0.05)
        sp.add_argument("--commodity", type=str,   default="WTI")

    # curve
    sp_c = sub.add_parser("curve", help="Futures curve + convenience yield")
    _add_spot(sp_c)
    sp_c.add_argument("--storage",      type=float, default=0.02)
    sp_c.add_argument("--backwardation", action="store_true")

    # schwartz
    sp_s = sub.add_parser("schwartz", help="Schwartz 1F calibration + pricing")
    _add_spot(sp_s)
    sp_s.add_argument("--kappa", type=float, default=0.8)
    sp_s.add_argument("--vol",   type=float, default=0.30)

    # spread
    sp_sp = sub.add_parser("spread", help="Spread options (Margrabe/Kirk)")
    sp_sp.add_argument("--f1",    type=float, default=100.0)
    sp_sp.add_argument("--f2",    type=float, default=90.0)
    sp_sp.add_argument("--sig1",  type=float, default=0.20)
    sp_sp.add_argument("--sig2",  type=float, default=0.25)
    sp_sp.add_argument("--rho",   type=float, default=0.60)
    sp_sp.add_argument("--tenor", type=float, default=1.0)
    sp_sp.add_argument("--r",     type=float, default=0.05)

    # option
    sp_o = sub.add_parser("option", help="Commodity option via Schwartz MC")
    _add_spot(sp_o)
    sp_o.add_argument("--kappa", type=float, default=0.8)
    sp_o.add_argument("--vol",   type=float, default=0.30)
    sp_o.add_argument("--tenor", type=float, default=1.0)

    args = p.parse_args()

    dispatch = {"curve": cmd_curve, "schwartz": cmd_schwartz,
                "spread": cmd_spread, "option": cmd_option}

    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
