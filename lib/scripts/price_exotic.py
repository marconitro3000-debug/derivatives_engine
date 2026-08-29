"""
scripts/price_exotic.py  —  Price exotic options from live market data.

Usage
-----
    python scripts/price_exotic.py TICKER EXOTIC_TYPE [options]

Exotic types
------------
  barrier   -- down-out, down-in, up-out, up-in
  asian     -- arithmetic and geometric average price
  lookback  -- floating or fixed strike
  digital   -- cash-or-nothing, asset-or-nothing, one-touch, no-touch

Examples
--------
    python scripts/price_exotic.py AAPL barrier --strike 185 --expiry 0.25 --barrier 165 --barrier-type down-out
    python scripts/price_exotic.py SPY  asian   --strike 500 --expiry 1.0
    python scripts/price_exotic.py TSLA lookback --expiry 0.5
    python scripts/price_exotic.py NVDA digital  --strike 900 --expiry 0.25 --digital-type cash-or-nothing
    python scripts/price_exotic.py GS   digital  --expiry 0.25 --barrier 500 --digital-type one-touch
"""

import argparse
import io
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")

from options.charts import ACADEMIC_STYLE
import matplotlib.pyplot as plt
plt.rcParams.update(ACADEMIC_STYLE)

from exotics.barrier  import price_barrier, mc_barrier, greeks_barrier
from exotics.asian    import price_asian_geo, price_asian_kv, mc_asian_arith
from exotics.lookback import price_lookback_float, mc_lookback
from exotics.digital  import (price_cash_or_nothing, price_asset_or_nothing,
                               price_one_touch, price_no_touch, mc_digital)
from exotics.charts   import (plot_barrier_payoff, plot_asian_paths,
                               plot_lookback_paths, plot_digital_payoff,
                               plot_exotic_comparison)
from options.black_scholes import price as bs_price


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    prog="price_exotic.py",
    description="Price exotic options using live Yahoo Finance spot and ATM IV.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument("ticker",      help="Underlying ticker (e.g. AAPL, SPY)")
parser.add_argument("exotic_type", choices=["barrier", "asian", "lookback", "digital"],
                    help="Type of exotic option")

parser.add_argument("--strike",   "-K", type=float, default=None,
                    help="Strike price (required for barrier, asian, digital)")
parser.add_argument("--expiry",   "-T", type=float, required=True,
                    help="Time to expiry in years (0.25 = 3 months)")
parser.add_argument("--type",     choices=["call", "put"], default="call",
                    help="Call or put  (default: call)")
parser.add_argument("--barrier",  "-B", type=float, default=None,
                    help="Barrier level H (for barrier and digital one-touch)")
parser.add_argument("--barrier-type", default="down-out",
                    choices=["down-out", "down-in", "up-out", "up-in"],
                    help="Barrier variety  (default: down-out)")
parser.add_argument("--digital-type", default="cash-or-nothing",
                    choices=["cash-or-nothing", "asset-or-nothing", "one-touch", "no-touch"],
                    help="Digital variety  (default: cash-or-nothing)")
parser.add_argument("--cash",     type=float, default=1.0,
                    help="Cash payout for digital/one-touch  (default: 1.0)")
parser.add_argument("--vol",  "-v", type=float, default=None,
                    help="Override implied vol (e.g. 0.20 for 20%%)")
parser.add_argument("--spot", "-S", type=float, default=None,
                    help="Override spot price")
parser.add_argument("--rate", "-r", type=float, default=0.045,
                    help="Risk-free rate  (default: 0.045)")
parser.add_argument("--sims",     type=int, default=100_000,
                    help="Monte Carlo simulations  (default: 100,000)")
args = parser.parse_args()

TICKER = args.ticker.upper()
T      = args.expiry
r      = args.rate
otype  = args.type
EXOTIC = args.exotic_type


# ── Output directory ──────────────────────────────────────────────────────────

RUN_CODE = f"{TICKER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", RUN_CODE,
)
os.makedirs(OUT_DIR, exist_ok=True)

_log = io.StringIO()


class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d): [s.write(d) for s in self.streams]
    def flush(self): [s.flush() for s in self.streams]

sys.stdout = _Tee(sys.__stdout__, _log)


# ── Fetch market data ─────────────────────────────────────────────────────────

tk = yf.Ticker(TICKER)

if args.spot is not None:
    S = args.spot
else:
    hist = tk.history(period="1d")
    if hist.empty:
        print(f"ERROR: cannot fetch price for {TICKER}", file=sys.stderr)
        sys.exit(1)
    S = float(hist["Close"].iloc[-1])

print(f"Spot {TICKER}:  ${S:.2f}")

if args.vol is not None:
    sig = args.vol
    print(f"Vol (manual): {sig:.2%}")
else:
    sig = None
    try:
        now = datetime.now()
        for exp in tk.options:
            T_exp = (datetime.strptime(exp, "%Y-%m-%d") - now).days / 365.0
            if T_exp >= 0.04:
                chain = tk.option_chain(exp).calls
                near  = chain[abs(chain["strike"] - S) / S < 0.02]
                if len(near) > 0:
                    sig = float(near["impliedVolatility"].mean())
                    break
    except Exception:
        pass
    if not sig or sig <= 0:
        sig = 0.20
        print("ATM IV unavailable — using 20% flat")
    else:
        print(f"Live ATM IV:  {sig:.2%}")

# Defaults
K   = args.strike or round(S * 1.02, 0)  # slightly OTM by default
H   = args.barrier

saved = []


def _save_fig(fig, name: str) -> str:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return name


# ── Header ────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print(f"  {TICKER}  |  {EXOTIC.upper()}  |  {otype.upper()}")
print(f"  Spot S = ${S:,.2f}  |  σ = {sig:.2%}  |  r = {r:.2%}")
if K:
    print(f"  Strike K = ${K:,.2f}  |  T = {T:.3f}Y ({T*365:.0f}d)")
if H:
    print(f"  Barrier H = ${H:,.2f}")
print(f"  Run ID  {RUN_CODE}")
print("=" * 60)
print()


# ── Barrier ───────────────────────────────────────────────────────────────────

if EXOTIC == "barrier":
    if H is None:
        # default barrier: 15% below spot for down, 15% above for up
        H = S * 0.85 if "down" in args.barrier_type else S * 1.15
        print(f"No barrier specified — using H = ${H:.2f} ({args.barrier_type})")

    btype = args.barrier_type

    # Closed-form (continuous monitoring)
    cf    = price_barrier(S, K, T, r, sig, H, otype, btype)
    van   = cf["vanilla"]
    disc  = cf["discount"]

    print(f"── Barrier: {btype} {otype} (Reiner-Rubinstein, continuous)")
    print(f"  Vanilla price:       ${van:.4f}")
    print(f"  Barrier price:       ${cf['price']:.4f}")
    print(f"  Barrier discount:    ${disc:.4f}  ({disc/van*100:.1f}% cheaper)" if van > 0 else "")

    # Monte Carlo (discrete daily monitoring)
    mc = mc_barrier(S, K, T, r, sig, H, otype, btype,
                    n_sims=args.sims, n_steps=int(T * 252), seed=42)
    print(f"\n── MC (discrete daily monitoring, {args.sims:,} paths)")
    print(f"  MC price:            ${mc['price']:.4f}  ±{mc['std_error']:.4f}")
    print(f"  Continuous gap:      ${abs(cf['price'] - mc['price']):.4f}  "
          f"(discretisation effect)")

    # Greeks
    g = greeks_barrier(S, K, T, r, sig, H, otype, btype)
    print(f"\n── Greeks (finite differences)")
    print(f"  Delta:          {g['delta']:+.4f}")
    print(f"  Gamma:           {g['gamma']:.5f}")
    print(f"  Theta (day):   ${g['theta_day']:.4f}")
    print(f"  Vega (1% σ):   ${g['vega_1pct']:.4f}")

    # Charts
    H_up = S * 1.15 if "down" in btype else H
    H_dn = H if "down" in btype else S * 0.85
    fn = plot_barrier_payoff(S, K, T, r, sig, H, otype, btype, OUT_DIR, TICKER)
    if fn: saved.append(fn)
    fn = plot_exotic_comparison(S, K, T, r, sig, H_dn, H_up, otype, OUT_DIR, TICKER)
    if fn: saved.append(fn)


# ── Asian ─────────────────────────────────────────────────────────────────────

elif EXOTIC == "asian":
    geo = price_asian_geo(S, K, T, r, sig, otype)
    kv  = price_asian_kv(S, K, T, r, sig, otype)
    mc  = mc_asian_arith(S, K, T, r, sig, otype, n_sims=args.sims, seed=42)
    van = bs_price(S, K, T, r, sig, otype)

    print(f"── Asian {otype} — fixed strike  K=${K:,.2f}")
    print(f"  Vanilla BS:               ${van:.4f}")
    print(f"  Geometric (closed-form):  ${geo['price']:.4f}  "
          f"(σ_adj={geo['adj_vol']:.3f})")
    print(f"  Arithmetic Kemna-Vorst:   ${kv['price']:.4f}  "
          f"(σ_adj={kv['adj_vol']:.3f})")
    print(f"  Arithmetic MC ({args.sims:,}):  ${mc['price']:.4f}  ±{mc['std_error']:.4f}")
    print(f"  95% CI:                  [${mc['conf_95_lo']:.4f}, ${mc['conf_95_hi']:.4f}]")
    print(f"\n  Reduction vs vanilla:     "
          f"${van - mc['price']:.4f}  ({(van-mc['price'])/van*100:.1f}% cheaper)")

    fn = plot_asian_paths(S, K, T, r, sig, otype, OUT_DIR, TICKER)
    if fn: saved.append(fn)

    H_dn = S * 0.85
    H_up = S * 1.15
    fn = plot_exotic_comparison(S, K, T, r, sig, H_dn, H_up, otype, OUT_DIR, TICKER)
    if fn: saved.append(fn)


# ── Lookback ──────────────────────────────────────────────────────────────────

elif EXOTIC == "lookback":
    cf = price_lookback_float(S, T, r, sig, otype)
    mc = mc_lookback(S, T, r, sig, otype, "float", n_sims=args.sims, n_steps=int(T*252), seed=42)

    # For reference: fixed-strike lookback
    mc_fix = None
    if K:
        mc_fix = mc_lookback(S, T, r, sig, otype, "fixed", K=K,
                             n_sims=args.sims, n_steps=int(T*252), seed=42)

    van = bs_price(S, K or S, T, r, sig, otype)

    print(f"── Lookback {otype} (floating strike)")
    print(f"  Closed-form (continuous): ${cf['price']:.4f}")
    print(f"  MC (discrete daily):      ${mc['price']:.4f}  ±{mc['std_error']:.4f}")
    print(f"  Vanilla BS (for ref):     ${van:.4f}")
    print(f"  Premium over vanilla:     ${cf['price'] - van:.4f}  "
          f"({(cf['price']-van)/van*100:.1f}% more expensive)")

    if mc_fix:
        print(f"\n── Lookback {otype} (fixed strike K=${K:,.2f})")
        print(f"  MC price:               ${mc_fix['price']:.4f}  ±{mc_fix['std_error']:.4f}")

    fn = plot_lookback_paths(S, T, r, sig, otype, OUT_DIR, TICKER)
    if fn: saved.append(fn)


# ── Digital ───────────────────────────────────────────────────────────────────

elif EXOTIC == "digital":
    dtype = args.digital_type

    if dtype == "cash-or-nothing":
        cf  = price_cash_or_nothing(S, K, T, r, sig, otype, args.cash)
        mc  = mc_digital(S, K, T, r, sig, "cash-or-nothing", otype,
                         cash=args.cash, n_sims=args.sims, seed=42)
        van = bs_price(S, K, T, r, sig, otype)

        print(f"── Cash-or-Nothing {otype}  K=${K:,.2f}  pays=${args.cash:.2f}")
        print(f"  Price (closed-form):  ${cf['price']:.4f}")
        print(f"  Risk-neutral prob ITM: {cf['prob_itm']:.3%}  (d2={cf['d2']:.4f})")
        print(f"  MC price:             ${mc['price']:.4f}  ±{mc['std_error']:.4f}")
        print(f"  Vanilla BS:           ${van:.4f}")

    elif dtype == "asset-or-nothing":
        cf  = price_asset_or_nothing(S, K, T, r, sig, otype)
        mc  = mc_digital(S, K, T, r, sig, "asset-or-nothing", otype, n_sims=args.sims, seed=42)
        van = bs_price(S, K, T, r, sig, otype)

        print(f"── Asset-or-Nothing {otype}  K=${K:,.2f}")
        print(f"  Price (closed-form):  ${cf['price']:.4f}")
        print(f"  d1 prob:               {cf['prob']:.3%}  (d1={cf['d1']:.4f})")
        print(f"  MC price:             ${mc['price']:.4f}  ±{mc['std_error']:.4f}")
        print(f"  Note: vanilla = AoN − CoN×K×e^{{-rT}} = "
              f"${cf['price'] - price_cash_or_nothing(S,K,T,r,sig,otype,K)['price']:.4f}")

    elif dtype == "one-touch":
        if H is None:
            H = S * 0.85 if "down" in args.barrier_type else S * 1.15
        touch = "down" if H < S else "up"
        cf  = price_one_touch(S, T, r, sig, H, touch, args.cash)
        mc  = mc_digital(S, K or S, T, r, sig, "one-touch", otype,
                         H=H, cash=args.cash, n_sims=args.sims, seed=42)

        print(f"── One-Touch ({touch})  H=${H:,.2f}  pays=${args.cash:.2f}")
        print(f"  Price (closed-form):  ${cf['price']:.4f}")
        print(f"  Prob of touch:         {cf['prob_touch']:.3%}")
        print(f"  MC price:             ${mc['price']:.4f}  ±{mc['std_error']:.4f}")

    elif dtype == "no-touch":
        if H is None:
            H = S * 0.85 if "down" in args.barrier_type else S * 1.15
        touch = "down" if H < S else "up"
        cf  = price_no_touch(S, T, r, sig, H, touch, args.cash)
        mc  = mc_digital(S, K or S, T, r, sig, "no-touch", otype,
                         H=H, cash=args.cash, n_sims=args.sims, seed=42)

        print(f"── No-Touch ({touch})  H=${H:,.2f}  pays=${args.cash:.2f}")
        print(f"  Price (closed-form):  ${cf['price']:.4f}")
        print(f"  Prob of no-touch:      {cf['prob_no_touch']:.3%}")
        print(f"  MC price:             ${mc['price']:.4f}  ±{mc['std_error']:.4f}")

    fn = plot_digital_payoff(S, K or S, T, r, sig, otype, args.cash, OUT_DIR, TICKER)
    if fn: saved.append(fn)


# ── Save log ──────────────────────────────────────────────────────────────────

sys.stdout = sys.__stdout__

with open(os.path.join(OUT_DIR, "results.txt"), "w", encoding="utf-8") as f:
    f.write(_log.getvalue())

print(f"\nSaved to  output/{RUN_CODE}/")
for fn in sorted(saved):
    sz = os.path.getsize(os.path.join(OUT_DIR, fn)) // 1024
    print(f"  {sz:4d} KB  {fn}")
print(f"           results.txt")
