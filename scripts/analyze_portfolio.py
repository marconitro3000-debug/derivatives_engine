"""
scripts/analyze_portfolio.py
Portfolio risk analysis: Greeks, VaR, CVaR, stress testing.

Usage
-----
python scripts/analyze_portfolio.py [--vol 0.20] [--confidence 0.99]

Output: output/portfolio_<YYYYMMDD_HHMMSS>/
  greeks.png      — per-position Greeks bar chart
  var_dist.png    — P&L distribution with VaR / CVaR
  var_compare.png — VaR comparison across methods
  stress.png      — scenario P&L bar chart
  grid.png        — spot × vol P&L heatmap
  results.txt     — full text log
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")

from portfolio import (
    Position, Portfolio,
    var_historical, var_parametric, var_monte_carlo, var_cornish_fisher,
    var_comparison, backtest_var,
    STANDARD_SCENARIOS, stress_portfolio, spot_vol_grid,
)
from portfolio.charts import (
    plot_pnl_distribution, plot_var_comparison,
    plot_greeks_bar, plot_stress_results, plot_spot_vol_grid,
)


def build_demo_portfolio(annual_vol: float = 0.20) -> Portfolio:
    """Build a representative mixed options + equity portfolio."""
    S, r = 100.0, 0.05

    positions = [
        Position("call", 20, dict(S=S, K=100, T=0.5, r=r, sigma=annual_vol,
                                   multiplier=100), "Long ATM call"),
        Position("put", 15, dict(S=S, K=100, T=0.5, r=r, sigma=annual_vol,
                                  multiplier=100), "Long ATM put"),
        Position("call", -10, dict(S=S, K=110, T=0.5, r=r, sigma=annual_vol,
                                    multiplier=100), "Short OTM call"),
        Position("put", -8, dict(S=S, K=90, T=0.5, r=r, sigma=annual_vol,
                                  multiplier=100), "Short OTM put"),
        Position("stock", 500, dict(S=S, multiplier=1), "Long equity"),
    ]
    return Portfolio(positions)


def main():
    parser = argparse.ArgumentParser(prog="analyze_portfolio.py")
    parser.add_argument("--vol",        type=float, default=0.20)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--sims",       type=int,   default=100_000)
    args = parser.parse_args()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("output", f"portfolio_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 62)
    print("  PORTFOLIO RISK ANALYSIS")
    print("=" * 62)

    port = build_demo_portfolio(args.vol)

    # ── Portfolio summary ─────────────────────────────────────────────────────
    print(f"\n{port}")
    print(f"\n  {'Label':<22}  {'Qty':>6}  {'Value':>12}  {'Delta':>10}  "
          f"{'Gamma':>8}  {'Vega':>8}  {'Theta':>8}")
    for row in port.greeks_by_position():
        print(f"  {row['label']:<22}  {row['qty']:>6.0f}  {row['value']:>12,.0f}  "
              f"{row['delta']:>10,.1f}  {row['gamma']:>8.3f}  "
              f"{row['vega']:>8.1f}  {row['theta']:>8.1f}")

    ag = port.aggregate_greeks()
    print(f"\n  {'TOTAL':<22}  {'':>6}  {port.total_value():>12,.0f}  "
          f"{ag['delta']:>10,.1f}  {ag['gamma']:>8.3f}  "
          f"{ag['vega']:>8.1f}  {ag['theta']:>8.1f}")

    # ── Greeks chart ──────────────────────────────────────────────────────────
    fig = plot_greeks_bar(port)
    fig.savefig(os.path.join(out_dir, "greeks.png"), dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt; plt.close("all")

    # ── VaR / CVaR ────────────────────────────────────────────────────────────
    print(f"\nVaR/CVaR  [{args.confidence*100:.0f}% confidence]")
    rng     = np.random.default_rng(42)
    daily_v = args.vol / np.sqrt(252)
    pnl_hist = port.pnl_vector(rng.normal(0, daily_v, 5_000))

    results = var_comparison(pnl_hist, args.confidence)
    for name, r in results.items():
        print(f"  {r}")

    mc_r = var_monte_carlo(port, args.confidence, n_sims=args.sims,
                            annual_vol=args.vol)
    print(f"  {mc_r}")

    # VaR charts
    fig = plot_pnl_distribution(results["historical"],
                                 title="Portfolio P&L Distribution")
    fig.savefig(os.path.join(out_dir, "var_dist.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    fig = plot_var_comparison(results)
    fig.savefig(os.path.join(out_dir, "var_compare.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── Stress testing ────────────────────────────────────────────────────────
    print(f"\nSTRESS TEST  ({len(STANDARD_SCENARIOS)} scenarios)")
    stress = stress_portfolio(port, STANDARD_SCENARIOS)
    stress_sorted = sorted(stress, key=lambda r: r.pnl)
    print(f"  {'Scenario':<26}  {'PnL':>12}  {'%':>8}")
    for r in stress_sorted:
        print(f"  {r.scenario.name:<26}  {r.pnl:>+12,.0f}  {r.pnl_pct:>+7.1f}%")

    fig = plot_stress_results(stress)
    fig.savefig(os.path.join(out_dir, "stress.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── Spot × Vol grid ───────────────────────────────────────────────────────
    print("\nSPOT × VOL P&L GRID")
    spot_shocks = np.linspace(-0.40, 0.40, 9)
    vol_shocks  = np.linspace(-0.20, 0.20, 7)
    grid        = spot_vol_grid(port, spot_shocks, vol_shocks)
    print(f"  {'Spot↓ / Vol→':<14}", end="")
    for dv in vol_shocks:
        print(f"  {dv:>+.0%}", end="")
    print()
    for ds, row in zip(spot_shocks, grid):
        print(f"  {ds:>+.0%}{'':<9}", end="")
        for v in row:
            print(f"  {v:>+6,.0f}", end="")
        print()

    fig = plot_spot_vol_grid(grid, spot_shocks, vol_shocks)
    fig.savefig(os.path.join(out_dir, "grid.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    print(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    main()
