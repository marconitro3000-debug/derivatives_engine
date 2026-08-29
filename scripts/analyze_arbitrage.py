"""
scripts/analyze_arbitrage.py
CLI: static arbitrage detection in the implied-volatility surface.

Runs three scanners in sequence:
  1. Vol surface  — calendar-spread and butterfly arbitrage (SVI-fitted)
  2. Put-call parity — PCP violations adjusted for bid-ask
  3. Local volatility — Dupire local vol vs SSVI implied vol comparison

Usage
-----
python scripts/analyze_arbitrage.py SPY
python scripts/analyze_arbitrage.py AAPL --r 0.045 --output output/arb_aapl
python scripts/analyze_arbitrage.py SPY --no-charts
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Static arbitrage scanner — vol surface, PCP, local vol"
    )
    p.add_argument("ticker",          type=str,  help="Yahoo Finance ticker (e.g. SPY)")
    p.add_argument("--r",             type=float, default=0.04,
                   help="Risk-free rate (default 0.04)")
    p.add_argument("--q",             type=float, default=0.00,
                   help="Dividend yield (default 0.00)")
    p.add_argument("--min-volume",    type=int,   default=5,
                   help="Minimum option volume for inclusion (default 5)")
    p.add_argument("--max-expiries",  type=int,   default=6,
                   help="Number of expiries to download (default 6)")
    p.add_argument("--output",        type=str,   default="output",
                   help="Output directory for charts")
    p.add_argument("--no-charts",     action="store_true",
                   help="Skip saving charts")
    return p.parse_args()


def main():
    args   = _parse()
    ticker = args.ticker.upper()
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'━'*60}")
    print(f"  Arbitrage Scanner  —  {ticker}")
    print(f"{'━'*60}")

    # ── 1. Vol surface arbitrage ──────────────────────────────────────────────
    print("\n[1/3] Scanning vol surface for calendar + butterfly arbitrage…")
    from arbitrage.vol_surface import VolSurfaceArbScanner

    try:
        scanner = VolSurfaceArbScanner()
        arb_result = scanner.scan_live(
            ticker,
            r             = args.r,
            min_volume    = args.min_volume,
            max_expiries  = args.max_expiries,
        )
        print(arb_result.summary())

        if not args.no_charts:
            _plot_vol_surface(arb_result, args.output, ticker)

    except Exception as e:
        print(f"  ✗ Vol surface scan failed: {e}")
        arb_result = None

    # ── 2. Put-call parity ────────────────────────────────────────────────────
    print("\n[2/3] Scanning put-call parity…")
    from arbitrage.put_call_parity import PCPScanner

    try:
        pcp_result = PCPScanner(min_profit=0.02).scan_live(
            ticker,
            r            = args.r,
            q            = args.q,
            min_volume   = args.min_volume,
            max_expiries = args.max_expiries,
        )
        print(pcp_result.summary())

        if not args.no_charts:
            _plot_pcp(pcp_result, args.output, ticker)

    except Exception as e:
        print(f"  ✗ PCP scan failed: {e}")
        pcp_result = None

    # ── 3. Local vol vs implied vol ───────────────────────────────────────────
    print("\n[3/3] Building Dupire local vol surface from SSVI calibration…")

    if arb_result is not None:
        try:
            _run_local_vol(arb_result, args, ticker)
        except Exception as e:
            print(f"  ✗ Local vol failed: {e}")
    else:
        print("  ✗ Skipped (no vol surface result).")

    print(f"\n{'━'*60}")
    print(f"  Done. Charts saved to: {args.output}/")
    print(f"{'━'*60}\n")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_vol_surface(result, outdir: str, ticker: str):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{ticker} — Implied Vol Surface  ({result.scan_time})", fontsize=13)

    # Panel 1: total variance slices
    ax = axes[0]
    colors = cm.viridis(np.linspace(0, 1, len(result.maturities)))
    for i, T in enumerate(result.maturities):
        w = result.total_var_matrix[i]
        ax.plot(result.k_grid, w, color=colors[i], label=f"T={T:.3f}Y")
    ax.set_xlabel("Log-moneyness k")
    ax.set_ylabel("Total variance  w(k,T)")
    ax.set_title("Total Variance Slices")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Highlight calendar violations
    for v in result.calendar_violations:
        ax.axvline(v.worst_k, color="red", linestyle="--", alpha=0.6,
                   label=f"Cal arb k={v.worst_k:+.2f}")

    # Panel 2: risk-neutral density
    ax = axes[1]
    for i, T in enumerate(result.maturities):
        g = result.density_matrix[i]
        ax.plot(result.k_grid, g, color=colors[i], label=f"T={T:.3f}Y")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Log-moneyness k")
    ax.set_ylabel("Density proxy g(k,T)")
    ax.set_title("Risk-Neutral Density  (g < 0 → butterfly arb)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, f"{ticker.lower()}_vol_surface_arb.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved: {path}")


def _plot_pcp(result, outdir: str, ticker: str):
    import matplotlib.pyplot as plt

    if len(result.all_deviations) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(result.all_deviations, "o-", markersize=4, label="C − P − (Fwd)")
    ax.fill_between(
        range(len(result.all_deviations)),
        -result.max_deviation * 0.1,
        result.max_deviation * 0.1,
        alpha=0.15, color="green", label="±10% of max dev"
    )
    ax.set_xlabel("Option pair index")
    ax.set_ylabel("PCP deviation ($)")
    ax.set_title(f"{ticker} — Put-Call Parity Deviations  (n={result.n_pairs} pairs)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, f"{ticker.lower()}_pcp.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved: {path}")


def _run_local_vol(arb_result, args, ticker: str):
    import matplotlib.pyplot as plt
    from ml.ssvi import calibrate_ssvi
    from volatility.local_vol import from_ssvi_and_atm

    slices = arb_result.slices
    sorted_T = sorted(slices.keys())

    if len(sorted_T) < 2:
        print("  Not enough slices for SSVI surface calibration.")
        return

    # Collect per-slice SVI params for SSVI joint calibration
    spot = arb_result.spot
    r    = args.r
    k_ref = np.linspace(-0.25, 0.25, 41)

    k_lists = []
    w_lists = []
    theta_list = []
    atm_ivs = []

    for T in sorted_T:
        sf    = slices[T]
        p     = sf.params
        w_mkt = p.total_var(k_ref)
        k_lists.append(k_ref)
        w_lists.append(w_mkt)

        # ATM total variance = w(k=0)
        theta = float(p.total_var(np.array([0.0]))[0])
        theta_list.append(theta)
        atm_ivs.append(np.sqrt(theta / T))

    # Joint SSVI calibration across slices
    try:
        ssvi = calibrate_ssvi(k_lists, w_lists, theta_list)
    except Exception as e:
        print(f"  SSVI calibration failed: {e}")
        return

    T_arr   = np.array(sorted_T)
    iv_arr  = np.array(atm_ivs)
    lv_surf = from_ssvi_and_atm(ssvi, T_arr, iv_arr)

    # Plot local vol vs implied vol for each slice
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{ticker} — Dupire Local Vol vs SSVI Implied Vol", fontsize=13)
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 1, len(sorted_T)))

    k_dense = np.linspace(-0.30, 0.30, 201)

    ax0, ax1 = axes
    for i, T in enumerate(sorted_T):
        sl = lv_surf.slice(T, k_dense)
        ax0.plot(sl.k, sl.local_vol  * 100, color=colors[i], label=f"T={T:.3f}Y")
        ax0.plot(sl.k, sl.implied_vol * 100, color=colors[i], linestyle="--", alpha=0.5)
    ax0.set_xlabel("Log-moneyness k")
    ax0.set_ylabel("Volatility (%)")
    ax0.set_title("Local Vol (solid) vs Implied Vol (dashed)")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    for i, T in enumerate(sorted_T):
        ratio = lv_surf.lv_iv_ratio(T, k_dense)
        ax1.plot(k_dense, ratio, color=colors[i], label=f"T={T:.3f}Y")
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Log-moneyness k")
    ax1.set_ylabel("σ_local / σ_implied")
    ax1.set_title("Local Vol / Implied Vol Ratio")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.output, f"{ticker.lower()}_local_vol.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Chart saved: {path}")

    # Print ATM summary table
    print("\n  ATM Local Vol vs Implied Vol:")
    print(f"  {'T':>6}  {'ATM IV':>8}  {'ATM LV':>8}  {'LV/IV':>6}")
    for T in sorted_T:
        iv = lv_surf.atm_iv(T)
        lv = float(lv_surf(np.array([0.0]), T)[0])
        print(f"  {T:>6.3f}  {iv*100:>7.2f}%  {lv*100:>7.2f}%  {lv/iv:>6.3f}")


if __name__ == "__main__":
    main()
