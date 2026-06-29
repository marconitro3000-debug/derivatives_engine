"""
scripts/ml_pricing.py
ML-based pricing and forecasting CLI.

Subcommands
-----------
lsm      — Longstaff-Schwartz American option pricing
           python scripts/ml_pricing.py lsm AAPL --strike 185 --expiry 1.0
ssvi     — SVI / SSVI smile calibration
           python scripts/ml_pricing.py ssvi AAPL
forecast — Vol forecasting: HAR, GBM, LSTM
           python scripts/ml_pricing.py forecast AAPL

Output: output/ml_<cmd>_<YYYYMMDD_HHMMSS>/
  lsm_convergence.png   — price vs n_sims convergence
  ssvi_smile.png        — SVI fitted smile
  forecast.png          — predicted vs actual RV + model comparison
  results.txt           — full text log
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
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

_BLUE   = "#0072B2"
_ORANGE = "#E69F00"
_GREEN  = "#009E73"
_RED    = "#D55E00"
_PURPLE = "#CC79A7"
STYLE = {
    "font.family": "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
}

from ml import (
    price_american_lsm, price_bermudan_lsm,
    SVIParams, calibrate_svi,
    train_har, train_gbm, compare_models,
)


def _make_outdir(name: str) -> str:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join("output", f"ml_{name}_{ts}")
    os.makedirs(out, exist_ok=True)
    return out


# ── LSM ───────────────────────────────────────────────────────────────────────

def cmd_lsm(args, out_dir: str):
    try:
        import yfinance as yf
        ticker = yf.Ticker(args.ticker)
        S = ticker.fast_info["lastPrice"]
        print(f"  Live price {args.ticker}: ${S:.2f}")
    except Exception:
        S = 100.0
        print(f"  Using synthetic S = {S:.2f}")

    r, sigma = args.rate, args.vol
    K, T     = args.strike, args.expiry

    print(f"\nLONGSTAFF-SCHWARTZ AMERICAN OPTION")
    print(f"  {args.ticker}  S={S:.2f}  K={K:.2f}  T={T}y  r={r:.2%}  σ={sigma:.0%}")
    print(f"  Type: {args.type}")

    # European BS benchmark
    from options.black_scholes import price as bs_price
    eu = bs_price(S, K, T, r, sigma, args.type)
    print(f"\n  European BS  : {eu:>10.4f}")

    # Convergence study
    print(f"\n  {'N_sims':<12}  {'Price':>10}  {'StdErr':>10}  {'95% CI':>20}  {'EE prem':>10}")
    prices = []
    sims_list = [2_000, 5_000, 10_000, 25_000, 50_000]
    for n in sims_list:
        r_lsm = price_american_lsm(S, K, T, r, sigma, args.type,
                                    n_sims=n, n_steps=100, seed=42)
        prices.append(r_lsm.price)
        print(f"  {n:<12,}  {r_lsm.price:>10.4f}  {r_lsm.std_error:>10.4f}  "
              f"[{r_lsm.conf_95_lo:.4f}, {r_lsm.conf_95_hi:.4f}]  "
              f"{r_lsm.early_exercise_premium:>10.4f}")

    # Bermudan comparison
    ex_dates = [T * k / 4 for k in range(1, 5)]
    berm = price_bermudan_lsm(S, K, T, r, sigma, ex_dates, args.type,
                               n_sims=20_000, n_steps=200)
    print(f"\n  Bermudan (4 dates): {berm.price:.4f}")

    # Chart
    plt.rcParams.update(STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"LSM American {args.type.title()}  {args.ticker}  K={K:.0f}  T={T}y", fontsize=11)

    ax1.semilogx(sims_list, prices, color=_BLUE, lw=2, marker="o", ms=6, label="LSM")
    ax1.axhline(eu, color=_RED, ls="--", lw=1.5, label=f"European = {eu:.3f}")
    ax1.axhline(berm.price, color=_GREEN, ls=":", lw=1.5, label=f"Bermudan = {berm.price:.3f}")
    ax1.set(xlabel="Simulations (log)", ylabel="Price ($)", title="Price Convergence")
    ax1.legend()

    # Final LSM detailed result
    final = price_american_lsm(S, K, T, r, sigma, args.type,
                                n_sims=50_000, n_steps=200, seed=42)
    bd = final.exercise_boundary
    valid = ~np.isnan(bd)
    if valid.any():
        steps = np.arange(len(bd))[valid] / len(bd) * T
        ax2.plot(steps, bd[valid], color=_BLUE, lw=2, label="Exercise boundary")
        ax2.axhline(K, color=_RED, ls="--", lw=1.2, label=f"Strike = {K:.0f}")
        ax2.set(xlabel="Time (years)", ylabel="Spot price",
                title="Early Exercise Boundary")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lsm_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  lsm_convergence.png")


# ── SSVI ──────────────────────────────────────────────────────────────────────

def cmd_ssvi(args, out_dir: str):
    try:
        import yfinance as yf
        ticker = yf.Ticker(args.ticker)
        S = ticker.fast_info["lastPrice"]
        # Try to get option chain IV
        exp_dates = ticker.options
        if exp_dates:
            chain = ticker.option_chain(exp_dates[1])
            calls = chain.calls.dropna(subset=["impliedVolatility", "strike"])
            calls = calls[(calls["inTheMoney"] == False) | (calls["inTheMoney"] == True)]
            F     = S
            T     = 0.25
            strikes = calls["strike"].values
            ivs     = calls["impliedVolatility"].values
            mask    = (ivs > 0.01) & (ivs < 5.0)
            strikes, ivs = strikes[mask], ivs[mask]
            k     = np.log(strikes / F)
            w_mkt = ivs**2 * T
            print(f"  Live data: {len(k)} strikes from {args.ticker}")
        else:
            raise RuntimeError("No option data")
    except Exception as e:
        print(f"  Synthetic smile (no live data: {e})")
        true_p = SVIParams(a=0.04, b=0.10, rho=-0.30, m=0.0, sigma=0.15)
        k      = np.linspace(-0.40, 0.40, 20)
        rng    = np.random.default_rng(42)
        w_mkt  = true_p.total_var(k) + rng.normal(0, 2e-5, len(k))
        w_mkt  = np.maximum(w_mkt, 1e-6)
        T      = 1.0

    print(f"\nSVI CALIBRATION  (T={T}y)")
    fitted = calibrate_svi(k, w_mkt)
    w_fit  = fitted.total_var(k)

    rmse_w  = np.sqrt(np.mean((w_fit - w_mkt)**2))
    iv_mkt  = np.sqrt(np.maximum(w_mkt, 0) / T)
    iv_fit  = np.sqrt(np.maximum(w_fit, 0) / T)
    rmse_iv = np.sqrt(np.mean((iv_fit - iv_mkt)**2)) * 100

    print(f"  a={fitted.a:.5f}  b={fitted.b:.5f}  rho={fitted.rho:.4f}  "
          f"m={fitted.m:.4f}  sigma={fitted.sigma:.4f}")
    print(f"  RMSE (total var): {rmse_w:.2e}")
    print(f"  RMSE (IV bps)   : {rmse_iv:.2f}")

    plt.rcParams.update(STYLE)
    k_dense = np.linspace(k.min() - 0.1, k.max() + 0.1, 300)
    iv_dense = fitted.implied_vol(k_dense, T)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(k * 100, iv_mkt * 100, color="black", s=30, zorder=5, label="Market")
    ax.plot(k_dense * 100, iv_dense * 100, color=_BLUE, lw=2, label="SVI fitted")
    ax.set(title=f"SVI Smile  {args.ticker}  T={T}y",
           xlabel="Log-moneyness (%)", ylabel="Implied Vol (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ssvi_smile.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ssvi_smile.png")


# ── Forecast ──────────────────────────────────────────────────────────────────

def cmd_forecast(args, out_dir: str):
    try:
        import yfinance as yf
        hist = yf.Ticker(args.ticker).history(period="5y")
        if len(hist) < 300:
            raise ValueError("Not enough data")
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna().values
        rv      = log_ret**2   # daily squared returns as RV proxy
        print(f"  Live data: {len(rv)} daily observations")
    except Exception as e:
        print(f"  Synthetic GARCH series ({e})")
        rng = np.random.default_rng(42)
        n   = 1500
        rv  = np.zeros(n)
        rv[0] = 0.0004
        for t in range(1, n):
            eps   = rng.standard_normal()
            rv[t] = max(0.00005 + 0.10 * rv[t-1] * eps**2 + 0.85 * rv[t-1], 1e-8)

    print(f"\nVOLATILITY FORECASTING  ({args.ticker})")
    results = compare_models(rv, include_lstm=args.lstm)

    print(f"\n  {'Model':<20} {'RMSE':>12} {'MAE':>12} {'QLIKE':>10} {'R²':>8}")
    for r in results:
        print(f"  {r.model_name:<20} {r.rmse:>12.8f} {r.mae:>12.8f} "
              f"{r.qlike:>10.4f} {r.r2:>8.4f}")

    # Charts
    plt.rcParams.update(STYLE)
    cols = [_BLUE, _ORANGE, _GREEN]
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 3 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]
    fig.suptitle(f"Vol Forecasting  {args.ticker}", fontsize=11)

    for ax, r, col in zip(axes, results, cols):
        x = np.arange(len(r.actuals))
        ax.plot(x, np.sqrt(r.actuals) * np.sqrt(252) * 100,
                color="black", alpha=0.5, lw=0.8, label="Realized")
        ax.plot(x, np.sqrt(r.predictions) * np.sqrt(252) * 100,
                color=col, lw=1.5, label=f"{r.model_name} (RMSE={r.rmse:.2e})")
        ax.set(ylabel="Ann. vol (%)")
        ax.legend(fontsize=8)

    axes[-1].set(xlabel="Test days")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forecast.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  forecast.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="ml_pricing.py")
    sub    = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("lsm")
    p.add_argument("ticker",  default="AAPL", nargs="?")
    p.add_argument("--strike",type=float, default=100.0)
    p.add_argument("--expiry",type=float, default=1.0)
    p.add_argument("--type",  default="put")
    p.add_argument("--vol",   type=float, default=0.20)
    p.add_argument("--rate",  type=float, default=0.05)

    p = sub.add_parser("ssvi")
    p.add_argument("ticker", default="AAPL", nargs="?")

    p = sub.add_parser("forecast")
    p.add_argument("ticker",  default="SPY", nargs="?")
    p.add_argument("--lstm",  action="store_true", default=False)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    out_dir = _make_outdir(args.cmd)

    print("=" * 60)
    print(f"  ML PRICING — {args.cmd.upper()}")
    print("=" * 60)

    if   args.cmd == "lsm":      cmd_lsm(args, out_dir)
    elif args.cmd == "ssvi":     cmd_ssvi(args, out_dir)
    elif args.cmd == "forecast": cmd_forecast(args, out_dir)

    print(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    main()
