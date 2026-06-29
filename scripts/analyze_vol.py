"""
scripts/analyze_vol.py
Volatility analysis CLI: realized estimators, GARCH(1,1), VRP.

Usage
-----
# All estimators + GARCH + VRP for AAPL
python scripts/analyze_vol.py AAPL

# Custom window and GARCH horizon
python scripts/analyze_vol.py SPY --window 21 --horizon 60

# Just realized estimators, no GARCH
python scripts/analyze_vol.py NVDA --no-garch

Output: output/<TICKER_YYYYMMDD_HHMMSS>_vol/
  realized_vol.png     — price + 5 estimators overlaid
  garch_forecast.png   — GARCH conditional vol + forecast cone
  garch_diagnostics.png— residuals, ACF, QQ, vol histogram
  vrp.png              — IV vs RV + VRP history
  results.txt          — full text summary
"""

import argparse
import io
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import yfinance as yf

from volatility.realized import all_estimators, close_to_close
from volatility.garch     import fit as garch_fit, forecast as garch_forecast
from volatility.variance_swap import vrp, vrp_summary, realized_variance_from_prices
from volatility.charts    import (plot_realized_vol, plot_garch_forecast,
                                   plot_garch_diagnostics, plot_vrp)


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


def _make_outdir(ticker: str) -> str:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join("output", f"{ticker}_{ts}_vol")
    os.makedirs(out, exist_ok=True)
    return out


def _fetch_atm_iv(ticker: str) -> float | None:
    """Try to get the ATM 30-day implied vol from the nearest-expiry options chain."""
    try:
        tk   = yf.Ticker(ticker)
        spot = tk.history(period="5d")["Close"].iloc[-1]
        exps = tk.options
        if not exps:
            return None
        # Find expiry closest to 30 calendar days out
        today  = pd.Timestamp.today()
        target = today + pd.Timedelta(days=30)
        exp    = min(exps, key=lambda e: abs(pd.Timestamp(e) - target))
        chain  = tk.option_chain(exp)
        calls  = chain.calls
        # ATM: strike closest to spot
        atm    = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]
        iv     = float(atm["impliedVolatility"])
        return iv if 0.01 < iv < 5.0 else None
    except Exception:
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="analyze_vol.py",
        description="Volatility analysis: realized estimators, GARCH, VRP",
    )
    parser.add_argument("ticker",           type=str)
    parser.add_argument("--period",   "-p", type=str, default="2y",
                        help="yfinance period (default 2y)")
    parser.add_argument("--window",   "-w", type=int, default=21,
                        help="Rolling window in days (default 21)")
    parser.add_argument("--horizon",  "-H", type=int, default=30,
                        help="GARCH forecast horizon in days (default 30)")
    parser.add_argument("--no-garch",       action="store_true",
                        help="Skip GARCH estimation")
    args = parser.parse_args()

    out_dir = _make_outdir(args.ticker)
    tee     = _Tee(); sys.stdout = tee

    # ── 1. Fetch OHLCV data ───────────────────────────────────────────────────
    print("=" * 60)
    print(f"  VOLATILITY ENGINE — {args.ticker}")
    print("=" * 60)
    print(f"\nFetching {args.period} OHLCV data from Yahoo Finance...")

    try:
        df = yf.download(args.ticker, period=args.period,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("No data returned")
    except Exception as e:
        sys.stdout = tee._out
        print(f"Error fetching data: {e}")
        return

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close  = df["Close"]
    n_days = len(close)
    print(f"  {n_days} trading days  ({close.index[0].date()} → {close.index[-1].date()})")
    print(f"  Latest close: ${float(close.iloc[-1]):.2f}")

    # ── 2. Realized vol estimators ────────────────────────────────────────────
    print(f"\nREALIZED VOLATILITY  (window = {args.window}d)")
    vol_df = all_estimators(df, window=args.window, annualize=True)
    latest = vol_df.iloc[-1]
    for col in vol_df.columns:
        print(f"  {col:<20}: {latest[col]:.2%}")

    # ── 3. GARCH(1,1) ─────────────────────────────────────────────────────────
    garch_result = None
    fcast_df     = None

    if not args.no_garch:
        print("\nFITTING GARCH(1,1)...")
        log_ret = np.log(close / close.shift(1)).dropna()
        try:
            garch_result = garch_fit(log_ret)
            print(garch_result.summary())

            fcast_df = garch_forecast(garch_result, h=args.horizon)
            print(f"\nFORECAST ({args.horizon}d)")
            for _, row in fcast_df.iloc[[0, 4, 9, 19, -1]].iterrows():
                print(f"  t+{int(row.horizon):3d}d  "
                      f"vol={row.forecast_vol:.2%}  "
                      f"[{row.vol_lb_95:.2%}, {row.vol_ub_95:.2%}]")
        except Exception as e:
            print(f"  GARCH fit failed: {e}")

    # ── 4. Volatility risk premium ────────────────────────────────────────────
    print("\nVOLATILITY RISK PREMIUM")
    atm_iv = _fetch_atm_iv(args.ticker)
    rv_now = float(latest["Yang-Zhang"])

    if atm_iv is not None:
        vrp_now = atm_iv - rv_now
        print(f"  ATM IV (30d):    {atm_iv:.2%}")
        print(f"  YZ RV ({args.window}d):    {rv_now:.2%}")
        print(f"  VRP = IV − RV:   {vrp_now:+.2%}  "
              f"({'market paying for protection' if vrp_now > 0 else 'vol is cheap'})")

        # Historical VRP: align rolling RV with a constant IV estimate (simplified)
        rv_ts  = vol_df["Yang-Zhang"].dropna()
        iv_ts  = pd.Series(atm_iv, index=rv_ts.index, name="IV")
        vrp_ts = vrp(iv_ts, rv_ts)
        vstats = vrp_summary(vrp_ts)
        print(f"\n  VRP stats (full history):")
        print(f"    Mean:    {vstats['mean']:+.2%}  "
              f"  Median: {vstats['median']:+.2%}")
        print(f"    Std:     {vstats['std']:.2%}  "
              f"  % positive: {vstats['pct_pos']:.0%}")
    else:
        print("  Could not fetch ATM IV (options chain unavailable)")
        iv_ts  = None
        vrp_ts = None

    # ── 5. Charts ─────────────────────────────────────────────────────────────
    print("\nCHARTS")

    f1 = plot_realized_vol(close, vol_df, out_dir, args.ticker)
    print(f"  {f1}")

    if garch_result is not None and fcast_df is not None:
        f2 = plot_garch_forecast(garch_result, fcast_df, close, out_dir, args.ticker)
        f3 = plot_garch_diagnostics(garch_result, out_dir, args.ticker)
        print(f"  {f2}")
        print(f"  {f3}")

    if iv_ts is not None and vrp_ts is not None:
        f4 = plot_vrp(iv_ts, vol_df["Yang-Zhang"], vrp_ts, out_dir, args.ticker)
        print(f"  {f4}")

    sys.stdout = tee._out
    with open(os.path.join(out_dir, "results.txt"), "w") as fh:
        fh.write(tee.getvalue())
    print(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    main()
