"""
volatility/realized.py
Realized (historical) volatility estimators from OHLCV data.

All estimators return annualized volatility (not variance) by default.
Pass annualize=False to get daily vol.

Estimators (in order of efficiency):
  close_to_close  — classical, biased in presence of drift, ignores intraday
  parkinson       — uses High-Low range, ~5× more efficient than CC
  garman_klass    — OHLC, ~8× more efficient than CC
  rogers_satchell — drift-invariant, doesn't require Open
  yang_zhang      — best overall: handles overnight gaps + drift
  ewma            — exponentially weighted, responds faster to vol changes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


# ── helpers ───────────────────────────────────────────────────────────────────

def _ann(var_daily: pd.Series, annualize: bool) -> pd.Series:
    """Convert daily variance series to vol (annualized or daily)."""
    vol = np.sqrt(var_daily)
    return vol * np.sqrt(_TRADING_DAYS) if annualize else vol


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(1, window // 2)).mean()


# ── estimators ────────────────────────────────────────────────────────────────

def close_to_close(close: pd.Series, window: int = 21,
                   annualize: bool = True) -> pd.Series:
    """
    Close-to-Close realized vol (classical historical vol).

    σ²_CC = (1/n) Σ r²_t    where r_t = ln(C_t / C_{t-1})

    Biased downward in the presence of drift; ignores intraday price moves.
    Efficiency relative to Parkinson: ~1×  (baseline)
    """
    r    = np.log(close / close.shift(1))
    var  = _rolling_mean(r ** 2, window)
    return _ann(var, annualize).rename("CC")


def parkinson(high: pd.Series, low: pd.Series, window: int = 21,
              annualize: bool = True) -> pd.Series:
    """
    Parkinson (1980) estimator using the High-Low range.

    σ²_P = 1/(4 ln 2) × E[(ln H/L)²]

    ~5× more efficient than CC. Assumes no drift (slight bias otherwise).
    Ignores overnight gaps.
    """
    hl  = np.log(high / low) ** 2
    var = _rolling_mean(hl, window) / (4.0 * np.log(2.0))
    return _ann(var, annualize).rename("Parkinson")


def garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series,
                 close: pd.Series, window: int = 21,
                 annualize: bool = True) -> pd.Series:
    """
    Garman-Klass (1980) estimator using OHLC data.

    σ²_GK = E[½·(ln H/L)² − (2ln2 − 1)·(ln C/O)²]

    ~8× more efficient than CC.
    Assumes log-normal prices with no drift.
    """
    hl  = 0.5 * np.log(high / low) ** 2
    co  = (2.0 * np.log(2.0) - 1.0) * np.log(close / open_) ** 2
    var = _rolling_mean(hl - co, window)
    return _ann(var, annualize).rename("Garman-Klass")


def rogers_satchell(open_: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series, window: int = 21,
                    annualize: bool = True) -> pd.Series:
    """
    Rogers-Satchell (1991) estimator.

    σ²_RS = E[ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)]

    Drift-invariant (does not require μ = 0).
    Ignores overnight gaps.
    """
    hc  = np.log(high / close)
    ho  = np.log(high / open_)
    lc  = np.log(low  / close)
    lo  = np.log(low  / open_)
    var = _rolling_mean(hc * ho + lc * lo, window)
    return _ann(var, annualize).rename("Rogers-Satchell")


def yang_zhang(open_: pd.Series, high: pd.Series, low: pd.Series,
               close: pd.Series, window: int = 21,
               annualize: bool = True) -> pd.Series:
    """
    Yang-Zhang (2000) estimator — minimum-variance, drift-invariant,
    handles overnight gaps.

    σ²_YZ = σ²_open + k·σ²_CC + (1−k)·σ²_RS

    where σ²_open = Var[ln(O_t/C_{t-1})] (overnight return variance)
    and k = 0.34 / (1.34 + (n+1)/(n-1)).

    ~14× more efficient than CC for typical equity data.
    """
    n    = window
    k    = 0.34 / (1.34 + (n + 1) / (n - 1))

    # Overnight return
    r_on   = np.log(open_ / close.shift(1))
    var_on = _rolling_mean((r_on - _rolling_mean(r_on, n)) ** 2, n)

    # Open-to-close (CC on open-to-close)
    r_oc   = np.log(close / open_)
    var_oc = _rolling_mean((r_oc - _rolling_mean(r_oc, n)) ** 2, n)

    # Rogers-Satchell (variance)
    rs_var = rogers_satchell(open_, high, low, close, window, annualize=False) ** 2

    var = var_on + k * var_oc + (1.0 - k) * rs_var
    return _ann(var, annualize).rename("Yang-Zhang")


def ewma(close: pd.Series, lam: float = 0.94,
         annualize: bool = True) -> pd.Series:
    """
    Exponentially Weighted Moving Average (EWMA / RiskMetrics) volatility.

    σ²_t = λ·σ²_{t-1} + (1−λ)·r²_t

    λ = 0.94 is the RiskMetrics daily default.
    Responds quickly to vol shocks; never reaches a long-run mean.
    """
    r   = np.log(close / close.shift(1)).fillna(0.0)
    var = r.ewm(com=lam / (1.0 - lam), adjust=False).var()
    return _ann(var, annualize).rename(f"EWMA(λ={lam})")


# ── combined ──────────────────────────────────────────────────────────────────

def all_estimators(df: pd.DataFrame, window: int = 21,
                   annualize: bool = True) -> pd.DataFrame:
    """
    Compute all estimators from an OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame with columns ['Open','High','Low','Close']
         (column names case-insensitive)

    Returns
    -------
    DataFrame with columns: CC, Parkinson, Garman-Klass, Rogers-Satchell,
                             Yang-Zhang, EWMA(λ=0.94)
    """
    cols = {c.lower(): c for c in df.columns}
    O = df[cols.get("open",  "Open")]
    H = df[cols.get("high",  "High")]
    L = df[cols.get("low",   "Low")]
    C = df[cols.get("close", "Close")]

    return pd.concat([
        close_to_close(C, window, annualize),
        parkinson(H, L, window, annualize),
        garman_klass(O, H, L, C, window, annualize),
        rogers_satchell(O, H, L, C, window, annualize),
        yang_zhang(O, H, L, C, window, annualize),
        ewma(C, annualize=annualize),
    ], axis=1)


def realized_variance(close: pd.Series, window: int = 21) -> pd.Series:
    """
    Daily realized variance (sum of squared intraday returns in rolling window).
    Returns variance in daily units (not annualized).
    """
    r = np.log(close / close.shift(1))
    return (r ** 2).rolling(window, min_periods=1).sum() / window
