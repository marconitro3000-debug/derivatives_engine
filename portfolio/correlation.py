"""Real-market volatility and correlation for a basket of tickers.

Pulls daily closes via `data.ingestion.get_history` (yfinance, same source
already used for realized-vol calibration elsewhere in the app) and derives
annualized per-name volatility and the realized correlation matrix from
overlapping daily log returns. No synthetic/hardcoded correlation — if a
ticker's history can't be fetched, it's dropped and reported as missing
rather than silently defaulted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.ingestion import get_history

TRADING_DAYS_PER_YEAR = 252


@dataclass
class CorrelationResult:
    tickers: list[str]
    volatility: dict[str, float]  # annualized, per ticker
    correlation: list[list[float]]  # tickers x tickers, same order as `tickers`
    missing: list[str]
    lookback_days: int


def estimate_correlation(tickers: list[str], period: str = "1y") -> CorrelationResult:
    closes: dict[str, pd.Series] = {}
    missing: list[str] = []
    for ticker in tickers:
        try:
            df = get_history(ticker, period=period)
            series = df["Close"].dropna()
            if len(series) < 20:
                missing.append(ticker)
                continue
            closes[ticker] = series
        except Exception:
            missing.append(ticker)

    ok_tickers = [t for t in tickers if t in closes]
    if not ok_tickers:
        return CorrelationResult(tickers=[], volatility={}, correlation=[], missing=missing, lookback_days=0)

    price_frame = pd.DataFrame({t: closes[t] for t in ok_tickers}).dropna(how="any")
    log_returns = np.log(price_frame / price_frame.shift(1)).dropna(how="any")

    volatility = {
        t: float(log_returns[t].std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) for t in ok_tickers
    }
    corr_matrix = log_returns.corr().reindex(index=ok_tickers, columns=ok_tickers).to_numpy()

    return CorrelationResult(
        tickers=ok_tickers,
        volatility=volatility,
        correlation=corr_matrix.tolist(),
        missing=missing,
        lookback_days=int(len(log_returns)),
    )
