"""Thin ingestion API with SQLite caching and source fallback."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from core.daycount import year_fraction
from data.normalization import normalize_ohlcv
from data.registry import SourceName, get_source
from db.database import insert_market_snapshot, insert_option_chain, insert_rate_curve, upsert_ohlcv

# US Treasury constant-maturity series on FRED, keyed by the tenor labels this
# module has always used (kept stable so callers/UI don't need to change).
_FRED_TREASURY_SERIES = {"1M": "DGS1MO", "3M": "DGS3MO", "1Y": "DGS1", "2Y": "DGS2"}
_PROTOTYPE_RATES = {"1M": 0.045, "3M": 0.046, "1Y": 0.043, "2Y": 0.041}


def get_quote(symbol: str, source: SourceName = "auto", db_path: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    candidates = ["yfinance", "stooq"] if source == "auto" else [source]
    for name in candidates:
        try:
            snap = get_source(name).snapshot(symbol)
            insert_market_snapshot(snap.symbol, snap.source, snap.observed_at, snap.price, snap.currency, snap.raw, path=db_path)
            return snap.__dict__
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("; ".join(errors))


def get_history(
    symbol: str,
    start: date | str | None = None,
    end: date | str | None = None,
    source: SourceName = "auto",
    period: str = "1y",
    db_path: str | None = None,
) -> pd.DataFrame:
    errors: list[str] = []
    candidates = ["yfinance", "stooq"] if source == "auto" else [source]
    for name in candidates:
        try:
            adapter = get_source(name)
            kwargs: dict[str, Any] = {"period": period}
            if start:
                kwargs["start"] = str(start)
            if end:
                kwargs["end"] = str(end)
            df = adapter.history(symbol, **kwargs)
            rows = normalize_ohlcv(df, symbol=symbol, source=adapter.name)
            upsert_ohlcv(rows, path=db_path)
            return df
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("; ".join(errors))


def ingest_history(symbol: str, source_name: SourceName = "auto", db_path: str | None = None, **kwargs: Any) -> int:
    df = get_history(symbol, source=source_name, db_path=db_path, **kwargs)
    source = "yfinance" if source_name == "auto" else str(source_name)
    return len(normalize_ohlcv(df, symbol=symbol, source=source))


def ingest_snapshot(symbol: str, source_name: SourceName = "auto", db_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    return get_quote(symbol, source=source_name, db_path=db_path)


def get_dividend_yield(symbol: str) -> dict[str, Any]:
    """Real trailing-twelve-month dividend yield (see YFinanceSource.dividend_yield)."""
    yld = get_source("yfinance").dividend_yield(symbol)
    return {"symbol": symbol.upper(), "dividend_yield": yld, "source": "yfinance", "window": "ttm"}


def get_crypto_price(asset: str, db_path: str | None = None) -> dict[str, Any]:
    snap = get_source("coingecko").snapshot(asset)
    insert_market_snapshot(snap.symbol, snap.source, snap.observed_at, snap.price, snap.currency, snap.raw, path=db_path)
    return snap.__dict__


def get_rates_curve(currency: str = "USD", db_path: str | None = None) -> dict[str, Any]:
    """USD risk-free curve from live FRED Treasury constant-maturity series.

    Prefers live sources in order: FRED Treasury constant-maturity series
    (needs FRED_API_KEY), then a Yahoo Treasury yield-index proxy
    (^IRX/^FVX/^TNX, no key required) — falling back to a fixed prototype
    curve only if every live source is unreachable or the currency isn't
    USD. The workbench should never hard-crash for lack of an API key, but
    it also shouldn't silently serve a stale constant when a real market
    source is available.
    """
    observed_at = datetime.utcnow().isoformat()
    if currency.upper() != "USD":
        return {"currency": currency, "source": "prototype_fallback", "tenors": dict(_PROTOTYPE_RATES), "observed_at": observed_at}

    if os.getenv("FRED_API_KEY"):
        fred = get_source("fred")
        start = (datetime.utcnow() - timedelta(days=30)).date().isoformat()
        tenors: dict[str, float] = {}
        for tenor, series_id in _FRED_TREASURY_SERIES.items():
            try:
                df = fred.history(series_id, observation_start=start)
                if df.empty:
                    continue
                tenors[tenor] = float(df.iloc[-1]["close"]) / 100.0  # FRED reports percent
            except Exception:
                continue

        if tenors:
            merged = {**_PROTOTYPE_RATES, **tenors}
            insert_rate_curve(currency, merged, source="fred", observed_at=observed_at, path=db_path)
            return {"currency": currency, "source": "fred", "tenors": merged, "observed_at": observed_at}

    try:
        tenors = get_source("yfinance").treasury_yields()
        insert_rate_curve(currency, tenors, source="yfinance_treasury_proxy", observed_at=observed_at, path=db_path)
        return {"currency": currency, "source": "yfinance_treasury_proxy", "tenors": tenors, "observed_at": observed_at}
    except Exception:
        pass

    return {"currency": currency, "source": "prototype_fallback", "tenors": dict(_PROTOTYPE_RATES), "observed_at": observed_at}


def get_option_chain(symbol: str, max_expiries: int = 6, db_path: str | None = None) -> dict[str, Any]:
    """Real option chain (calls + puts) from Yahoo Finance, cached to SQLite.

    Same source/library already proven for arbitrage scanning in
    arbitrage/vol_surface.py, exposed here as a general-purpose data accessor.
    """
    import yfinance as yf

    fields = ["strike", "bid", "ask", "lastPrice", "impliedVolatility", "volume", "openInterest"]
    tk = yf.Ticker(symbol)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])

    all_expiries = tk.options
    if not all_expiries:
        raise RuntimeError(f"No option expiries available for {symbol}")

    today = datetime.now().date()
    expiries: list[str] = []
    for exp in all_expiries:
        maturity_years = year_fraction(today, exp, "act365f")
        if maturity_years > 0.02:
            expiries.append(exp)
        if len(expiries) >= max_expiries:
            break

    chains: list[dict[str, Any]] = []
    for exp in expiries:
        chain = tk.option_chain(exp)
        calls_df = chain.calls[[c for c in fields if c in chain.calls.columns]]
        puts_df = chain.puts[[c for c in fields if c in chain.puts.columns]]
        # NaN/Inf (illiquid strikes with no bid/ask/IV) aren't valid JSON — normalize to None.
        calls = calls_df.astype(object).where(calls_df.notna(), None).to_dict(orient="records")
        puts = puts_df.astype(object).where(puts_df.notna(), None).to_dict(orient="records")
        record = {"symbol": symbol.upper(), "expiry": exp, "calls": calls, "puts": puts}
        chains.append(record)
        insert_option_chain(symbol.upper(), "yfinance", exp, record, path=db_path)

    return {"symbol": symbol.upper(), "spot": spot, "source": "yfinance", "expiries": expiries, "chains": chains}
