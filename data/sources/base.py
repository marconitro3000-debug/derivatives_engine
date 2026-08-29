from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    source: str
    price: float
    observed_at: str
    currency: str | None = None
    raw: dict | None = None

    @classmethod
    def now(cls, symbol: str, source: str, price: float, currency: str | None = None, raw: dict | None = None) -> "MarketSnapshot":
        return cls(
            symbol=symbol,
            source=source,
            price=float(price),
            observed_at=datetime.now(timezone.utc).isoformat(),
            currency=currency,
            raw=raw or {},
        )


class MarketDataSource(Protocol):
    name: str

    def history(self, symbol: str, **kwargs) -> pd.DataFrame:
        ...

    def snapshot(self, symbol: str, **kwargs) -> MarketSnapshot:
        ...


def normalize_ohlcv(df: pd.DataFrame, symbol: str, source: str) -> list[dict]:
    if df is None or df.empty:
        return []
    clean = df.copy()
    clean.columns = [str(c).lower().replace(" ", "_") for c in clean.columns]
    if "date" not in clean.columns:
        clean = clean.reset_index()
        clean.columns = [str(c).lower().replace(" ", "_") for c in clean.columns]
    if "adj_close" not in clean.columns and "adjclose" in clean.columns:
        clean["adj_close"] = clean["adjclose"]
    rows = []
    for _, row in clean.iterrows():
        date_val = row.get("date") or row.get("datetime") or row.get("index")
        if hasattr(date_val, "date"):
            date_str = date_val.date().isoformat()
        else:
            date_str = str(date_val)[:10]
        close = row.get("close")
        if close is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "source": source,
                "date": date_str,
                "open": _float_or_none(row.get("open")),
                "high": _float_or_none(row.get("high")),
                "low": _float_or_none(row.get("low")),
                "close": float(close),
                "adj_close": _float_or_none(row.get("adj_close", row.get("close"))),
                "volume": _float_or_none(row.get("volume")),
            }
        )
    return rows


def _float_or_none(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None
