"""Source registry for public market data adapters."""

from __future__ import annotations

from typing import Literal

from data.sources.coingecko_source import CoinGeckoSource
from data.sources.fred_source import FredSource
from data.sources.stooq_source import StooqSource
from data.sources.yfinance_source import YFinanceSource

SourceName = Literal["auto", "yfinance", "stooq", "coingecko", "fred"]


def get_source(name: SourceName):
    if name == "auto":
        name = "yfinance"
    if name == "yfinance":
        return YFinanceSource()
    if name == "stooq":
        return StooqSource()
    if name == "coingecko":
        return CoinGeckoSource()
    if name == "fred":
        return FredSource()
    raise ValueError(f"Unsupported source: {name}")
