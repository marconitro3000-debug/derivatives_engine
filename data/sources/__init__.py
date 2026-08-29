"""Market-data source adapters."""

from data.sources.base import MarketSnapshot, normalize_ohlcv
from data.sources.coingecko_source import CoinGeckoSource
from data.sources.fred_source import FredSource
from data.sources.stooq_source import StooqSource
from data.sources.yfinance_source import YFinanceSource

__all__ = [
    "CoinGeckoSource",
    "FredSource",
    "MarketSnapshot",
    "StooqSource",
    "YFinanceSource",
    "normalize_ohlcv",
]
