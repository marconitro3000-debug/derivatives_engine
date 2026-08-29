from __future__ import annotations

import requests

from data.sources.base import MarketSnapshot


class CoinGeckoSource:
    name = "coingecko"
    base_url = "https://api.coingecko.com/api/v3"

    def snapshot(self, symbol: str, vs_currency: str = "usd", **kwargs) -> MarketSnapshot:
        # CoinGecko uses coin ids, e.g. bitcoin, ethereum, solana.
        url = f"{self.base_url}/simple/price"
        params = {"ids": symbol, "vs_currencies": vs_currency, "include_last_updated_at": "true"}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if symbol not in data or vs_currency not in data[symbol]:
            raise RuntimeError(f"No CoinGecko price for {symbol}/{vs_currency}")
        return MarketSnapshot.now(
            symbol=symbol,
            source=self.name,
            price=float(data[symbol][vs_currency]),
            currency=vs_currency.upper(),
            raw=data[symbol],
        )

    def history(self, symbol: str, vs_currency: str = "usd", days: str = "365", **kwargs):
        import pandas as pd
        url = f"{self.base_url}/coins/{symbol}/market_chart"
        params = {"vs_currency": vs_currency, "days": days, "interval": kwargs.get("interval", "daily")}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        prices = r.json().get("prices", [])
        df = pd.DataFrame(prices, columns=["timestamp_ms", "close"])
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date.astype(str)
        df["open"] = df["high"] = df["low"] = df["close"]
        df["volume"] = None
        return df[["date", "open", "high", "low", "close", "volume"]]
