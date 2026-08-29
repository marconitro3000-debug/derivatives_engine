from __future__ import annotations

from io import StringIO
import requests
import pandas as pd

from data.sources.base import MarketSnapshot


class StooqSource:
    name = "stooq"
    base_url = "https://stooq.com/q/d/l/"

    def history(self, symbol: str, **kwargs) -> pd.DataFrame:
        # Stooq symbols often use suffixes such as aapl.us, msft.us, spy.us
        params = {"s": symbol.lower(), "i": kwargs.get("interval", "d")}
        r = requests.get(self.base_url, params=params, timeout=20)
        r.raise_for_status()
        if "No data" in r.text or len(r.text.strip()) < 20:
            raise RuntimeError(f"No Stooq history for {symbol}")
        return pd.read_csv(StringIO(r.text))

    def snapshot(self, symbol: str, **kwargs) -> MarketSnapshot:
        df = self.history(symbol, **kwargs)
        if df.empty:
            raise RuntimeError(f"No Stooq snapshot for {symbol}")
        last = df.iloc[-1]
        return MarketSnapshot.now(symbol=symbol, source=self.name, price=float(last["Close"]), raw=last.to_dict())
