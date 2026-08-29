from __future__ import annotations

import requests
import pandas as pd
import os

from data.sources.base import MarketSnapshot


class FredSource:
    name = "fred"
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    def history(self, symbol: str, api_key: str | None = None, **kwargs) -> pd.DataFrame:
        key = api_key or os.getenv("FRED_API_KEY")
        if not key:
            raise RuntimeError("FRED requires FRED_API_KEY. Add it to .env or environment variables.")
        params = {
            "series_id": symbol,
            "api_key": key,
            "file_type": "json",
            "observation_start": kwargs.get("observation_start", "2000-01-01"),
        }
        r = requests.get(self.base_url, params=params, timeout=30)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        df = pd.DataFrame(obs)
        if df.empty:
            return df
        df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
        df = df.dropna(subset=["value"])
        return pd.DataFrame({
            "date": df["date"],
            "open": df["value"],
            "high": df["value"],
            "low": df["value"],
            "close": df["value"],
            "volume": None,
        })

    def snapshot(self, symbol: str, **kwargs) -> MarketSnapshot:
        df = self.history(symbol, **kwargs)
        if df.empty:
            raise RuntimeError(f"No FRED data for {symbol}")
        last = df.iloc[-1]
        return MarketSnapshot.now(symbol=symbol, source=self.name, price=float(last["close"]), raw=last.to_dict())
