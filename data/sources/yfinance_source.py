from __future__ import annotations

import pandas as pd

from data.sources.base import MarketSnapshot


class YFinanceSource:
    name = "yfinance"

    def history(self, symbol: str, period: str = "1y", interval: str = "1d", **kwargs) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install yfinance to use YFinanceSource") from exc
        start = kwargs.get("start")
        end = kwargs.get("end")
        if start or end:
            df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        else:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df

    def snapshot(self, symbol: str, **kwargs) -> MarketSnapshot:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install yfinance to use YFinanceSource") from exc
        t = yf.Ticker(symbol)
        info = getattr(t, "fast_info", {}) or {}
        price = info.get("last_price") or info.get("lastPrice")
        currency = info.get("currency")
        if price is None:
            hist = t.history(period="5d", interval="1d")
            if hist.empty:
                raise RuntimeError(f"No yfinance price available for {symbol}")
            price = float(hist["Close"].dropna().iloc[-1])
        return MarketSnapshot.now(symbol=symbol, source=self.name, price=float(price), currency=currency, raw=dict(info))

    def treasury_yields(self) -> dict[str, float]:
        """Live USD Treasury curve proxy built from Yahoo's Treasury yield
        indices (^IRX 13-week bill, ^FVX 5Y note, ^TNX 10Y note): used as a
        real-market fallback when no FRED_API_KEY is configured, so the
        risk-free rate is never a hardcoded constant unless every live
        source is genuinely unreachable. 1M/3M are set to the 13-week bill
        rate (no shorter live index is available); 1Y/2Y are linearly
        interpolated between the 13-week and 5Y points."""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install yfinance to use YFinanceSource") from exc

        points: dict[float, float] = {}
        for ticker, tenor_years in (("^IRX", 0.25), ("^FVX", 5.0), ("^TNX", 10.0)):
            hist = yf.Ticker(ticker).history(period="5d", interval="1d")
            if hist.empty or "Close" not in hist:
                continue
            points[tenor_years] = float(hist["Close"].dropna().iloc[-1]) / 100.0

        if 0.25 not in points:
            raise RuntimeError("No live Treasury yield data available from yfinance")

        short_tenor, short_yield = 0.25, points[0.25]
        longer = [(t, y) for t, y in points.items() if t > short_tenor]
        if not longer:
            return {"1M": short_yield, "3M": short_yield, "1Y": short_yield, "2Y": short_yield}
        long_tenor, long_yield = min(longer)

        def interp(target_years: float) -> float:
            weight = min(target_years / long_tenor, 1.0)
            return short_yield + (long_yield - short_yield) * weight

        return {"1M": short_yield, "3M": short_yield, "1Y": interp(1.0), "2Y": interp(2.0)}

    def dividend_yield(self, symbol: str) -> float:
        """Trailing-twelve-month dividend yield from real dividend payments,
        rather than yfinance's `info["dividendYield"]` field (known to be
        inconsistently scaled/unreliable) — sum of the last 365 days of
        dividends actually paid, divided by the current price."""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install yfinance to use YFinanceSource") from exc
        t = yf.Ticker(symbol)
        divs = t.dividends
        if divs is None or divs.empty:
            return 0.0
        cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
        ttm_dividends = float(divs[divs.index >= cutoff].sum())
        if ttm_dividends <= 0:
            return 0.0
        hist = t.history(period="5d", interval="1d")
        if hist.empty or "Close" not in hist:
            return 0.0
        price = float(hist["Close"].dropna().iloc[-1])
        return ttm_dividends / price if price > 0 else 0.0
