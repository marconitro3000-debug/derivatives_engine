from __future__ import annotations

import pandas as pd
import pytest

from data import ingestion
from data.normalization import normalize_ohlcv
from db.database import connect, init_db


def test_normalize_ohlcv_rows():
    df = pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1234],
        }
    )
    rows = normalize_ohlcv(df, symbol="AAPL", source="unit")
    assert rows == [
        {
            "symbol": "AAPL",
            "source": "unit",
            "date": "2024-01-02",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "adj_close": 100.5,
            "volume": 1234.0,
        }
    ]


def test_get_rates_curve_yfinance_proxy_without_api_key(tmp_path, monkeypatch):
    db_file = tmp_path / "test.sqlite"
    init_db(db_file)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    class FakeYFinance:
        def treasury_yields(self):
            return {"1M": 0.052, "3M": 0.052, "1Y": 0.048, "2Y": 0.044}

    monkeypatch.setattr(ingestion, "get_source", lambda name: FakeYFinance())

    result = ingestion.get_rates_curve("USD", db_path=db_file)

    assert result["source"] == "yfinance_treasury_proxy"
    assert result["tenors"]["1M"] == pytest.approx(0.052)
    with connect(db_file) as conn:
        rows = conn.execute("SELECT * FROM rates").fetchall()
    assert len(rows) == 4


def test_get_rates_curve_static_fallback_when_all_sources_fail(tmp_path, monkeypatch):
    db_file = tmp_path / "test.sqlite"
    init_db(db_file)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    class FailingYFinance:
        def treasury_yields(self):
            raise RuntimeError("network down")

    monkeypatch.setattr(ingestion, "get_source", lambda name: FailingYFinance())

    result = ingestion.get_rates_curve("USD", db_path=db_file)

    assert result["source"] == "prototype_fallback"
    assert result["tenors"]["1M"] == 0.045


def test_get_rates_curve_uses_live_fred_when_available(tmp_path, monkeypatch):
    db_file = tmp_path / "test.sqlite"
    init_db(db_file)
    monkeypatch.setenv("FRED_API_KEY", "test-key")

    class FakeFred:
        def history(self, series_id, **kwargs):
            value = {"DGS1MO": 5.00, "DGS3MO": 5.10, "DGS1": 4.80, "DGS2": 4.50}[series_id]
            return pd.DataFrame({"close": [value]})

    monkeypatch.setattr(ingestion, "get_source", lambda name: FakeFred())

    result = ingestion.get_rates_curve("USD", db_path=db_file)

    assert result["source"] == "fred"
    assert result["tenors"]["1M"] == pytest.approx(0.05)
    assert result["tenors"]["2Y"] == pytest.approx(0.045)
    with connect(db_file) as conn:
        rows = conn.execute("SELECT * FROM rates").fetchall()
    assert len(rows) == 4


def test_get_rates_curve_non_usd_uses_prototype(tmp_path, monkeypatch):
    db_file = tmp_path / "test.sqlite"
    init_db(db_file)
    monkeypatch.setenv("FRED_API_KEY", "test-key")

    result = ingestion.get_rates_curve("EUR", db_path=db_file)

    assert result["source"] == "prototype_fallback"


def test_get_option_chain_real_source_call(tmp_path, monkeypatch):
    import yfinance

    db_file = tmp_path / "test.sqlite"
    init_db(db_file)
    future_expiry = (pd.Timestamp.now() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    class FakeChain:
        calls = pd.DataFrame([{
            "strike": 100.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1,
            "impliedVolatility": 0.25, "volume": 10, "openInterest": 100,
        }])
        puts = pd.DataFrame([{
            "strike": 100.0, "bid": 0.9, "ask": 1.1, "lastPrice": 1.0,
            "impliedVolatility": 0.27, "volume": 8, "openInterest": 90,
        }])

    class FakeTicker:
        options = (future_expiry,)

        def history(self, period="1d"):
            return pd.DataFrame({"Close": [123.45]})

        def option_chain(self, expiry):
            return FakeChain()

    monkeypatch.setattr(yfinance, "Ticker", lambda symbol: FakeTicker())

    result = ingestion.get_option_chain("aapl", max_expiries=3, db_path=db_file)

    assert result["symbol"] == "AAPL"
    assert result["spot"] == 123.45
    assert result["expiries"] == [future_expiry]
    assert result["chains"][0]["calls"][0]["strike"] == 100.0
    with connect(db_file) as conn:
        rows = conn.execute("SELECT * FROM option_chains").fetchall()
    assert len(rows) == 1
