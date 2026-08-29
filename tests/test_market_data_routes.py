from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

import api.routes.market_data as market_data_routes
from api.main import app
from ml.ssvi import SVIParams


def test_market_rates_endpoint(monkeypatch):
    monkeypatch.setattr(
        market_data_routes,
        "get_rates_curve",
        lambda currency: {"currency": currency, "source": "prototype_fallback", "tenors": {"1M": 0.045}},
    )
    client = TestClient(app)
    res = client.get("/api/market/rates/USD")
    assert res.status_code == 200
    assert res.json()["tenors"]["1M"] == 0.045


def test_market_option_chain_endpoint(monkeypatch):
    monkeypatch.setattr(
        market_data_routes,
        "get_option_chain",
        lambda symbol, max_expiries=6: {
            "symbol": symbol.upper(), "spot": 100.0, "source": "yfinance", "expiries": [], "chains": [],
        },
    )
    client = TestClient(app)
    res = client.get("/api/market/option-chain/AAPL")
    assert res.status_code == 200
    assert res.json()["symbol"] == "AAPL"


def test_market_vol_surface_endpoint(monkeypatch):
    @dataclass
    class FakeSliceFit:
        params: SVIParams
        rmse_iv: float
        n_points: int
        arb_free: bool

    class FakeResult:
        spot = 100.0
        scan_time = "2026-07-10 00:00:00"
        slices = {1.0: FakeSliceFit(params=SVIParams(a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.2), rmse_iv=0.001, n_points=20, arb_free=True)}
        calendar_violations: list = []
        butterfly_violations: list = []
        is_arbitrage_free = True

    class FakeScanner:
        def scan_live(self, symbol, r=0.04, max_expiries=6):
            return FakeResult()

    monkeypatch.setattr(market_data_routes, "VolSurfaceArbScanner", lambda: FakeScanner())
    client = TestClient(app)
    res = client.get("/api/market/vol-surface/AAPL")
    assert res.status_code == 200
    body = res.json()
    assert body["symbol"] == "AAPL"
    assert body["is_arbitrage_free"] is True
    assert body["slices"][0]["arb_free"] is True
    assert len(body["slices"][0]["smile"]) == 41
