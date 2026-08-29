from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.routes.market_data as market_data_routes
from api.main import app
from core.daycount import year_fraction
from core.models.heston import default_params
from core.models.heston import price as heston_price
from options.black_scholes import greeks as bs_greeks, price as bs_price, put_call_parity_check
from rates.curves import DiscountCurve


# ── core/daycount.py ─────────────────────────────────────────────────────────

def test_year_fraction_act365f():
    d0, d1 = date(2024, 3, 1), date(2024, 6, 1)
    expected = (d1 - d0).days / 365.0
    assert year_fraction(d0, d1, "act365f") == pytest.approx(expected)


def test_year_fraction_act360():
    d0, d1 = date(2024, 3, 1), date(2024, 6, 1)
    expected = (d1 - d0).days / 360.0
    assert year_fraction(d0, d1, "act360") == pytest.approx(expected)


def test_year_fraction_thirty360_round_months():
    assert year_fraction(date(2024, 3, 1), date(2024, 6, 1), "thirty360") == pytest.approx(0.25)


def test_year_fraction_accepts_string_dates():
    assert year_fraction("2024-01-01", "2024-01-31", "act365f") == pytest.approx(30 / 365.0)


def test_year_fraction_unknown_convention_raises():
    with pytest.raises(ValueError):
        year_fraction(date(2024, 1, 1), date(2024, 2, 1), "bogus")


# ── rates/curves.py ──────────────────────────────────────────────────────────

def test_discount_curve_from_tenor_dict_reproduces_pillars():
    curve = DiscountCurve.from_tenor_dict({"1M": 0.045, "3M": 0.046, "1Y": 0.043, "2Y": 0.041})
    assert curve.zero_rate(1.0) == pytest.approx(0.043, abs=1e-9)
    assert curve.zero_rate(2.0) == pytest.approx(0.041, abs=1e-9)


def test_discount_curve_discount_factor_decreasing_in_maturity():
    curve = DiscountCurve.from_tenor_dict({"1M": 0.045, "1Y": 0.043, "2Y": 0.041})
    assert curve.discount_factor(0.5) > curve.discount_factor(1.5) > curve.discount_factor(2.5)


def test_discount_curve_from_tenor_dict_rejects_unrecognized_tenors():
    with pytest.raises(ValueError):
        DiscountCurve.from_tenor_dict({"7Y": 0.04})


# ── options/black_scholes.py dividend yield ─────────────────────────────────

def test_black_scholes_dividend_lowers_call_raises_put():
    S, K, T, r, sigma = 100, 100, 1, 0.03, 0.2
    assert bs_price(S, K, T, r, sigma, "call", q=0.03) < bs_price(S, K, T, r, sigma, "call")
    assert bs_price(S, K, T, r, sigma, "put", q=0.03) > bs_price(S, K, T, r, sigma, "put")


def test_black_scholes_put_call_parity_with_dividend():
    S, K, T, r, sigma, q = 100, 95, 0.75, 0.04, 0.25, 0.02
    c = bs_price(S, K, T, r, sigma, "call", q)
    p = bs_price(S, K, T, r, sigma, "put", q)
    check = put_call_parity_check(S, K, T, r, c, p, q)
    assert check["error"] == pytest.approx(0, abs=1e-9)


def test_black_scholes_dividend_defaults_to_zero_backward_compatible():
    S, K, T, r, sigma = 100, 100, 1, 0.03, 0.2
    assert bs_price(S, K, T, r, sigma, "call") == bs_price(S, K, T, r, sigma, "call", q=0.0)
    assert bs_greeks(S, K, T, r, sigma) == bs_greeks(S, K, T, r, sigma, q=0.0)


# ── core/models/heston.py dividend yield ────────────────────────────────────

def test_heston_dividend_lowers_call_price():
    p = default_params()
    S, K, T, r = 100, 100, 1, 0.03
    assert heston_price(S, K, T, r, p, "call", q=0.03) < heston_price(S, K, T, r, p, "call")


def test_heston_dividend_defaults_to_zero_backward_compatible():
    p = default_params()
    S, K, T, r = 100, 100, 1, 0.03
    assert heston_price(S, K, T, r, p, "call") == pytest.approx(heston_price(S, K, T, r, p, "call", q=0.0))


# ── real dividend data source ───────────────────────────────────────────────

def test_yfinance_dividend_yield_uses_trailing_twelve_months(monkeypatch):
    import yfinance

    now = pd.Timestamp.now(tz="UTC")
    divs = pd.Series([0.25, 0.25], index=pd.DatetimeIndex([now - pd.Timedelta(days=90), now - pd.Timedelta(days=180)]))

    class FakeTicker:
        dividends = divs

        def history(self, period="5d", interval="1d"):
            return pd.DataFrame({"Close": [50.0]})

    monkeypatch.setattr(yfinance, "Ticker", lambda symbol: FakeTicker())

    from data.sources.yfinance_source import YFinanceSource

    assert YFinanceSource().dividend_yield("TEST") == pytest.approx(0.5 / 50.0)


def test_yfinance_dividend_yield_zero_when_no_dividends(monkeypatch):
    import yfinance

    class FakeTicker:
        dividends = pd.Series([], dtype=float)

        def history(self, period="5d", interval="1d"):
            return pd.DataFrame({"Close": [50.0]})

    monkeypatch.setattr(yfinance, "Ticker", lambda symbol: FakeTicker())

    from data.sources.yfinance_source import YFinanceSource

    assert YFinanceSource().dividend_yield("TEST") == 0.0


# ── API endpoints ────────────────────────────────────────────────────────────

def test_api_market_rates_interpolated(monkeypatch):
    monkeypatch.setattr(
        market_data_routes,
        "get_rates_curve",
        lambda currency: {"currency": currency, "source": "fred", "tenors": {"1M": 0.045, "1Y": 0.043, "2Y": 0.041}},
    )
    client = TestClient(app)
    res = client.get("/api/market/rates/USD?maturity_years=1.5")
    assert res.status_code == 200
    body = res.json()
    assert "interpolated_rate" in body
    assert 0.041 < body["interpolated_rate"] < 0.043


def test_api_market_rates_without_maturity_unaffected(monkeypatch):
    monkeypatch.setattr(
        market_data_routes,
        "get_rates_curve",
        lambda currency: {"currency": currency, "source": "fred", "tenors": {"1Y": 0.043}},
    )
    client = TestClient(app)
    res = client.get("/api/market/rates/USD")
    assert res.status_code == 200
    assert "interpolated_rate" not in res.json()


def test_api_market_dividend_yield(monkeypatch):
    monkeypatch.setattr(
        market_data_routes,
        "get_dividend_yield",
        lambda symbol: {"symbol": symbol.upper(), "dividend_yield": 0.015, "source": "yfinance", "window": "ttm"},
    )
    client = TestClient(app)
    res = client.get("/api/market/dividend-yield/AAPL")
    assert res.status_code == 200
    assert res.json()["dividend_yield"] == 0.015


def test_api_price_option_black_scholes_respects_dividend_yield():
    client = TestClient(app)
    res_no_div = client.post("/api/price/option", json={"spot": 100, "strike": 100, "model": "black_scholes", "dividend_yield": 0.0})
    res_div = client.post("/api/price/option", json={"spot": 100, "strike": 100, "model": "black_scholes", "dividend_yield": 0.05})
    assert res_div.json()["price"] < res_no_div.json()["price"]
