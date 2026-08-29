from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.routes.vanilla_option as vanilla_option_routes
from api.main import app
from options.binomial_tree import binomial_price
from options.black_scholes import price as bs_price
from options.monte_carlo import mc_price


def test_binomial_converges_to_black_scholes():
    bs = bs_price(100, 100, 1, 0.05, 0.2, "call")
    tree = binomial_price(100, 100, 1, 0.05, 0.2, "call", "european", n_steps=400)
    assert tree["price"] == pytest.approx(bs, abs=0.05)
    assert tree["early_exercise"] == 0.0


def test_binomial_american_put_has_early_exercise_premium():
    euro = binomial_price(100, 110, 1, 0.05, 0.25, "put", "european", n_steps=300)
    amer = binomial_price(100, 110, 1, 0.05, 0.25, "put", "american", n_steps=300)
    assert amer["early_exercise"] >= 0
    assert amer["price"] >= euro["price"]


def test_monte_carlo_matches_black_scholes_within_confidence_interval():
    bs = bs_price(100, 100, 1, 0.05, 0.2, "call")
    mc = mc_price(100, 100, 1, 0.05, 0.2, "european_call", n_sims=50_000, seed=7)
    assert mc["conf_95_lo"] - 0.5 <= bs <= mc["conf_95_hi"] + 0.5


def test_api_price_option_black_scholes():
    client = TestClient(app)
    res = client.post("/api/price/option", json={"spot": 100, "strike": 100, "model": "black_scholes"})
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "black_scholes"
    assert body["price"] > 0
    assert "delta" in body["greeks"]


def test_api_price_option_binomial():
    client = TestClient(app)
    res = client.post(
        "/api/price/option",
        json={"spot": 100, "strike": 100, "model": "binomial", "style": "american", "n_steps": 50},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "binomial"
    assert body["style"] == "american"


def test_api_price_option_monte_carlo():
    client = TestClient(app)
    res = client.post(
        "/api/price/option",
        json={"spot": 100, "strike": 100, "model": "monte_carlo", "n_sims": 2000},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "monte_carlo"
    assert "std_error" in body


def test_api_price_option_heston_not_calibrated_returns_409():
    client = TestClient(app)
    res = client.post(
        "/api/price/option",
        json={"underlying": "NEVER_CALIBRATED_TICKER_XYZ", "spot": 100, "strike": 100, "model": "heston"},
    )
    assert res.status_code == 409
    assert "calibrate" in res.json()["detail"].lower()


def test_api_compare_option_models_includes_fast_models_and_flags_uncalibrated():
    client = TestClient(app)
    res = client.post(
        "/api/price/option/compare",
        json={"underlying": "NEVER_CALIBRATED_TICKER_XYZ", "spot": 100, "strike": 100, "n_sims": 2000},
    )
    assert res.status_code == 200
    models = {r["model"]: r for r in res.json()["results"]}
    assert models["black_scholes"]["price"] > 0
    assert models["binomial"]["price"] > 0
    assert models["monte_carlo"]["price"] > 0
    assert models["heston"]["calibrated"] is False
    assert models["svi"]["calibrated"] is False


def test_calibrate_and_price_heston_round_trip_with_synthetic_data(tmp_path, monkeypatch):
    db_path = str(tmp_path / "calib_test.sqlite")
    monkeypatch.setattr(vanilla_option_routes, "_DB_PATH", db_path)

    client = TestClient(app)
    ticker = "SYNTH_TEST_TICKER"

    calib_res = client.post(f"/api/calibrate/{ticker}?model=heston&source=synthetic")
    assert calib_res.status_code == 200
    body = calib_res.json()
    assert body["model"] == "heston"
    assert body["rmse"] < 1.0

    status_res = client.get(f"/api/calibration/{ticker}?model=heston")
    assert status_res.status_code == 200
    assert status_res.json()["calibrated"] is True

    price_res = client.post(
        "/api/price/option",
        json={"underlying": ticker, "spot": 100, "strike": 105, "maturity_years": 0.5, "model": "heston"},
    )
    assert price_res.status_code == 200
    priced = price_res.json()
    assert priced["price"] > 0
    assert "delta" in priced["greeks"]
    assert "vega" not in priced["greeks"]
