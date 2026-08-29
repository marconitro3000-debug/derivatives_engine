from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.routes.vanilla_option as vanilla_option_routes
from api.main import app
from core.calibration.calibrator import CalibrationResult
from core.calibration.store import CalibrationStore
from ml.ssvi import SSVIParams
from options.binomial_tree import binomial_price
from options.black_scholes import price as bs_price
from options.pde import price as pde_price
from volatility.local_vol import LocalVolSurface


# ── options/pde.py — validated against known closed forms ──────────────────

def test_pde_european_call_matches_black_scholes():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    bs = bs_price(S, K, T, r, sigma, "call")
    pde = pde_price(S, K, T, r, sigma, "call", "european")["price"]
    assert pde == pytest.approx(bs, rel=0.005)


def test_pde_european_put_matches_black_scholes():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    bs = bs_price(S, K, T, r, sigma, "put")
    pde = pde_price(S, K, T, r, sigma, "put", "european")["price"]
    assert pde == pytest.approx(bs, rel=0.005)


def test_pde_american_call_equals_european_when_no_dividend():
    """Classic result: an American call on a non-dividend-paying stock is
    never optimally exercised early, so it must equal the European price."""
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    euro = pde_price(S, K, T, r, sigma, "call", "european", q=0.0)["price"]
    amer = pde_price(S, K, T, r, sigma, "call", "american", q=0.0)["price"]
    assert amer == pytest.approx(euro, abs=1e-9)


def test_pde_american_put_matches_binomial_tree():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    pde_amer = pde_price(S, K, T, r, sigma, "put", "american")["price"]
    tree_amer = binomial_price(S, K, T, r, sigma, "put", "american", n_steps=500)["price"]
    assert pde_amer == pytest.approx(tree_amer, rel=0.01)


def test_pde_american_put_has_early_exercise_premium():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    euro = pde_price(S, K, T, r, sigma, "put", "european")["price"]
    amer = pde_price(S, K, T, r, sigma, "put", "american")["price"]
    assert amer >= euro


def test_pde_matches_black_scholes_across_maturities():
    S, K, r, sigma = 100, 100, 0.05, 0.2
    for T in (0.1, 0.5, 1.0, 2.0):
        bs = bs_price(S, K, T, r, sigma, "call")
        pde = pde_price(S, K, T, r, sigma, "call", "european")["price"]
        assert pde == pytest.approx(bs, rel=0.005), f"T={T}"


def test_pde_accepts_local_vol_callable_sigma():
    ssvi = SSVIParams(rho=-0.2, eta=1.0, gamma=0.4)
    T_nodes = np.array([0.25, 0.5, 1.0])
    theta_nodes = np.array([0.04, 0.08, 0.16]) * T_nodes
    lv = LocalVolSurface(ssvi, T_nodes, theta_nodes)

    S, K, T, r = 100, 100, 0.5, 0.04

    def sigma_local(x, t):
        F = S * np.exp(r * max(t, 1e-6))
        k = x - np.log(F)
        return float(lv(np.array([k]), max(t, 1e-6))[0])

    res = pde_price(S, K, T, r, sigma_local, "call", "european")
    assert res["price"] > 0


def test_pde_invalid_option_type_raises():
    with pytest.raises(ValueError):
        pde_price(100, 100, 1, 0.05, 0.2, "straddle")


def test_pde_invalid_style_raises():
    with pytest.raises(ValueError):
        pde_price(100, 100, 1, 0.05, 0.2, "call", "bermudan")


# ── API: local_vol + style=american routes through the PDE solver ──────────

def test_api_local_vol_american_round_trip(tmp_path, monkeypatch):
    db_path = str(tmp_path / "calib_pde.sqlite")
    monkeypatch.setattr(vanilla_option_routes, "_DB_PATH", db_path)

    store = CalibrationStore(db_path)
    store.save(
        "SYNTH_PDE_TICKER",
        CalibrationResult(
            model="local_vol",
            params={"rho": -0.2, "eta": 1.0, "gamma": 0.4, "T_nodes": [0.25, 0.5, 1.0], "theta_nodes": [0.01, 0.02, 0.04]},
            rmse=0.001, max_error=0.002, n_points=30, arb_free=True, elapsed_sec=0.1,
        ),
        spot=100.0, r=0.04,
    )

    client = TestClient(app)
    res = client.post(
        "/api/price/option",
        json={
            "underlying": "SYNTH_PDE_TICKER", "spot": 100, "strike": 100, "maturity_years": 0.5,
            "model": "local_vol", "style": "american", "option_type": "put",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "local_vol"
    assert body["style"] == "american"
    assert body["price"] > 0
    assert "delta" in body["greeks"]


def test_api_local_vol_american_at_least_european_price(tmp_path, monkeypatch):
    db_path = str(tmp_path / "calib_pde2.sqlite")
    monkeypatch.setattr(vanilla_option_routes, "_DB_PATH", db_path)

    store = CalibrationStore(db_path)
    store.save(
        "SYNTH_PDE_TICKER_2",
        CalibrationResult(
            model="local_vol",
            params={"rho": -0.2, "eta": 1.0, "gamma": 0.4, "T_nodes": [0.25, 0.5, 1.0], "theta_nodes": [0.01, 0.02, 0.04]},
            rmse=0.001, max_error=0.002, n_points=30, arb_free=True, elapsed_sec=0.1,
        ),
        spot=100.0, r=0.04,
    )

    client = TestClient(app)
    base = {"underlying": "SYNTH_PDE_TICKER_2", "spot": 100, "strike": 105, "maturity_years": 0.5, "model": "local_vol", "option_type": "put"}
    euro = client.post("/api/price/option", json={**base, "style": "european", "n_sims": 20000}).json()
    amer = client.post("/api/price/option", json={**base, "style": "american"}).json()
    assert amer["price"] >= euro["price"] - 0.5  # allow MC noise on the European leg
