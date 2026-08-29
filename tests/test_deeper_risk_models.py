from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.routes.vanilla_option as vanilla_option_routes
from api.main import app
from core.calibration.calibrator import CalibrationResult
from core.calibration.ssvi_surface import calibrate_ssvi_surface_from_chain
from core.calibration.store import CalibrationStore
from core.models.merton import MertonParams, default_params, price as merton_price
from ml.ssvi import SSVIParams
from options.black_scholes import price as bs_price
from options.engine import PricingEngine
from volatility.local_vol import LocalVolSurface, mc_price_local_vol


# ── core/models/merton.py ────────────────────────────────────────────────────

def test_merton_converges_to_black_scholes_at_zero_intensity():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    bs = bs_price(S, K, T, r, sigma, "call")
    merton_no_jumps = merton_price(S, K, T, r, sigma, MertonParams(0.0, -0.05, 0.10), "call")
    assert merton_no_jumps == pytest.approx(bs, abs=1e-8)


def test_merton_negative_mean_jump_raises_call_price_vs_pure_diffusion():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    bs = bs_price(S, K, T, r, sigma, "call")
    with_jumps = merton_price(S, K, T, r, sigma, default_params(), "call")
    assert with_jumps > bs  # added tail variance increases option value


# ── core/calibration/ssvi_surface.py ────────────────────────────────────────

def test_ssvi_surface_calibration_recovers_known_params():
    true_ssvi = SSVIParams(rho=-0.3, eta=1.0, gamma=0.4)
    T_nodes = [0.1, 0.3, 0.6, 1.0]
    spot, r = 100.0, 0.04
    strikes, maturities, ivs = [], [], []
    for T in T_nodes:
        theta = 0.04 * T  # 20% ATM vol term structure
        F = spot * np.exp(r * T)
        for k in np.linspace(-0.3, 0.3, 9):
            K = F * np.exp(k)
            w = true_ssvi.total_var(np.array([k]), theta)[0]
            strikes.append(K)
            maturities.append(T)
            ivs.append(np.sqrt(max(w, 1e-8) / T))

    result = calibrate_ssvi_surface_from_chain(
        spot, np.array(strikes), np.array(maturities), np.array(ivs), r=r
    )
    assert result.rho == pytest.approx(true_ssvi.rho, abs=0.05)
    assert result.rmse < 1e-4
    assert result.arb_free is True


def test_ssvi_surface_calibration_requires_multiple_maturities():
    with pytest.raises(ValueError):
        calibrate_ssvi_surface_from_chain(100.0, np.array([95, 100, 105]), np.array([0.5, 0.5, 0.5]), np.array([0.2, 0.2, 0.2]))


# ── volatility/local_vol.py pricing ─────────────────────────────────────────

def test_local_vol_mc_price_is_positive_and_near_atm_bs():
    ssvi = SSVIParams(rho=-0.2, eta=1.0, gamma=0.4)
    T_nodes = np.array([0.25, 0.5, 1.0])
    theta_nodes = np.array([0.04, 0.08, 0.16]) * T_nodes  # flat 20% vol term structure
    lv = LocalVolSurface(ssvi, T_nodes, theta_nodes)

    S, K, T, r = 100, 100, 0.5, 0.04
    px, se = mc_price_local_vol(S, K, T, r, lv, option="call", n_sims=40_000, n_steps=50, seed=7)
    assert px > 0
    atm_iv = lv.atm_iv(T) if hasattr(lv, "atm_iv") else np.sqrt(lv.theta(T) / T)
    bs = bs_price(S, K, T, r, atm_iv, "call")
    assert px == pytest.approx(bs, abs=4 * se + 0.5)


# ── Heston vega_v0 / rho ─────────────────────────────────────────────────────

def test_heston_vega_rho_present_and_finite(tmp_path, monkeypatch):
    db_path = str(tmp_path / "calib.sqlite")
    monkeypatch.setattr(vanilla_option_routes, "_DB_PATH", db_path)

    eng = PricingEngine(model="heston", db_path=db_path, source="synthetic")
    eng.update("SYNTH_HESTON_TEST")

    extra = vanilla_option_routes._heston_vega_rho(eng, "SYNTH_HESTON_TEST", 100, 0.5, "call", 100.0, 0.0)
    assert extra is not None
    assert np.isfinite(extra["vega_v0"])
    assert np.isfinite(extra["rho"])


# ── API endpoints ────────────────────────────────────────────────────────────

def test_api_price_option_merton():
    client = TestClient(app)
    res = client.post("/api/price/option", json={"spot": 100, "strike": 100, "model": "merton"})
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "merton"
    assert body["price"] > 0
    assert body["jump_intensity"] == 2.0


def test_api_price_option_local_vol_not_calibrated_returns_409():
    client = TestClient(app)
    res = client.post(
        "/api/price/option",
        json={"underlying": "NEVER_CALIBRATED_LV_TICKER", "spot": 100, "strike": 100, "model": "local_vol"},
    )
    assert res.status_code == 409
    assert "calibrate" in res.json()["detail"].lower()


def test_api_calibrate_and_price_local_vol_round_trip(tmp_path, monkeypatch):
    db_path = str(tmp_path / "calib_lv.sqlite")
    monkeypatch.setattr(vanilla_option_routes, "_DB_PATH", db_path)

    true_ssvi = SSVIParams(rho=-0.25, eta=1.2, gamma=0.35)
    T_nodes = [0.2, 0.5, 0.9]
    spot, r = 100.0, 0.04
    strikes, maturities, ivs, weights = [], [], [], []
    for T in T_nodes:
        theta = 0.05 * T
        F = spot * np.exp(r * T)
        for k in np.linspace(-0.25, 0.25, 7):
            K = F * np.exp(k)
            w = true_ssvi.total_var(np.array([k]), theta)[0]
            strikes.append(K)
            maturities.append(T)
            ivs.append(np.sqrt(max(w, 1e-8) / T))
            weights.append(1.0)

    def fake_load_chain(ticker, r_, moneyness_range, min_volume, max_expiries):
        return spot, np.array(strikes), np.array(maturities), np.array(ivs), np.array(weights)

    monkeypatch.setattr(vanilla_option_routes, "_load_chain_yfinance", fake_load_chain)

    client = TestClient(app)
    calib_res = client.post("/api/calibrate/SYNTH_LV_TICKER?model=local_vol")
    assert calib_res.status_code == 200
    body = calib_res.json()
    assert body["model"] == "local_vol"
    assert body["arb_free"] is True

    status_res = client.get("/api/calibration/SYNTH_LV_TICKER?model=local_vol")
    assert status_res.json()["calibrated"] is True

    price_res = client.post(
        "/api/price/option",
        json={"underlying": "SYNTH_LV_TICKER", "spot": spot, "strike": 100, "maturity_years": 0.5, "model": "local_vol", "n_sims": 4000},
    )
    assert price_res.status_code == 200
    priced = price_res.json()
    assert priced["price"] > 0
    assert "delta" in priced["greeks"]


def test_api_compare_option_models_includes_merton_and_local_vol_note():
    client = TestClient(app)
    res = client.post(
        "/api/price/option/compare",
        json={"underlying": "NEVER_CALIBRATED_LV_TICKER_2", "spot": 100, "strike": 100, "n_sims": 2000},
    )
    assert res.status_code == 200
    models = {r["model"]: r for r in res.json()["results"]}
    assert models["merton"]["price"] > 0
    assert models["local_vol"]["calibrated"] is False
