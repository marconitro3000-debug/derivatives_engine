from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint():
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_price_phoenix_endpoint():
    client = TestClient(app)
    res = client.post(
        "/api/price/phoenix?include_greeks=false&include_stress=false&include_heatmap=false",
        json={"spot": 100, "notional": 1000, "n_sims": 1_000, "steps_per_year": 40, "volatility": 0.2},
    )
    assert res.status_code == 200
    body = res.json()
    assert "fair_value" in body
    assert 0 <= body["autocall_probability"] <= 1
    assert 0 <= body["barrier_touch_probability"] <= 1
