"""API contract tests.

Every test uses ``synthetic: true`` so the suite never depends on Yahoo Finance
being up, on market hours, or on the shape of today's chain.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(scope="module")
def handle():
    r = client.post("/api/surface/fit",
                    json={"ticker": "TEST", "synthetic": True, "epochs": 150})
    assert r.status_code == 200, r.text
    return r.json()["handle"]


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_fit_returns_all_three_surfaces_scored(handle):
    body = client.post("/api/surface/fit",
                       json={"ticker": "TEST", "synthetic": True, "epochs": 150}).json()
    names = {m["surface"] for m in body["metrics"]}
    assert "Neural (SSVI prior + penalties)" in names
    assert body["n_quotes"] > 0 and body["n_expiries"] >= 2
    for m in body["metrics"]:
        assert m["iv_rmse_bps"] >= 0
        assert 0.0 <= m["calendar_violation_pct"] <= 100.0


def test_iv_endpoint_returns_the_arbitrage_diagnostics_with_the_quote(handle):
    r = client.post("/api/surface/iv", json={
        "handle": handle, "log_moneyness": [-0.1, 0.0, 0.1], "maturity": [0.5, 0.5, 0.5],
    })
    body = r.json()
    assert len(body["implied_vol"]) == 3
    assert all(v > 0 for v in body["implied_vol"])
    assert all(w > 0 for w in body["total_variance"])
    assert all(g > 0 for g in body["butterfly_g"])      # fitted surface is clean
    assert all(d > 0 for d in body["dw_dT"])


def test_iv_endpoint_rejects_mismatched_lengths(handle):
    r = client.post("/api/surface/iv",
                    json={"handle": handle, "log_moneyness": [0.0, 0.1], "maturity": [0.5]})
    assert r.status_code == 422


def test_unknown_handle_is_a_404():
    r = client.post("/api/surface/iv",
                    json={"handle": "NOPE:2020-01-01", "log_moneyness": [0.0],
                          "maturity": [0.5]})
    assert r.status_code == 404
    assert "fit" in r.json()["detail"]


def test_grid_endpoint_shape(handle):
    body = client.get(f"/api/surface/grid?handle={handle}&n_k=7&n_T=5").json()
    assert len(body["log_moneyness"]) == 7
    assert len(body["implied_vol"]) == 7 and len(body["implied_vol"][0]) == 5


def test_arbitrage_endpoint_reports_a_clean_surface(handle):
    body = client.get(f"/api/surface/arbitrage?handle={handle}").json()
    assert body["grid_points"] > 0
    assert body["arbitrage_free"] is True


def test_price_off_surface_is_consistent_with_black_scholes(handle):
    """The endpoint's price must equal BS at the vol it reports."""
    from options.black_scholes import price as bs_price

    body = client.post("/api/surface/price", json={
        "handle": handle, "strike": 100.0, "maturity": 0.5, "option": "call",
    }).json()

    expected = body["discount_factor"] * bs_price(
        body["forward"], 100.0, 0.5, 0.0, body["implied_vol"], "call"
    )
    assert body["price"] == pytest.approx(expected, rel=1e-9)
    assert body["price"] > 0


def test_black_scholes_endpoint():
    body = client.post("/api/price/black-scholes", json={
        "spot": 100.0, "strike": 100.0, "maturity": 1.0,
        "rate": 0.03, "vol": 0.2, "option": "call",
    }).json()
    assert 0 < body["delta"] < 1
    assert body["gamma"] > 0 and body["vega"] > 0
    assert body["theta"] < 0


def test_black_scholes_endpoint_validates_its_inputs():
    r = client.post("/api/price/black-scholes", json={
        "spot": 100.0, "strike": 100.0, "maturity": 1.0,
        "rate": 0.03, "vol": 0.2, "option": "straddle",
    })
    assert r.status_code == 422
