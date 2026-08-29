from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import PhoenixRequest, model_to_dict
from db.database import init_db, insert_pricing_run
from structured.greeks import finite_difference_greeks
from structured.heatmap import spot_vol_heatmap
from structured.phoenix_autocall import PhoenixAutocallSpec, price_phoenix_autocall
from structured.stress import run_stress_scenarios
from structured.validation import validate_phoenix_spec

router = APIRouter(prefix="/api/price", tags=["pricing"])


def build_spec(req: PhoenixRequest) -> PhoenixAutocallSpec:
    return PhoenixAutocallSpec(**req.to_engine_dict())


@router.post("/phoenix")
def price_phoenix(req: PhoenixRequest, include_greeks: bool = True, include_stress: bool = True, include_heatmap: bool = True):
    try:
        spec = build_spec(req)
        result = price_phoenix_autocall(spec)
        if include_greeks:
            result["greeks"] = finite_difference_greeks(spec)
        if include_stress:
            result["stress"] = run_stress_scenarios(spec)
        if include_heatmap:
            result["heatmap"] = spot_vol_heatmap(spec)
        result["validation"] = validate_phoenix_spec(spec)
        result["issuer_price_pct"] = result["fair_value_pct"] + 0.75
        result["client_price_pct"] = result["fair_value_pct"] + 1.25
        result["bid_ask_pct"] = {"bid": result["fair_value_pct"] - 0.35, "ask": result["fair_value_pct"] + 0.35}
        init_db()
        insert_pricing_run("phoenix_autocall", model_to_dict(req), result)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
