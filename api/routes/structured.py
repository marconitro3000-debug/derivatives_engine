from __future__ import annotations

from fastapi import APIRouter

from api.schemas import PhoenixRequest
from api.routes.pricing import build_spec
from structured.stress import run_stress_scenarios

router = APIRouter(prefix="/api/stress", tags=["structured"])


@router.post("/phoenix")
def stress_phoenix(req: PhoenixRequest):
    return run_stress_scenarios(build_spec(req))
