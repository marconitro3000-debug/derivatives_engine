from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import WorkflowRequest
from api.routes.pricing import build_spec
from structured.validation import validate_phoenix_spec

router = APIRouter(prefix="/api/validate", tags=["validation"])


@router.post("/workflow")
def validate_workflow(req: WorkflowRequest):
    try:
        product_validation = validate_phoenix_spec(build_spec(req.product_spec))
        graph_complete = bool(req.nodes and req.edges)
        return {
            "product_type": req.product_type,
            "graph_complete": graph_complete,
            "product_validation": product_validation,
            "status": "ok" if graph_complete and product_validation["status"] == "ok" else "warning",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
