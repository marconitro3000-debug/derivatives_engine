from __future__ import annotations

from fastapi import APIRouter

from api.schemas import TermSheetRequest

router = APIRouter(prefix="/api/term-sheet", tags=["term-sheet"])


@router.post("/phoenix")
def phoenix_term_sheet(req: TermSheetRequest):
    coupon = req.coupon_pa if req.coupon_pa is not None else req.coupon_rate
    barrier = req.capital_barrier if req.capital_barrier is not None else req.barrier
    return {
        "title": "Autocall Note",
        "issuer": req.issuer,
        "currency": req.currency,
        "underlying": req.underlying,
        "notional": req.notional,
        "maturity_years": req.maturity_years,
        "coupon": coupon,
        "autocall_level": req.autocall_level,
        "capital_barrier": barrier,
        "memory_coupon": req.memory,
        "risk_note": "Capital is at risk if the barrier condition is breached and the final underlying level is below the initial level.",
        "scope": "Research-oriented educational prototype, not investment advice.",
    }
