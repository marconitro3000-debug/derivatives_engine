"""
api/routes/pricing.py
Closed-form Black-Scholes pricing, independent of any fitted surface.

Kept alongside the surface endpoints because it is the reference the surface is
built on: an IV out of `/api/surface/iv` is only meaningful together with the
pricing formula it inverts. Useful as a sanity check when a surface price looks
wrong -- feed the surface's vol in here and confirm the two agree.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import PriceRequest, PriceResponse
from options.black_scholes import greeks, price

router = APIRouter(prefix="/api/price", tags=["pricing"])


@router.post("/black-scholes", response_model=PriceResponse)
def black_scholes(req: PriceRequest) -> PriceResponse:
    """European option price and Greeks under Black-Scholes with a dividend yield.

    Vega is per 1% of volatility, theta per calendar day, rho per 1% of rate --
    the scalings a trader reads, not the raw partial derivatives.
    """
    try:
        px = price(req.spot, req.strike, req.maturity, req.rate, req.vol,
                   req.option, req.dividend_yield)
        g = greeks(req.spot, req.strike, req.maturity, req.rate, req.vol,
                   req.dividend_yield)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    is_call = req.option == "call"
    return PriceResponse(
        price=px,
        delta=g["delta_call"] if is_call else g["delta_put"],
        gamma=g["gamma"],
        vega=g["vega"],
        theta=g["theta_call"] if is_call else g["theta_put"],
        rho=g["rho_call"] if is_call else g["rho_put"],
    )
