"""
api/main.py
FastAPI service exposing a fitted neural volatility surface.

The service is deliberately thin. Fitting a surface takes seconds and holds a
torch graph; serving one is a lookup. So `POST /api/surface/fit` trains a
surface for a ticker and caches it in memory under a handle, and every other
endpoint reads that cached surface. That split is also what a real vol service
looks like: calibration runs on a schedule, pricing runs on every request.

    uvicorn api.main:app --reload
    open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.pricing import router as pricing_router
from api.routes.surface import router as surface_router

app = FastAPI(
    title="Neural Volatility Surface",
    version="1.0.0",
    description=(
        "Fits an arbitrage-penalised neural implied-volatility surface to a live "
        "option chain and serves implied vols, prices and no-arbitrage diagnostics "
        "off it."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "neural-vol-surface"}


app.include_router(surface_router)
app.include_router(pricing_router)
