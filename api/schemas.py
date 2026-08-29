"""api/schemas.py — request and response models for the surface service."""

from __future__ import annotations

from pydantic import BaseModel, Field


# -- fitting ------------------------------------------------------------------

class FitRequest(BaseModel):
    ticker: str = Field(..., description="Underlying symbol, e.g. SPY")
    max_expiries: int = Field(8, ge=2, le=20)
    epochs: int = Field(1500, ge=100, le=20_000)
    prior: str = Field("ssvi", pattern="^(ssvi|flat)$")
    synthetic: bool = Field(
        False,
        description="Fit a generated arbitrage-free chain instead of live data. "
                    "Lets the service be exercised without network access.",
    )


class FitMetrics(BaseModel):
    surface: str
    iv_rmse_bps: float
    iv_max_error_bps: float
    price_rmse: float
    pct_inside_spread: float
    calendar_violation_pct: float
    butterfly_violation_pct: float


class FitResponse(BaseModel):
    handle: str = Field(..., description="Identifier to pass to the query endpoints")
    ticker: str
    asof: str
    spot: float
    n_quotes: int
    n_expiries: int
    train_seconds: float
    val_iv_rmse_bps: float
    metrics: list[FitMetrics] = Field(
        ..., description="Neural surface and parametric baselines, scored identically"
    )


# -- querying -----------------------------------------------------------------

class IVQuery(BaseModel):
    handle: str
    log_moneyness: list[float] = Field(..., description="k = log(K / F_T)")
    maturity: list[float] = Field(..., description="Years to expiry, aligned with k")


class IVResponse(BaseModel):
    implied_vol: list[float]
    total_variance: list[float]
    dw_dT: list[float] = Field(..., description="Calendar condition: must be >= 0")
    butterfly_g: list[float] = Field(..., description="Durrleman g: must be >= 0")


class SurfaceGridResponse(BaseModel):
    log_moneyness: list[float]
    maturity: list[float]
    implied_vol: list[list[float]] = Field(..., description="Row-major [k][T] grid")


class ArbitrageResponse(BaseModel):
    surface: str
    grid_points: int
    calendar_violations: int
    butterfly_violations: int
    worst_dw_dT: float
    worst_butterfly_g: float
    arbitrage_free: bool


# -- pricing ------------------------------------------------------------------

class PriceRequest(BaseModel):
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    maturity: float = Field(..., gt=0, description="Years")
    rate: float = 0.04
    vol: float = Field(..., gt=0)
    dividend_yield: float = 0.0
    option: str = Field("call", pattern="^(call|put)$")


class PriceResponse(BaseModel):
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class SurfacePriceRequest(BaseModel):
    handle: str
    strike: float = Field(..., gt=0)
    maturity: float = Field(..., gt=0)
    option: str = Field("call", pattern="^(call|put)$")


class SurfacePriceResponse(BaseModel):
    price: float
    implied_vol: float
    forward: float
    discount_factor: float
    log_moneyness: float
