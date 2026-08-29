from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


def model_to_dict(model: BaseModel, **kwargs: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)  # type: ignore[attr-defined]
    return model.dict(**kwargs)


class PhoenixRequest(BaseModel):
    product_type: str = "phoenix_autocall"
    underlying: str = "AAPL"
    spot: float = Field(default=175.0, gt=0)
    notional: float = Field(default=1_000_000.0, gt=0)
    maturity_years: float = Field(default=1.0, gt=0)
    coupon_pa: float | None = Field(default=None, ge=0)
    coupon_rate: float | None = Field(default=0.08, ge=0)
    coupon_frequency: int = Field(default=4, gt=0)
    coupon_barrier: float | None = Field(default=None, ge=0)
    capital_barrier: float | None = Field(default=None, ge=0)
    barrier: float | None = Field(default=0.70, ge=0)
    autocall_level: float = Field(default=1.00, gt=0)
    autocall_start: float = Field(default=0.25, ge=0)
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.0
    volatility: float = Field(default=0.28, ge=0)
    observation_dates: list[float] | None = None
    n_paths: int | None = Field(default=None, ge=100)
    n_sims: int | None = Field(default=50_000, ge=100)
    steps_per_year: int = Field(default=252, ge=12)
    memory: bool = True
    seed: int = 42
    return_paths: bool = False
    return_distributions: bool = True
    objectives: dict[str, Any] = Field(default_factory=dict)

    def to_engine_dict(self) -> dict[str, Any]:
        data = model_to_dict(self, exclude_none=True)
        if self.coupon_pa is None and self.coupon_rate is not None:
            data["coupon_pa"] = self.coupon_rate
        if self.capital_barrier is None and self.barrier is not None:
            data["capital_barrier"] = self.barrier
        if self.coupon_barrier is None:
            data["coupon_barrier"] = data.get("capital_barrier", self.barrier or 0.70)
        if self.n_paths is None and self.n_sims is not None:
            data["n_paths"] = self.n_sims
        data.pop("product_type", None)
        data.pop("coupon_rate", None)
        data.pop("barrier", None)
        data.pop("n_sims", None)
        return data


class WorkflowRequest(BaseModel):
    name: str = "Autocall Workflow"
    product_type: str = "phoenix_autocall"
    objective: dict[str, Any] = Field(default_factory=dict)
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[list[str]] = Field(default_factory=list)
    product_spec: PhoenixRequest


class TermSheetRequest(PhoenixRequest):
    issuer: str = "Demo Issuer"
    currency: str = "USD"


class VanillaOptionRequest(BaseModel):
    underlying: str = "AAPL"
    spot: float = Field(default=175.0, gt=0)
    strike: float = Field(default=175.0, gt=0)
    maturity_years: float = Field(default=1.0, gt=0)
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.0
    volatility: float = Field(default=0.28, gt=0)
    option_type: Literal["call", "put"] = "call"
    model: Literal["black_scholes", "binomial", "monte_carlo", "heston", "svi", "merton", "local_vol"] = "black_scholes"
    style: Literal["european", "american"] = "european"
    n_steps: int = Field(default=500, ge=10)
    n_sims: int = Field(default=20_000, ge=1_000)
    seed: int = 42
    jump_intensity: float = Field(default=2.0, ge=0.0)
    jump_mean: float = -0.05
    jump_vol: float = Field(default=0.10, ge=0.0)


class PortfolioPositionRequest(BaseModel):
    ticker: str
    notional: float = Field(gt=0)


class PortfolioHedgeRequest(BaseModel):
    positions: list[PortfolioPositionRequest]
    max_drawdown_target_pct: float = Field(default=15.0, gt=0)
    hedge_strike_pct: float = Field(default=0.90, gt=0, le=1.5)
    horizon_years: float = Field(default=1.0, gt=0)
    risk_free_rate: float = 0.045
    n_paths: int = Field(default=20_000, ge=1_000)
