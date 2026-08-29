from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from api.schemas import PortfolioHedgeRequest
from portfolio.correlation import estimate_correlation
from portfolio.hedge_optimizer import Position, optimize_hedge

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


@router.post("/hedge")
def hedge_portfolio(req: PortfolioHedgeRequest):
    tickers = [p.ticker.upper() for p in req.positions]
    corr = estimate_correlation(tickers)
    if corr.missing:
        raise HTTPException(
            status_code=502,
            detail=f"No live historical data available for: {', '.join(corr.missing)}",
        )

    notionals_by_ticker = {p.ticker.upper(): p.notional for p in req.positions}
    positions = [Position(ticker=t, notional=notionals_by_ticker[t]) for t in corr.tickers]

    try:
        result = optimize_hedge(
            positions=positions,
            volatility=corr.volatility,
            correlation=corr.correlation,
            max_drawdown_target_pct=req.max_drawdown_target_pct,
            hedge_strike_pct=req.hedge_strike_pct,
            horizon_years=req.horizon_years,
            risk_free_rate=req.risk_free_rate,
            n_paths=req.n_paths,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "tickers": result.tickers,
        "hedge_ratio": result.hedge_ratio,
        "target_met": result.target_met,
        "premium_cost": result.premium_cost,
        "premium_cost_pct": result.premium_cost_pct,
        "correlation": result.correlation,
        "volatility": result.volatility,
        "unhedged": asdict(result.unhedged),
        "hedged": asdict(result.hedged),
        "frontier": [asdict(p) for p in result.frontier],
        "lookback_days": corr.lookback_days,
    }
