"""
api/routes/surface.py
Fit a surface, then query it.

Fitted surfaces live in a process-local dictionary keyed by a handle. That is
the right scope for a demo service and the wrong one for production -- a real
deployment would persist the calibrated weights and the chain snapshot to a
store so that a restart, or a second worker, does not lose the calibration. The
handle indirection is here so that swap is a change of one module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import (
    ArbitrageResponse,
    FitMetrics,
    FitRequest,
    FitResponse,
    IVQuery,
    IVResponse,
    SurfaceGridResponse,
    SurfacePriceRequest,
    SurfacePriceResponse,
)
from core.diagnostics import butterfly_g, scan_arbitrage
from marketdata import fetch_chain, synthetic_snapshot
from nn import TrainConfig, compare, train_surface

router = APIRouter(prefix="/api/surface", tags=["surface"])


@dataclass
class FittedSurface:
    snapshot: object
    result: object


_CACHE: dict[str, FittedSurface] = {}


def _get(handle: str) -> FittedSurface:
    if handle not in _CACHE:
        raise HTTPException(
            status_code=404,
            detail=f"unknown handle {handle!r}; POST /api/surface/fit first",
        )
    return _CACHE[handle]


@router.post("/fit", response_model=FitResponse)
def fit(req: FitRequest) -> FitResponse:
    """Fit the neural surface to a chain and score it against the baselines."""
    try:
        snapshot = (
            synthetic_snapshot(ticker=req.ticker, noise_bps=25)
            if req.synthetic
            else fetch_chain(req.ticker, max_expiries=req.max_expiries)
        )
    except Exception as exc:                                # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"chain unavailable: {exc}") from exc

    try:
        result = train_surface(snapshot, TrainConfig(epochs=req.epochs), prior=req.prior)
    except Exception as exc:                                # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"calibration failed: {exc}") from exc

    handle = f"{snapshot.ticker.upper()}:{snapshot.asof}"
    _CACHE[handle] = FittedSurface(snapshot=snapshot, result=result)

    metrics = [
        FitMetrics(
            surface=s.name,
            iv_rmse_bps=s.fit.rmse_vol_bps,
            iv_max_error_bps=s.fit.max_vol_bps,
            price_rmse=s.fit.rmse_price,
            pct_inside_spread=s.fit.pct_inside_spread,
            calendar_violation_pct=s.arb.calendar_pct,
            butterfly_violation_pct=s.arb.butterfly_pct,
        )
        for s in compare(snapshot, result)
    ]

    return FitResponse(
        handle=handle,
        ticker=snapshot.ticker,
        asof=str(snapshot.asof),
        spot=snapshot.spot,
        n_quotes=len(snapshot),
        n_expiries=len(snapshot.forwards),
        train_seconds=round(result.elapsed_sec, 2),
        val_iv_rmse_bps=round(result.best_val_rmse_bps, 2),
        metrics=metrics,
    )


@router.get("/handles", response_model=list[str])
def handles() -> list[str]:
    return sorted(_CACHE)


@router.post("/iv", response_model=IVResponse)
def implied_vol(query: IVQuery) -> IVResponse:
    """Implied vol at arbitrary ``(k, T)``, with both no-arbitrage quantities.

    The diagnostics ship with the quote rather than behind a separate call: a
    consumer pricing an exotic off this surface should be able to see, at the
    point they are pricing, whether the surface is locally sound.
    """
    fitted = _get(query.handle)
    k = np.asarray(query.log_moneyness, dtype=float)
    T = np.asarray(query.maturity, dtype=float)
    if k.shape != T.shape:
        raise HTTPException(status_code=422, detail="k and T must be the same length")

    surf = fitted.result.model
    return IVResponse(
        implied_vol=surf.implied_vol(k, T).tolist(),
        total_variance=surf.total_variance(k, T).tolist(),
        dw_dT=surf.dw_dT(k, T).tolist(),
        butterfly_g=butterfly_g(surf, k, T).tolist(),
    )


@router.get("/grid", response_model=SurfaceGridResponse)
def grid(handle: str, n_k: int = 41, n_T: int = 21) -> SurfaceGridResponse:
    """A dense IV grid over the fitted chain's own range -- for plotting."""
    fitted = _get(handle)
    snap = fitted.snapshot
    ks = np.linspace(float(snap.k.min()), float(snap.k.max()), n_k)
    Ts = np.linspace(float(snap.T.min()), float(snap.T.max()), n_T)
    kk, TT = np.meshgrid(ks, Ts, indexing="ij")
    iv = fitted.result.model.implied_vol(kk.ravel(), TT.ravel()).reshape(kk.shape)
    return SurfaceGridResponse(
        log_moneyness=ks.tolist(), maturity=Ts.tolist(), implied_vol=iv.tolist()
    )


@router.get("/arbitrage", response_model=ArbitrageResponse)
def arbitrage(handle: str) -> ArbitrageResponse:
    """Rescan the fitted surface for static arbitrage on a dense grid."""
    fitted = _get(handle)
    snap = fitted.snapshot
    rep = scan_arbitrage(
        fitted.result.model,
        k_range=(float(snap.k.min()) * 1.2, float(snap.k.max()) * 1.2),
        T_range=(float(snap.T.min()) * 0.5, float(snap.T.max()) * 1.2),
    )
    return ArbitrageResponse(
        surface=fitted.result.model.name,
        grid_points=rep.n_grid,
        calendar_violations=rep.calendar_violations,
        butterfly_violations=rep.butterfly_violations,
        worst_dw_dT=rep.worst_calendar,
        worst_butterfly_g=rep.worst_butterfly,
        arbitrage_free=rep.is_arbitrage_free,
    )


@router.post("/price", response_model=SurfacePriceResponse)
def price_off_surface(req: SurfacePriceRequest) -> SurfacePriceResponse:
    """Price a listed or unlisted strike off the fitted surface.

    The forward and discount factor are the ones implied from put-call parity at
    fitting time, interpolated in maturity -- so a price out of this endpoint is
    consistent with the market's own forward rather than with an assumed rate.
    """
    fitted = _get(req.handle)
    snap = fitted.snapshot

    Ts = np.array(sorted(snap.forwards))
    F = float(np.interp(req.maturity, Ts, [snap.forwards[t] for t in Ts]))
    DF = float(np.interp(req.maturity, Ts, [snap.discounts[t] for t in Ts]))

    k = float(np.log(req.strike / F))
    sigma = float(fitted.result.model.implied_vol(np.array([k]), np.array([req.maturity]))[0])

    from options.black_scholes import price as bs_price

    # S = F with r = 0 is the forward measure; discount once at the end.
    px = DF * bs_price(F, req.strike, req.maturity, 0.0, sigma, req.option)

    return SurfacePriceResponse(
        price=px,
        implied_vol=sigma,
        forward=F,
        discount_factor=DF,
        log_moneyness=k,
    )
