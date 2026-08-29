from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query
import numpy as np

from arbitrage.vol_surface import VolSurfaceArbScanner
from data.ingestion import get_dividend_yield, get_history, get_option_chain, get_quote, get_rates_curve
from rates.curves import DiscountCurve

router = APIRouter(prefix="/api/market", tags=["market-data"])


@router.get("/quote/{symbol}")
def market_quote(symbol: str, source: str = Query(default="auto")):
    try:
        return get_quote(symbol, source=source)  # type: ignore[arg-type]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/history/{symbol}")
def market_history(symbol: str, source: str = "auto", start: date | None = None, end: date | None = None, period: str = "1y", limit: int = 252):
    try:
        df = get_history(symbol, start=start, end=end, source=source, period=period)  # type: ignore[arg-type]
        return {
            "symbol": symbol,
            "source": source,
            "rows": len(df),
            "history": df.tail(max(1, min(limit, 2000))).reset_index().to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/calibration/{symbol}")
def market_calibration(symbol: str, source: str = "auto", period: str = "2y"):
    try:
        quote = get_quote(symbol, source=source)  # type: ignore[arg-type]
        df = get_history(symbol, source=source, period=period)  # type: ignore[arg-type]
        clean = df.copy()
        clean.columns = [str(c).lower().replace(" ", "_") for c in clean.columns]
        close_col = "adj_close" if "adj_close" in clean.columns else "close"
        closes = clean[close_col].dropna().astype(float)
        log_returns = np.log(closes / closes.shift(1)).dropna()

        def realized(window: int) -> float | None:
            if len(log_returns) < max(5, window):
                return None
            return float(log_returns.tail(window).std(ddof=1) * np.sqrt(252))

        history = clean.tail(504).reset_index()
        return {
            "symbol": symbol.upper(),
            "source": quote.get("source", source),
            "spot": float(quote["price"]),
            "currency": quote.get("currency"),
            "observed_at": quote.get("observed_at"),
            "rows": int(len(df)),
            "realized_vol_21d": realized(21),
            "realized_vol_63d": realized(63),
            "realized_vol_252d": realized(252),
            "volatility_used": realized(63) or realized(21) or 0.25,
            "history": history.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/rates/{currency}")
def market_rates(currency: str = "USD", maturity_years: float | None = Query(default=None, gt=0)):
    try:
        result = get_rates_curve(currency)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if maturity_years is not None:
        try:
            curve = DiscountCurve.from_tenor_dict(result["tenors"])
            result["interpolated_rate"] = curve.zero_rate(maturity_years)
        except Exception:
            # fall back silently to the flat-tenor response if the curve can't
            # be built (e.g. fewer than 2 tenor points) — still usable, just
            # without interpolation for this call.
            pass
    return result


@router.get("/dividend-yield/{symbol}")
def market_dividend_yield(symbol: str):
    try:
        return get_dividend_yield(symbol)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/option-chain/{symbol}")
def market_option_chain(symbol: str, max_expiries: int = 6):
    try:
        return get_option_chain(symbol, max_expiries=max_expiries)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/vol-surface/{symbol}")
def market_vol_surface(symbol: str, r: float = 0.04, max_expiries: int = 6, points: int = 41):
    try:
        result = VolSurfaceArbScanner().scan_live(symbol, r=r, max_expiries=max_expiries)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    k_grid = np.linspace(-0.4, 0.4, points)
    slices = []
    for maturity_years, slice_fit in sorted(result.slices.items()):
        ivs = slice_fit.params.implied_vol(k_grid, maturity_years)
        slices.append({
            "maturity_years": maturity_years,
            "svi_params": {
                "a": slice_fit.params.a,
                "b": slice_fit.params.b,
                "rho": slice_fit.params.rho,
                "m": slice_fit.params.m,
                "sigma": slice_fit.params.sigma,
            },
            "rmse_iv_bps": slice_fit.rmse_iv * 1e4,
            "n_points": slice_fit.n_points,
            "arb_free": slice_fit.arb_free,
            "smile": [{"log_moneyness": float(k), "implied_vol": float(v)} for k, v in zip(k_grid, ivs)],
        })

    return {
        "symbol": symbol.upper(),
        "spot": result.spot,
        "scan_time": result.scan_time,
        "is_arbitrage_free": result.is_arbitrage_free,
        "calendar_violations": len(result.calendar_violations),
        "butterfly_violations": len(result.butterfly_violations),
        "slices": slices,
    }
