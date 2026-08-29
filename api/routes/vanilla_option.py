from __future__ import annotations

import time
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from api.schemas import VanillaOptionRequest
from arbitrage.vol_surface import _load_chain_yfinance
from core.calibration.calibrator import CalibrationResult
from core.calibration.ssvi_surface import calibrate_ssvi_surface_from_chain
from core.calibration.store import CalibrationStore
from core.models import heston as heston_mod
from core.models.merton import MertonParams
from core.models.merton import price as merton_price
from ml.ssvi import SSVIParams
from options.binomial_tree import binomial_price
from options.black_scholes import greeks as bs_greeks, price as bs_price
from options.engine import PricingEngine
from options.greeks_fd import fd_greeks
from options.monte_carlo import mc_price
from options.pde import price as pde_price
from volatility.local_vol import LocalVolSurface, mc_price_local_vol

router = APIRouter(prefix="/api", tags=["vanilla-options"])

_DB_PATH = "data/derivatives_engine.sqlite"
_CALIBRATED_MODELS: tuple[str, ...] = ("heston", "svi", "local_vol")


def _engine(model: str, source: str = "yfinance") -> PricingEngine:
    return PricingEngine(model=model, db_path=_DB_PATH, source=source)


def _calibrated_greeks(eng: PricingEngine, ticker: str, K: float, T: float, option: str, spot: float, q: float) -> dict:
    """Delta/gamma/theta via spot/time bumps against the live calibrated surface.

    Vega/rho aren't included here: they aren't a single well-defined number
    against an already-calibrated stochastic-vol/smile model the way they are
    for a flat-vol pricer. For Heston specifically, `_heston_vega_rho` below
    fills in a documented v0-sensitivity proxy instead of omitting it outright.
    """
    h_s = spot * 0.01
    base = eng.price_option(ticker, K, T, option, spot=spot, q=q)
    up = eng.price_option(ticker, K, T, option, spot=spot + h_s, q=q)
    dn = eng.price_option(ticker, K, T, option, spot=spot - h_s, q=q)
    delta = (up - dn) / (2 * h_s)
    gamma = (up - 2 * base + dn) / (h_s**2)
    dt = 1.0 / 365.0
    theta_1d = eng.price_option(ticker, K, max(T - dt, 1e-6), option, spot=spot, q=q) - base
    return {"delta": float(delta), "gamma": float(gamma), "theta_1d": float(theta_1d)}


def _heston_vega_rho(eng: PricingEngine, ticker: str, K: float, T: float, option: str, spot: float, q: float) -> dict | None:
    """v0-sensitivity proxy for "vega" and a genuine rate-bump rho, computed by
    bumping the *calibrated* Heston parameters directly (not the engine's flat
    `sigma`, which doesn't exist for a stochastic-vol model).

    This is NOT the same quantity as Black-Scholes vega (sensitivity to a flat
    implied vol) — it's the price sensitivity to the calibrated instantaneous
    variance v0. Reported under a `vega_v0` key, not `vega`, to avoid implying
    they're comparable across models.
    """
    params = eng.store.latest_params(ticker, "heston")
    if params is None:
        return None
    p = heston_mod.HestonParams.from_dict(params)

    bump_v0 = 0.01
    p_up = heston_mod.HestonParams(v0=p.v0 + bump_v0, kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho)
    p_dn = heston_mod.HestonParams(v0=max(p.v0 - bump_v0, 1e-6), kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho)
    px_up = heston_mod.price(spot, K, T, eng.r, p_up, option, q)
    px_dn = heston_mod.price(spot, K, T, eng.r, p_dn, option, q)
    vega_v0 = (px_up - px_dn) / (2 * bump_v0)

    bump_r = 0.0001
    px_r_up = heston_mod.price(spot, K, T, eng.r + bump_r, p, option, q)
    px_r_dn = heston_mod.price(spot, K, T, eng.r - bump_r, p, option, q)
    rho = (px_r_up - px_r_dn) / (2 * bump_r) / 100.0  # per 1% r, matching the Black-Scholes convention

    return {"vega_v0": float(vega_v0), "rho": float(rho)}


def _local_vol_surface(ticker: str) -> LocalVolSurface:
    store = CalibrationStore(_DB_PATH)
    params = store.latest_params(ticker, "local_vol")
    if params is None:
        raise RuntimeError(f"No calibration for {ticker}/local_vol.")
    ssvi = SSVIParams(rho=params["rho"], eta=params["eta"], gamma=params["gamma"])
    return LocalVolSurface(ssvi, np.array(params["T_nodes"]), np.array(params["theta_nodes"]))


def _local_vol_sigma_fn(lv: LocalVolSurface, spot: float, r: float, q: float):
    """Vectorized sigma(x_array, t) closure for options/pde.py, anchored to a
    given spot's forward — matches the same moneyness convention
    volatility/local_vol.py's own mc_price_local_vol() uses (forward from the
    spot passed to that call). Must stay vectorized: the PDE grid evaluates
    this at every (x, t) pair, so a per-point Python call here made local-vol
    American pricing unusably slow (see options/pde.py's _sigma_fn docstring)."""
    def sigma_local(x: np.ndarray, t: float) -> np.ndarray:
        t_safe = max(t, 1e-6)
        F = spot * np.exp((r - q) * t_safe)
        k = np.asarray(x, dtype=float) - np.log(F)
        return lv(k, t_safe)
    return sigma_local


def _pde_american_local_vol_price(lv: LocalVolSurface, spot: float, K: float, T: float, r: float, q: float, option: str) -> float:
    sigma_fn = _local_vol_sigma_fn(lv, spot, r, q)
    return pde_price(spot, K, T, r, sigma_fn, option, "american", q)["price"]


def _price_one(req: VanillaOptionRequest) -> dict:
    S, K, T, r, sigma = req.spot, req.strike, req.maturity_years, req.risk_free_rate, req.volatility
    q = req.dividend_yield
    opt = req.option_type

    if req.model == "black_scholes":
        price = bs_price(S, K, T, r, sigma, opt, q)
        g = bs_greeks(S, K, T, r, sigma, q)
        greeks = {
            "delta": g[f"delta_{opt}"], "gamma": g["gamma"], "vega": g["vega"],
            "theta_1d": g[f"theta_{opt}"], "rho": g[f"rho_{opt}"],
        }
        return {"model": "black_scholes", "price": price, "greeks": greeks}

    if req.model == "binomial":
        # dividend_yield isn't applied here — the CRR tree's up/down factors would
        # need a (r-q) drift adjustment that options/binomial_tree.py doesn't take yet.
        res = binomial_price(S, K, T, r, sigma, opt, req.style, req.n_steps)
        pricer = lambda s, k, t, rr, vv, o: binomial_price(s, k, t, rr, vv, o, req.style, req.n_steps)["price"]
        greeks = fd_greeks(pricer, S, K, T, r, sigma, opt)
        return {
            "model": "binomial", "price": res["price"], "early_exercise_premium": res["early_exercise"],
            "n_steps": res["n_steps"], "style": res["style"], "greeks": greeks,
        }

    if req.model == "monte_carlo":
        # dividend_yield isn't applied here either — same reason (options/monte_carlo.py's
        # GBM simulator would need a (r-q) drift parameter it doesn't take yet).
        res = mc_price(S, K, T, r, sigma, f"european_{opt}", n_sims=req.n_sims, seed=req.seed)
        pricer = lambda s, k, t, rr, vv, o: mc_price(s, k, t, rr, vv, f"european_{o}", n_sims=req.n_sims, seed=req.seed)["price"]
        greeks = fd_greeks(pricer, S, K, T, r, sigma, opt)
        return {
            "model": "monte_carlo", "price": res["price"], "std_error": res["std_error"],
            "conf_95_lo": res["conf_95_lo"], "conf_95_hi": res["conf_95_hi"], "n_sims": res["n_sims"], "greeks": greeks,
        }

    if req.model == "merton":
        p = MertonParams(jump_intensity=req.jump_intensity, jump_mean=req.jump_mean, jump_vol=req.jump_vol)
        px = merton_price(S, K, T, r, sigma, p, opt, q)
        pricer = lambda s, k, t, rr, vv, o: merton_price(s, k, t, rr, vv, p, o, q)
        greeks = fd_greeks(pricer, S, K, T, r, sigma, opt)
        return {
            "model": "merton", "price": px, "greeks": greeks,
            "jump_intensity": p.jump_intensity, "jump_mean": p.jump_mean, "jump_vol": p.jump_vol,
        }

    if req.model == "local_vol":
        try:
            lv = _local_vol_surface(req.underlying)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} POST /api/calibrate/{req.underlying}?model=local_vol first.") from exc

        if req.style == "american":
            # American exercise needs a free-boundary solve — Monte Carlo would need
            # Longstaff-Schwartz regression; the Crank-Nicolson PDE solver (options/pde.py)
            # handles it directly by projecting onto intrinsic value at each time step.
            px = _pde_american_local_vol_price(lv, S, K, T, r, q, opt)
            h_s = S * 0.01
            up = _pde_american_local_vol_price(lv, S + h_s, K, T, r, q, opt)
            dn = _pde_american_local_vol_price(lv, max(S - h_s, 1e-6), K, T, r, q, opt)
            delta = (up - dn) / (2 * h_s)
            gamma = (up - 2 * px + dn) / (h_s**2)
            dt = 1.0 / 365.0
            theta_val = _pde_american_local_vol_price(lv, S, K, max(T - dt, 1e-6), r, q, opt)
            greeks = {"delta": float(delta), "gamma": float(gamma), "theta_1d": float(theta_val - px)}
            return {"model": "local_vol", "price": px, "style": "american", "greeks": greeks}

        px, se = mc_price_local_vol(S, K, T, r, lv, opt, n_sims=req.n_sims, seed=req.seed)
        h_s = S * 0.01
        up, _ = mc_price_local_vol(S + h_s, K, T, r, lv, opt, n_sims=req.n_sims, seed=req.seed)
        dn, _ = mc_price_local_vol(S - h_s, K, T, r, lv, opt, n_sims=req.n_sims, seed=req.seed)
        delta = (up - dn) / (2 * h_s)
        gamma = (up - 2 * px + dn) / (h_s**2)
        dt = 1.0 / 365.0
        theta_dn, _ = mc_price_local_vol(S, K, max(T - dt, 1e-6), r, lv, opt, n_sims=req.n_sims, seed=req.seed)
        greeks = {"delta": float(delta), "gamma": float(gamma), "theta_1d": float(theta_dn - px)}
        return {"model": "local_vol", "price": px, "std_error": se, "greeks": greeks}

    if req.model in ("heston", "svi"):
        eng = _engine(req.model)
        try:
            price = eng.price_option(req.underlying, K, T, opt, spot=S, q=q)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc} POST /api/calibrate/{req.underlying}?model={req.model} first."
            ) from exc
        greeks = _calibrated_greeks(eng, req.underlying, K, T, opt, S, q)
        if req.model == "heston":
            extra = _heston_vega_rho(eng, req.underlying, K, T, opt, S, q)
            if extra:
                greeks.update(extra)
        return {"model": req.model, "price": price, "greeks": greeks}

    raise ValueError(f"Unknown model: {req.model}")


@router.post("/price/option")
def price_option(req: VanillaOptionRequest):
    try:
        return _price_one(req)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/price/option/compare")
def compare_option_models(req: VanillaOptionRequest):
    results = []
    for model in ("black_scholes", "binomial", "monte_carlo", "merton"):
        variant = req.model_copy(update={"model": model})
        try:
            results.append(_price_one(variant))
        except Exception as exc:
            results.append({"model": model, "error": str(exc)})

    store = CalibrationStore(_DB_PATH)
    for model in _CALIBRATED_MODELS:
        if store.latest_params(req.underlying, model) is None:
            results.append({
                "model": model, "calibrated": False,
                "note": f"not calibrated for {req.underlying} — POST /api/calibrate/{req.underlying}?model={model} first",
            })
            continue
        variant = req.model_copy(update={"model": model})
        try:
            results.append(_price_one(variant))
        except Exception as exc:
            results.append({"model": model, "error": str(exc)})

    return {"underlying": req.underlying, "option_type": req.option_type, "strike": req.strike, "results": results}


@router.post("/calibrate/{ticker}")
def calibrate(
    ticker: str,
    model: Literal["heston", "svi", "local_vol"] = Query("heston"),
    source: Literal["yfinance", "synthetic"] = Query("yfinance"),
):
    if model == "local_vol":
        if source != "yfinance":
            raise HTTPException(status_code=400, detail="local_vol calibration requires source=yfinance (a real option chain).")
        try:
            spot, strikes, maturities, ivs, weights = _load_chain_yfinance(ticker, 0.04, (0.85, 1.15), 5, 6)
            fit = calibrate_ssvi_surface_from_chain(spot, strikes, maturities, ivs, r=0.04, weights=weights)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        store = CalibrationStore(_DB_PATH)
        result = CalibrationResult(
            model="local_vol",
            params={"rho": fit.rho, "eta": fit.eta, "gamma": fit.gamma, "T_nodes": fit.T_nodes, "theta_nodes": fit.theta_nodes},
            rmse=fit.rmse, max_error=fit.rmse, n_points=fit.n_points,
            arb_free=fit.arb_free, elapsed_sec=fit.elapsed_sec,
        )
        store.save(ticker, result, spot=fit.spot, r=0.04)
        return {
            "ticker": ticker, "model": "local_vol", "rmse": result.rmse, "max_error": result.max_error,
            "n_points": result.n_points, "arb_free": result.arb_free, "elapsed_sec": result.elapsed_sec,
            "timestamp": result.timestamp, "warm_started": False,
        }

    eng = _engine(model, source=source)
    try:
        result = eng.update(ticker)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "ticker": ticker, "model": result.model, "rmse": result.rmse, "max_error": result.max_error,
        "n_points": result.n_points, "arb_free": result.arb_free, "elapsed_sec": result.elapsed_sec,
        "timestamp": result.timestamp, "warm_started": result.warm_started,
    }


@router.get("/calibration/{ticker}")
def calibration_status(ticker: str, model: Literal["heston", "svi", "local_vol"] = Query("heston")):
    store = CalibrationStore(_DB_PATH)
    params = store.latest_params(ticker, model)
    if params is None:
        return {"calibrated": False, "ticker": ticker, "model": model}
    latest = (store.history(ticker, model, limit=1) or [{}])[0]
    return {
        "calibrated": True, "ticker": ticker, "model": model,
        "rmse": latest.get("rmse"), "timestamp": latest.get("timestamp"), "arb_free": latest.get("arb_free"),
    }
