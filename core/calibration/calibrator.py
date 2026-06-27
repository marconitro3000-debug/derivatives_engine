"""
calibration/calibrator.py
Common calibration engine for SVI and Heston models.

Both models expose the same interface:
    calibrate(market_data, warm_start=None) -> CalibrationResult

The engine minimises the weighted RMSE between model implied vols and market
implied vols across the option chain. It supports:

  - warm start: begin optimisation from previous parameters (faster, smoother)
  - EWMA blending: smooth new parameters against the previous calibration
  - quality metrics: RMSE, max error, per-point residuals
"""

import time
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import least_squares

from core.models import svi as svi_mod
from core.models import heston as heston_mod


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    model: str                       # 'svi' or 'heston'
    params: dict                     # fitted parameters
    rmse: float                      # root-mean-square IV error
    max_error: float                 # worst single-point IV error
    n_points: int                    # number of market points used
    arb_free: bool                   # passed no-arbitrage check
    elapsed_sec: float               # wall-clock calibration time
    timestamp: float = field(default_factory=time.time)
    warm_started: bool = False

    def summary(self) -> str:
        return (f"[{self.model.upper()}] RMSE={self.rmse:.4%}  "
                f"max={self.max_error:.4%}  n={self.n_points}  "
                f"arb_free={self.arb_free}  "
                f"{'warm' if self.warm_started else 'cold'}  "
                f"{self.elapsed_sec:.2f}s")


# ── market data container ─────────────────────────────────────────────────────

@dataclass
class MarketData:
    """
    Option-chain snapshot used for calibration.

    spot       : underlying spot price
    r          : risk-free rate
    strikes    : array of strikes K
    maturities : array of maturities T (years), aligned with strikes
    ivs        : array of market implied vols, aligned with strikes/maturities
    weights    : optional per-point weights (e.g. by vega or liquidity)
    """
    spot: float
    r: float
    strikes: np.ndarray
    maturities: np.ndarray
    ivs: np.ndarray
    weights: np.ndarray = None

    def __post_init__(self):
        self.strikes    = np.asarray(self.strikes,    dtype=float)
        self.maturities = np.asarray(self.maturities, dtype=float)
        self.ivs        = np.asarray(self.ivs,        dtype=float)
        if self.weights is None:
            self.weights = np.ones_like(self.ivs)
        else:
            self.weights = np.asarray(self.weights, dtype=float)


# ── SVI calibrator ────────────────────────────────────────────────────────────

def _calibrate_svi(md: MarketData, warm_start: dict | None) -> CalibrationResult:
    """
    Calibrate SVI per maturity slice (SVI is defined per-expiry).
    Returns params as a dict keyed by maturity: {"slices": {T: {a,b,rho,m,sigma}}}.
    """
    t0 = time.time()
    unique_T = np.unique(md.maturities)

    warm_slices = (warm_start or {}).get("slices", {}) if warm_start else {}
    slices = {}
    all_model_iv, all_mkt_iv = [], []

    for T in unique_T:
        mask = md.maturities == T
        K_s  = md.strikes[mask]
        iv_s = md.ivs[mask]
        w_s  = md.weights[mask]
        F    = md.spot * np.exp(md.r * T)
        k    = np.log(K_s / F)

        T_key = f"{T:.6f}"
        if T_key in warm_slices:
            seed = svi_mod.SVIParams.from_dict(warm_slices[T_key]).as_array()
            warm = True
        else:
            atm_var = float(np.median(iv_s) ** 2 * T)
            seed = svi_mod.default_params(atm_var).as_array()
            warm = bool(warm_slices)

        def resid(x):
            p = svi_mod.SVIParams.from_array(x)
            w_model = svi_mod.total_variance(k, p)
            iv_model = np.sqrt(np.maximum(w_model, 1e-12) / T)
            return w_s * (iv_model - iv_s)

        lb = [0.0, 0.0, -0.999, -2.0, 1e-4]
        ub = [2.0, 4.0,  0.999,  2.0, 2.0]
        sol = least_squares(resid, seed, bounds=(lb, ub),
                            method="trf", max_nfev=5000)

        p_fit = svi_mod.SVIParams.from_array(sol.x)
        slices[T_key] = p_fit.to_dict()

        iv_model = svi_mod.implied_vol_svi(k, T, p_fit)
        all_model_iv.extend(iv_model)
        all_mkt_iv.extend(iv_s)

    all_model_iv = np.array(all_model_iv)
    all_mkt_iv   = np.array(all_mkt_iv)
    errors = np.abs(all_model_iv - all_mkt_iv)

    arb_free = all(
        svi_mod.is_butterfly_arbitrage_free(svi_mod.SVIParams.from_dict(s))
        for s in slices.values()
    )

    return CalibrationResult(
        model="svi",
        params={"slices": slices},
        rmse=float(np.sqrt(np.mean((all_model_iv - all_mkt_iv) ** 2))),
        max_error=float(errors.max()),
        n_points=len(all_mkt_iv),
        arb_free=arb_free,
        elapsed_sec=time.time() - t0,
        warm_started=bool(warm_slices),
    )


# ── Heston calibrator ─────────────────────────────────────────────────────────

def _calibrate_heston(md: MarketData, warm_start: dict | None) -> CalibrationResult:
    t0 = time.time()

    if warm_start is not None:
        seed = heston_mod.HestonParams.from_dict(warm_start).as_array()
        warm = True
    else:
        atm_var = float(np.median(md.ivs) ** 2)
        seed = heston_mod.default_params(atm_var).as_array()
        warm = False

    from options.black_scholes import price as bs_price
    mkt_prices = np.array([
        bs_price(md.spot, K, T, md.r, iv, "call")
        for K, T, iv in zip(md.strikes, md.maturities, md.ivs)
    ])

    def resid(x):
        p = heston_mod.HestonParams.from_array(x)
        model_prices = np.array([
            heston_mod.price(md.spot, K, T, md.r, p, "call")
            for K, T in zip(md.strikes, md.maturities)
        ])
        return md.weights * (model_prices - mkt_prices)

    lb = [1e-4, 0.1,  1e-4, 1e-2, -0.999]
    ub = [1.0,  20.0, 1.0,  5.0,   0.0]

    sol = least_squares(resid, seed, bounds=(lb, ub),
                        method="trf", max_nfev=120, ftol=1e-5, xtol=1e-5)

    p_fit = heston_mod.HestonParams.from_array(sol.x)

    from options.implied_vol import implied_vol
    iv_model = []
    for K, T in zip(md.strikes, md.maturities):
        try:
            px = heston_mod.price(md.spot, K, T, md.r, p_fit, "call")
            iv_model.append(implied_vol(md.spot, K, T, md.r, px, "call"))
        except (ValueError, ZeroDivisionError):
            iv_model.append(np.nan)
    iv_model = np.array(iv_model)
    valid = ~np.isnan(iv_model)
    errors = np.abs(iv_model[valid] - md.ivs[valid])

    return CalibrationResult(
        model="heston",
        params=p_fit.to_dict(),
        rmse=float(np.sqrt(np.mean((iv_model[valid] - md.ivs[valid]) ** 2))),
        max_error=float(errors.max()) if len(errors) else float("nan"),
        n_points=int(valid.sum()),
        arb_free=p_fit.feller_satisfied(),
        elapsed_sec=time.time() - t0,
        warm_started=warm,
    )


# ── EWMA blending ─────────────────────────────────────────────────────────────

def blend_params(old: dict, new: dict, alpha: float) -> dict:
    """
    EWMA blend of two parameter dicts: result = alpha*new + (1-alpha)*old.
    alpha=1.0 means fully trust the new calibration; alpha=0.0 keeps the old.
    """
    return {key: alpha * new[key] + (1 - alpha) * old[key] for key in new}


# ── public API ────────────────────────────────────────────────────────────────

def calibrate(model: str, market_data: MarketData,
              warm_start: dict | None = None,
              ewma_alpha: float | None = None) -> CalibrationResult:
    """
    Calibrate a model to a market-data snapshot.

    Parameters
    ----------
    model       : 'svi' or 'heston'
    market_data : MarketData snapshot
    warm_start  : previous parameter dict to seed optimisation (optional)
    ewma_alpha  : if set and warm_start given, EWMA-blend the fitted params
                  with warm_start (0 < alpha <= 1). Smooths day-to-day noise.

    Returns
    -------
    CalibrationResult
    """
    if model == "svi":
        result = _calibrate_svi(market_data, warm_start)
    elif model == "heston":
        result = _calibrate_heston(market_data, warm_start)
    else:
        raise ValueError("model must be 'svi' or 'heston'.")

    if ewma_alpha is not None and warm_start is not None:
        result.params = blend_params(warm_start, result.params, ewma_alpha)

    return result
