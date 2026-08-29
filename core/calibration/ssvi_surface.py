"""
core/calibration/ssvi_surface.py
Joint, calendar-consistent SSVI surface calibration (Gatheral & Jacquier 2014)
from a live option chain.

This feeds volatility/local_vol.py's Dupire local-vol surface, which needs a
single consistent total-variance surface w(k,T) across ALL maturities at
once. core/calibration/calibrator.py's `_calibrate_svi` fits each expiry
independently — fine for per-expiry option pricing, but not safe to
differentiate across maturities (which Dupire's dw/dT requires) since nothing
enforces consistency between adjacent slices. ml/ssvi.py::calibrate_ssvi
already does the joint fit; this module is the glue that turns a live option
chain into its inputs and packages the result for storage/reuse.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ml.ssvi import SSVIParams, calibrate_ssvi


@dataclass
class SSVISurfaceResult:
    rho: float
    eta: float
    gamma: float
    T_nodes: list[float]
    theta_nodes: list[float]
    spot: float
    rmse: float
    n_points: int
    arb_free: bool
    elapsed_sec: float


def calibrate_ssvi_surface_from_chain(
    spot: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    ivs: np.ndarray,
    r: float = 0.04,
    weights: np.ndarray | None = None,
) -> SSVISurfaceResult:
    """Fit a joint SSVI surface to a flattened multi-expiry option chain
    (same aligned-array shape used by core/calibration/calibrator.py's
    MarketData: one row per (strike, maturity, iv) observation)."""
    t0 = time.time()
    unique_T = sorted(np.unique(maturities).tolist())
    if len(unique_T) < 2:
        raise ValueError("SSVI surface calibration needs at least 2 maturities.")

    k_list, w_list, wt_list, theta_list = [], [], [], []
    for T in unique_T:
        mask = maturities == T
        F = spot * np.exp(r * T)
        k = np.log(strikes[mask] / F)
        w = ivs[mask] ** 2 * T
        wt = weights[mask] if weights is not None else np.ones_like(w)
        k_list.append(k)
        w_list.append(w)
        wt_list.append(wt)
        atm_var = float(np.interp(0.0, k, w)) if len(k) > 1 else float(np.mean(w))
        theta_list.append(max(atm_var, 1e-6))

    ssvi = calibrate_ssvi(k_list, w_list, theta_list, wt_list)

    errs = np.concatenate([ssvi.total_var(k, theta) - w for k, w, theta in zip(k_list, w_list, theta_list)])
    rmse = float(np.sqrt(np.mean(errs**2)))
    arb_free = all(ssvi.no_butterfly_arbitrage(theta) for theta in theta_list) and ssvi.no_calendar_spread_arbitrage(
        np.array(theta_list)
    )

    return SSVISurfaceResult(
        rho=ssvi.rho,
        eta=ssvi.eta,
        gamma=ssvi.gamma,
        T_nodes=unique_T,
        theta_nodes=theta_list,
        spot=spot,
        rmse=rmse,
        n_points=int(sum(len(k) for k in k_list)),
        arb_free=bool(arb_free),
        elapsed_sec=time.time() - t0,
    )
