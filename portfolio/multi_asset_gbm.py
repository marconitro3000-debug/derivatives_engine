"""Correlated multi-asset GBM Monte Carlo.

Extends the single-asset simulator in `structured/gbm.py` to N correlated
names via a Cholesky decomposition of the (realized) correlation matrix —
same discretization, same RNG convention (`numpy.random.default_rng`), just
vectorized across assets so all N names share one consistent set of
correlated shocks per path/step.
"""

from __future__ import annotations

import numpy as np


def _nearest_psd_correlation(corr: np.ndarray) -> np.ndarray:
    """Clip small/negative eigenvalues so a correlation matrix estimated
    from limited historical overlap is safe to Cholesky-decompose."""
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-10, None)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(psd))
    psd = psd / np.outer(d, d)
    np.fill_diagonal(psd, 1.0)
    return psd


def simulate_correlated_gbm_paths(
    spots: list[float],
    rate: float,
    volatilities: list[float],
    correlation: list[list[float]],
    maturity_years: float,
    n_paths: int,
    n_steps: int,
    dividend_yields: list[float] | None = None,
    seed: int | None = 42,
) -> np.ndarray:
    """Simulate correlated GBM paths for N assets.

    Returns an array of shape (n_paths, n_steps + 1, n_assets).
    """
    n_assets = len(spots)
    if n_assets == 0:
        raise ValueError("need at least one asset")
    spots_arr = np.asarray(spots, dtype=float)
    vols = np.asarray(volatilities, dtype=float)
    corr = _nearest_psd_correlation(np.asarray(correlation, dtype=float))
    div = np.asarray(dividend_yields if dividend_yields is not None else [0.0] * n_assets, dtype=float)

    if spots_arr.shape != (n_assets,) or vols.shape != (n_assets,) or corr.shape != (n_assets, n_assets):
        raise ValueError("spots, volatilities and correlation must have matching dimensions")
    if (spots_arr <= 0).any():
        raise ValueError("all spots must be positive")
    if maturity_years <= 0 or n_paths <= 0 or n_steps <= 0:
        raise ValueError("maturity_years, n_paths and n_steps must be positive")

    chol = np.linalg.cholesky(corr)
    rng = np.random.default_rng(seed)
    dt = maturity_years / n_steps
    sqrt_dt = np.sqrt(dt)
    drift = (rate - div - 0.5 * vols**2) * dt  # shape (n_assets,)

    paths = np.empty((n_paths, n_steps + 1, n_assets))
    paths[:, 0, :] = spots_arr
    for step in range(1, n_steps + 1):
        z = rng.standard_normal((n_paths, n_assets))
        correlated_z = z @ chol.T
        diffusion = vols * sqrt_dt * correlated_z
        paths[:, step, :] = paths[:, step - 1, :] * np.exp(drift + diffusion)

    return paths
