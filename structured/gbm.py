from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    spot: float,
    rate: float,
    volatility: float,
    maturity_years: float,
    n_paths: int,
    n_steps: int,
    dividend_yield: float = 0.0,
    seed: int | None = 42,
) -> np.ndarray:
    """Simulate GBM price paths with shape (n_paths, n_steps + 1)."""
    if spot <= 0:
        raise ValueError("spot must be positive")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")
    if maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")

    rng = np.random.default_rng(seed)
    dt = maturity_years / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (rate - dividend_yield - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = spot * np.exp(log_paths)
    return np.column_stack([np.full(n_paths, spot), paths])
