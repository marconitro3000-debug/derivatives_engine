"""Portfolio risk metrics from simulated value paths: VaR/CVaR (historical
simulation, same convention as the retired `portfolio/var.py` prototype)
and max drawdown per path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskReport:
    var_95_pct: float  # potential loss, positive number, % of initial value
    cvar_95_pct: float
    mean_pnl_pct: float
    drawdown_p50_pct: float
    drawdown_p95_pct: float
    terminal_pnl_pct: list[float]  # sampled, for charting


def max_drawdown_per_path(value_paths: np.ndarray) -> np.ndarray:
    """value_paths: (n_paths, n_steps+1). Returns (n_paths,) of max
    peak-to-trough drawdown (positive fraction) observed along each path."""
    running_peak = np.maximum.accumulate(value_paths, axis=1)
    drawdown = (running_peak - value_paths) / running_peak
    return drawdown.max(axis=1)


def risk_report(value_paths: np.ndarray, initial_value: float, confidence: float = 0.95) -> RiskReport:
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")
    terminal = value_paths[:, -1]
    pnl_pct = (terminal - initial_value) / initial_value * 100.0

    loss_pct = -pnl_pct  # positive = loss
    var = float(np.quantile(loss_pct, confidence))
    tail = loss_pct[loss_pct >= var]
    cvar = float(tail.mean()) if tail.size > 0 else var

    drawdowns_pct = max_drawdown_per_path(value_paths) * 100.0

    sample = pnl_pct[: min(500, len(pnl_pct))]
    return RiskReport(
        var_95_pct=var,
        cvar_95_pct=cvar,
        mean_pnl_pct=float(pnl_pct.mean()),
        drawdown_p50_pct=float(np.quantile(drawdowns_pct, 0.50)),
        drawdown_p95_pct=float(np.quantile(drawdowns_pct, 0.95)),
        terminal_pnl_pct=[float(x) for x in sample],
    )
