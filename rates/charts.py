"""Chart helpers for interest-rate curves."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from .curves import DiscountCurve


STYLE = {
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.35,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def _save(fig, out_dir: str, filename: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return filename


def plot_curve(curve: DiscountCurve, out_dir: str, max_maturity: float | None = None) -> list[str]:
    """Save zero-rate, discount-factor, and forward-rate charts."""
    plt.rcParams.update(STYLE)
    max_t = max_maturity or float(curve.times[-1])
    grid = np.linspace(max(1.0 / 365.0, max_t / 300), max_t, 220)

    zero = curve.zero_rate(grid) * 100
    dfs = curve.discount_factor(grid)
    fwd = np.array([curve.forward_rate(max(t - 0.25, 0.0), t) for t in grid]) * 100

    saved = []

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(grid, zero, color="#0072B2", lw=2.2, label="Zero rate")
    ax.scatter(curve.times, curve.zero_rate(curve.times) * 100, color="#D55E00", s=35, zorder=4, label="Pillars")
    ax.set_title(f"Zero Curve [{curve.name}]")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Zero rate")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.2f}%"))
    ax.legend()
    saved.append(_save(fig, out_dir, "zero_curve.png"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(grid, dfs, color="#009E73", lw=2.2)
    ax.scatter(curve.times, curve.discount_factors, color="#D55E00", s=35, zorder=4)
    ax.set_title(f"Discount Factors [{curve.name}]")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Discount factor")
    saved.append(_save(fig, out_dir, "discount_factors.png"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(grid, fwd, color="#CC79A7", lw=2.2)
    ax.set_title(f"3M Rolling Forward Rates [{curve.name}]")
    ax.set_xlabel("End maturity (years)")
    ax.set_ylabel("Forward rate")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.2f}%"))
    saved.append(_save(fig, out_dir, "forward_rates.png"))

    return saved
