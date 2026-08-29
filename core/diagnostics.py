"""
core/diagnostics.py
Static-arbitrage diagnostics for any `VolSurface`, and vol/price fit errors.

Two conditions decide whether a total-variance surface ``w(k, T)`` is free of
static arbitrage (Roper 2010; Gatheral & Jacquier 2014).

**Calendar spread.** Total variance must be non-decreasing in maturity along a
fixed log-moneyness:

    dw/dT >= 0

A violation means a longer-dated option is worth less than a shorter-dated one
struck at the same relative moneyness -- a calendar spread with negative cost.

**Butterfly.** The risk-neutral density implied by the surface must be
non-negative. In total-variance coordinates that is Durrleman's condition:

    g(k, T) = (1 - k w_k / (2w))^2  -  (w_k^2 / 4)(1/4 + 1/w)  +  w_kk / 2  >= 0

A violation means a butterfly spread has negative cost, i.e. the surface prices
a negative probability.

Both are checked on a dense grid that deliberately extends **beyond** the range
of the quotes. A surface that is arbitrage-free only where it saw data is not
usable for interpolation, and this is exactly where an unconstrained neural
network fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.surface import VolSurface


# -- results ------------------------------------------------------------------

@dataclass
class ArbitrageReport:
    """Outcome of a dense-grid static-arbitrage scan."""

    n_grid: int
    calendar_violations: int
    butterfly_violations: int
    worst_calendar: float      # most negative dw/dT found (0.0 if clean)
    worst_butterfly: float     # most negative g found (0.0 if clean)
    worst_calendar_at: tuple[float, float] | None   # (k, T)
    worst_butterfly_at: tuple[float, float] | None

    @property
    def is_arbitrage_free(self) -> bool:
        return self.calendar_violations == 0 and self.butterfly_violations == 0

    @property
    def calendar_pct(self) -> float:
        return 100.0 * self.calendar_violations / max(self.n_grid, 1)

    @property
    def butterfly_pct(self) -> float:
        return 100.0 * self.butterfly_violations / max(self.n_grid, 1)

    def __str__(self) -> str:
        if self.is_arbitrage_free:
            return f"arbitrage-free on {self.n_grid} grid points"
        parts = []
        if self.calendar_violations:
            k, T = self.worst_calendar_at
            parts.append(
                f"calendar {self.calendar_violations}/{self.n_grid} "
                f"({self.calendar_pct:.2f}%), worst dw/dT={self.worst_calendar:.2e} "
                f"at k={k:+.3f} T={T:.3f}"
            )
        if self.butterfly_violations:
            k, T = self.worst_butterfly_at
            parts.append(
                f"butterfly {self.butterfly_violations}/{self.n_grid} "
                f"({self.butterfly_pct:.2f}%), worst g={self.worst_butterfly:.2e} "
                f"at k={k:+.3f} T={T:.3f}"
            )
        return "; ".join(parts)


@dataclass
class FitReport:
    """Goodness of fit of a surface to the chain it was fitted to."""

    n: int
    rmse_vol_bps: float        # RMSE of implied vol, in basis points
    mae_vol_bps: float
    max_vol_bps: float
    rmse_price: float          # RMSE of re-priced mid, in currency units
    pct_inside_spread: float   # share of quotes re-priced inside the bid-ask

    def __str__(self) -> str:
        return (
            f"n={self.n}  IV RMSE={self.rmse_vol_bps:.1f}bp  "
            f"MAE={self.mae_vol_bps:.1f}bp  max={self.max_vol_bps:.1f}bp  "
            f"price RMSE={self.rmse_price:.4f}  "
            f"inside spread={self.pct_inside_spread:.1f}%"
        )


# -- arbitrage scan -----------------------------------------------------------

def butterfly_g(surface: VolSurface, k: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Durrleman's function ``g(k, T)``; non-negative iff no butterfly arbitrage."""
    w = np.maximum(surface.total_variance(k, T), 1e-12)
    w_k = surface.dw_dk(k, T)
    w_kk = surface.d2w_dk2(k, T)
    return (
        (1.0 - k * w_k / (2.0 * w)) ** 2
        - (w_k ** 2 / 4.0) * (0.25 + 1.0 / w)
        + w_kk / 2.0
    )


def scan_arbitrage(
    surface: VolSurface,
    k_range: tuple[float, float] = (-1.0, 0.6),
    T_range: tuple[float, float] = (0.02, 2.0),
    n_k: int = 121,
    n_T: int = 61,
    tol: float = 1e-8,
) -> ArbitrageReport:
    """Scan a surface for calendar and butterfly arbitrage on a dense grid.

    `tol` absorbs finite-difference noise: violations smaller than it are not
    economically meaningful and would otherwise flag a clean surface.
    """
    kk, TT = np.meshgrid(
        np.linspace(*k_range, n_k),
        np.linspace(*T_range, n_T),
        indexing="ij",
    )
    k, T = kk.ravel(), TT.ravel()

    dwdT = surface.dw_dT(k, T)
    g = butterfly_g(surface, k, T)

    cal_mask = dwdT < -tol
    bf_mask = g < -tol

    def _worst(values, mask):
        if not mask.any():
            return 0.0, None
        idx = int(np.argmin(values))
        return float(values[idx]), (float(k[idx]), float(T[idx]))

    worst_cal, at_cal = _worst(dwdT, cal_mask)
    worst_bf, at_bf = _worst(g, bf_mask)

    return ArbitrageReport(
        n_grid=k.size,
        calendar_violations=int(cal_mask.sum()),
        butterfly_violations=int(bf_mask.sum()),
        worst_calendar=worst_cal,
        worst_butterfly=worst_bf,
        worst_calendar_at=at_cal,
        worst_butterfly_at=at_bf,
    )


# -- fit quality --------------------------------------------------------------

def fit_report(surface: VolSurface, snapshot, weighted: bool = True) -> FitReport:
    """Score a fitted surface against the chain, in vol *and* in price space.

    Vol-space RMSE is the number every paper reports; price-space RMSE and the
    share of quotes re-priced inside the bid-ask are the ones a desk asks about,
    because a 20bp vol error on a one-week wing option is free while the same
    error at the money is not.
    """
    iv_model = surface.implied_vol(snapshot.k, snapshot.T)
    err = (iv_model - snapshot.iv) * 10_000.0                # basis points

    w = snapshot.weight if weighted else np.ones_like(err)
    w = w / w.sum()

    forwards = np.array([snapshot.forwards[t] for t in snapshot.T])
    discounts = np.array([snapshot.discounts[t] for t in snapshot.T])
    call_model = surface.call_price(snapshot.k, snapshot.T, forwards, discounts)

    # Convert modelled calls to puts by parity where the quote is a put.
    strike = forwards * np.exp(snapshot.k)
    put_model = call_model - discounts * (forwards - strike)
    px_model = np.where(snapshot.is_call, call_model, put_model)

    px_err = px_model - snapshot.mid
    inside = np.abs(px_err) <= snapshot.spread / 2.0

    return FitReport(
        n=len(snapshot),
        rmse_vol_bps=float(np.sqrt(np.sum(w * err ** 2))),
        mae_vol_bps=float(np.sum(w * np.abs(err))),
        max_vol_bps=float(np.max(np.abs(err))),
        rmse_price=float(np.sqrt(np.mean(px_err ** 2))),
        pct_inside_spread=float(100.0 * inside.mean()),
    )
