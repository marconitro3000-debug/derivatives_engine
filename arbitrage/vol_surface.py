"""
arbitrage/vol_surface.py
Static arbitrage detection in an implied-volatility surface.

Two classical no-arbitrage conditions on the total-variance surface w(k,T):

1. Calendar-spread arbitrage
   For T₁ < T₂:  w(k, T₁) ≤ w(k, T₂)  for all k.
   Violation → a calendar spread has negative time value (free money).

2. Butterfly arbitrage
   The risk-neutral density g(k,T) must be non-negative everywhere:
       g = (1 − k·∂w_k/(2w))² − (∂w_k)²·(¼ + 1/w)/4 + ∂²w_k/2 ≥ 0
   Violation → a butterfly spread has negative cost (free money).

Workflow
--------
1. Download a live option chain via YFinance.
2. Fit raw SVI (Gatheral 2004) to each maturity slice independently.
3. Evaluate both conditions on a dense log-moneyness grid.
4. Report all violations with strike, maturity, magnitude, and profit estimate.

References
----------
Gatheral & Jacquier (2014) "Arbitrage-Free SVI Volatility Surfaces", Quant Finance.
Roper (2010) "Arbitrage Free Implied Volatility Surfaces".
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from core.daycount import year_fraction
from ml.ssvi import SVIParams, calibrate_svi


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class CalendarViolation:
    """A region where w(k, T_short) > w(k, T_long)."""
    T_short: float
    T_long: float
    worst_k: float          # log-moneyness of largest violation
    magnitude: float        # max excess total variance (total-var units)
    n_violations: int       # number of grid points violated
    iv_spread_bps: float    # approx IV profit at worst_k (basis points)

    def __str__(self) -> str:
        return (f"CALENDAR  {self.T_short:.3f}→{self.T_long:.3f}Y  "
                f"k={self.worst_k:+.3f}  Δw={self.magnitude:.5f}  "
                f"≈{self.iv_spread_bps:.0f}bp  n={self.n_violations}")


@dataclass
class ButterflyViolation:
    """A region where the risk-neutral density g(k,T) < 0."""
    T: float
    worst_k: float          # log-moneyness of most negative density
    min_density: float      # minimum value of g(k)
    n_violations: int


    def __str__(self) -> str:
        return (f"BUTTERFLY T={self.T:.3f}Y  "
                f"k={self.worst_k:+.3f}  g_min={self.min_density:.5f}  "
                f"n={self.n_violations}")


@dataclass
class SliceFit:
    """SVI calibration result for a single maturity slice."""
    T: float
    params: SVIParams
    rmse_iv: float       # root-mean-square IV error (absolute, e.g. 0.002 = 20bp)
    max_err_iv: float    # worst per-point IV error
    n_points: int
    arb_free: bool       # Lee wing-condition check on params


@dataclass
class ArbScanResult:
    """Full arbitrage scan output."""
    ticker: str
    spot: float
    scan_time: str
    slices: dict[float, SliceFit]         # T → SliceFit
    k_grid: np.ndarray
    total_var_matrix: np.ndarray          # shape (n_T, n_k) — w(k, T)
    density_matrix: np.ndarray            # shape (n_T, n_k) — g(k, T)
    calendar_violations: list[CalendarViolation] = field(default_factory=list)
    butterfly_violations: list[ButterflyViolation] = field(default_factory=list)

    @property
    def maturities(self) -> np.ndarray:
        return np.array(sorted(self.slices.keys()))

    @property
    def is_calendar_arb_free(self) -> bool:
        return len(self.calendar_violations) == 0

    @property
    def is_butterfly_arb_free(self) -> bool:
        return len(self.butterfly_violations) == 0

    @property
    def is_arbitrage_free(self) -> bool:
        return self.is_calendar_arb_free and self.is_butterfly_arb_free

    def summary(self) -> str:
        T_list = [f"{T:.3f}" for T in self.maturities]
        lines = [
            f"━━━ Vol Surface Arb Scan ━━━",
            f"  Ticker : {self.ticker}  (S = {self.spot:.2f})",
            f"  Time   : {self.scan_time}",
            f"  Slices : {T_list}",
            "",
            f"  SVI fit quality:",
        ]
        for T, sf in sorted(self.slices.items()):
            arb = "✓" if sf.arb_free else "✗"
            lines.append(
                f"    T={T:.3f}Y  RMSE={sf.rmse_iv*1e4:.1f}bp  "
                f"MaxErr={sf.max_err_iv*1e4:.1f}bp  n={sf.n_points}  arb_free={arb}"
            )
        lines += [""]
        cal_status  = "CLEAN ✓" if self.is_calendar_arb_free  else f"{len(self.calendar_violations)} VIOLATIONS ✗"
        but_status  = "CLEAN ✓" if self.is_butterfly_arb_free else f"{len(self.butterfly_violations)} VIOLATIONS ✗"
        lines += [
            f"  Calendar arbitrage : {cal_status}",
            f"  Butterfly arbitrage: {but_status}",
        ]
        for v in self.calendar_violations:
            lines.append(f"    → {v}")
        for v in self.butterfly_violations:
            lines.append(f"    → {v}")
        return "\n".join(lines)


# ── Core scanner ──────────────────────────────────────────────────────────────

class VolSurfaceArbScanner:
    """
    Fits SVI to each maturity slice then checks for static arbitrage.

    Usage
    -----
    scanner = VolSurfaceArbScanner()
    result  = scanner.scan_from_chain(spot, strikes, maturities, ivs, ticker="SPY")
    print(result.summary())
    """

    def __init__(self, k_grid: Optional[np.ndarray] = None):
        # dense log-moneyness grid for evaluation
        self.k_grid = k_grid if k_grid is not None else np.linspace(-0.45, 0.45, 901)

    # ── public entry points ───────────────────────────────────────────────────

    def scan_from_chain(
        self,
        spot: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        ivs: np.ndarray,
        r: float = 0.04,
        ticker: str = "UNK",
        weights: Optional[np.ndarray] = None,
    ) -> ArbScanResult:
        """
        Calibrate SVI slices then run the full arbitrage scan.

        Parameters
        ----------
        spot, strikes, maturities, ivs, r  — option chain (aligned arrays)
        ticker  — label for display
        weights — optional per-point calibration weights
        """
        unique_T = np.unique(maturities)
        slices: dict[float, SliceFit] = {}

        for T in unique_T:
            mask = maturities == T
            K_s  = strikes[mask]
            iv_s = ivs[mask]
            w_s  = weights[mask] if weights is not None else None

            F    = spot * np.exp(r * T)
            k    = np.log(K_s / F)
            w_mkt = iv_s ** 2 * T        # total implied variance

            params = calibrate_svi(k, w_mkt, weights=w_s)

            w_fit  = params.total_var(k)
            iv_fit = np.sqrt(np.maximum(w_fit / T, 0))
            errs   = np.abs(iv_fit - iv_s)

            slices[T] = SliceFit(
                T         = T,
                params    = params,
                rmse_iv   = float(np.sqrt(np.mean(errs ** 2))),
                max_err_iv= float(errs.max()),
                n_points  = int(mask.sum()),
                arb_free  = _lee_wing_check(params),
            )

        return self._build_result(slices, spot, ticker)

    def scan_live(
        self,
        ticker: str,
        r: float = 0.04,
        moneyness_range: tuple = (0.80, 1.20),
        min_volume: int = 5,
        max_expiries: int = 8,
    ) -> ArbScanResult:
        """
        Download a live option chain from Yahoo Finance and scan for arbitrage.

        Uses both calls and puts for better smile coverage.
        """
        spot, strikes, maturities, ivs, weights = _load_chain_yfinance(
            ticker, r, moneyness_range, min_volume, max_expiries
        )
        return self.scan_from_chain(
            spot, strikes, maturities, ivs, r=r, ticker=ticker, weights=weights
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_result(
        self, slices: dict[float, SliceFit], spot: float, ticker: str
    ) -> ArbScanResult:
        k_grid = self.k_grid
        sorted_T = sorted(slices.keys())
        n_T = len(sorted_T)

        # Evaluate total-variance matrix and density matrix
        tv_matrix  = np.zeros((n_T, len(k_grid)))
        den_matrix = np.zeros((n_T, len(k_grid)))

        for i, T in enumerate(sorted_T):
            p = slices[T].params
            tv_matrix[i]  = p.total_var(k_grid)
            den_matrix[i] = _density(p, k_grid)

        cal_viols = self._check_calendar(sorted_T, slices, k_grid, tv_matrix)
        but_viols = self._check_butterfly(sorted_T, slices, k_grid, den_matrix)

        return ArbScanResult(
            ticker    = ticker,
            spot      = spot,
            scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            slices    = slices,
            k_grid    = k_grid,
            total_var_matrix = tv_matrix,
            density_matrix   = den_matrix,
            calendar_violations  = cal_viols,
            butterfly_violations = but_viols,
        )

    def _check_calendar(
        self, sorted_T, slices, k_grid, tv_matrix
    ) -> list[CalendarViolation]:
        violations = []
        for i in range(len(sorted_T) - 1):
            T1, T2 = sorted_T[i], sorted_T[i + 1]
            w1 = tv_matrix[i]
            w2 = tv_matrix[i + 1]
            excess = w1 - w2              # positive where calendar arb exists
            mask   = excess > 0

            if not mask.any():
                continue

            worst_idx = int(np.argmax(excess))
            k_worst   = float(k_grid[worst_idx])
            mag       = float(excess[worst_idx])

            # Approximate IV spread in basis points at the worst point
            w_ref = float(w2[worst_idx])
            iv_bps = _iv_spread_bps(mag, w_ref, T1)

            violations.append(CalendarViolation(
                T_short      = T1,
                T_long       = T2,
                worst_k      = k_worst,
                magnitude    = mag,
                n_violations = int(mask.sum()),
                iv_spread_bps= iv_bps,
            ))
        return violations

    def _check_butterfly(
        self, sorted_T, slices, k_grid, den_matrix
    ) -> list[ButterflyViolation]:
        violations = []
        tol = -1e-4       # small negative tolerance to avoid flagging numerical noise

        for i, T in enumerate(sorted_T):
            g    = den_matrix[i]
            mask = g < tol

            if not mask.any():
                continue

            worst_idx = int(np.argmin(g))
            violations.append(ButterflyViolation(
                T            = T,
                worst_k      = float(k_grid[worst_idx]),
                min_density  = float(g[worst_idx]),
                n_violations = int(mask.sum()),
            ))
        return violations


# ── Helper functions ──────────────────────────────────────────────────────────

def _density(params: SVIParams, k: np.ndarray) -> np.ndarray:
    """
    Risk-neutral density proxy g(k) for a single SVI slice:
        g = (1 − k·∂w_k/(2w))² − (∂w_k)²·(¼ + 1/w)/4 + ∂²w_k/2

    g ≥ 0 everywhere ⟺ no butterfly arbitrage.
    Uses central finite differences on the SVI total-variance function.
    """
    h  = 1e-5
    k  = np.asarray(k, dtype=float)
    w0 = params.total_var(k)
    wp = params.total_var(k + h)
    wm = params.total_var(k - h)

    dw   = (wp - wm) / (2 * h)
    d2w  = (wp - 2 * w0 + wm) / h ** 2

    # Protect against w → 0 (far OTM, very short tenor)
    w0_safe = np.maximum(w0, 1e-12)
    return (
        (1 - k * dw / (2 * w0_safe)) ** 2
        - dw ** 2 * (0.25 + 1.0 / w0_safe) / 4.0
        + d2w / 2.0
    )


def _lee_wing_check(p: SVIParams) -> bool:
    """Lee's moment formula: necessary condition for no butterfly arbitrage."""
    if p.b < 0 or p.sigma <= 0 or abs(p.rho) >= 1:
        return False
    if p.b * (1 + abs(p.rho)) > 4.0:
        return False
    w_min = p.a + p.b * p.sigma * np.sqrt(1 - p.rho ** 2)
    return bool(w_min >= 0)


def _iv_spread_bps(delta_w: float, w_ref: float, T: float) -> float:
    """Convert Δw (total-variance units) to approximate IV spread in bps."""
    if w_ref <= 0 or T <= 0:
        return float("nan")
    iv_ref = np.sqrt(w_ref / T)
    # Δ(σ²T) ≈ 2σ·T·Δσ  →  Δσ ≈ Δw / (2·σ·T)
    delta_iv = delta_w / (2 * iv_ref * T)
    return float(delta_iv * 1e4)   # basis points


def _load_chain_yfinance(
    ticker: str,
    r: float,
    moneyness_range: tuple,
    min_volume: int,
    max_expiries: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Download calls + puts from Yahoo Finance and return aligned arrays.
    Uses the mid-point of bid/ask if available, otherwise yfinance's impliedVolatility.
    """
    import yfinance as yf

    tk    = yf.Ticker(ticker)
    spot  = float(tk.history(period="1d")["Close"].iloc[-1])
    today = datetime.now().date()

    all_expiries = tk.options
    if not all_expiries:
        raise RuntimeError(f"No option expiries for {ticker}.")

    expiries = []
    for exp in all_expiries:
        T = year_fraction(today, exp, "act365f")
        if T > 0.03:                    # at least ~10 calendar days
            expiries.append(exp)
        if len(expiries) >= max_expiries:
            break

    lo, hi = moneyness_range
    K_all, T_all, iv_all, w_all = [], [], [], []

    for exp in expiries:
        T = year_fraction(today, exp, "act365f")
        if T <= 0:
            continue

        chain = tk.option_chain(exp)
        for df in (chain.calls, chain.puts):
            df = df[
                (df["volume"].fillna(0) >= min_volume) &
                (df["impliedVolatility"] > 0.005) &
                (df["strike"] >= spot * lo) &
                (df["strike"] <= spot * hi)
            ].copy()

            for _, row in df.iterrows():
                iv = float(row["impliedVolatility"])
                K_all.append(float(row["strike"]))
                T_all.append(T)
                iv_all.append(iv)
                w_all.append(np.sqrt(float(row["volume"]) + 1))

    if not K_all:
        raise RuntimeError(
            f"No liquid options for {ticker}. Try loosening moneyness_range or min_volume."
        )

    return (
        spot,
        np.array(K_all),
        np.array(T_all),
        np.array(iv_all),
        np.array(w_all),
    )


# ── Convenience function ──────────────────────────────────────────────────────

def scan_ticker(ticker: str, r: float = 0.04, **kwargs) -> ArbScanResult:
    """One-liner: download live chain and run full arbitrage scan."""
    return VolSurfaceArbScanner().scan_live(ticker, r=r, **kwargs)
