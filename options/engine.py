"""
options/engine.py
High-level self-updating pricing engine.

This is the main user-facing entry point for options. It ties together:
  - data loaders (yfinance / synthetic)
  - calibration (SVI / Heston) with warm start + EWMA smoothing
  - SQLite persistence (versioned history)
  - pricing from the live calibrated model

Typical usage
-------------
    from options.engine import PricingEngine

    eng = PricingEngine(model="heston", db_path="my_cals.db")

    # First call: cold calibration from market data, persisted
    eng.update("AAPL")

    # Later calls: warm-started from the stored params, smoothed
    eng.update("AAPL", ewma_alpha=0.3)

    # Price any option from the live calibrated surface
    eng.price_option("AAPL", K=190, T=0.5, option="call")

    # Inspect how parameters have drifted
    eng.drift("AAPL", "kappa")
"""

import numpy as np

from core.calibration.calibrator import calibrate, MarketData, CalibrationResult
from core.calibration.store import CalibrationStore
from core.data.loader import YFinanceLoader, SyntheticLoader
from core.models import heston as heston_mod
from core.models import svi as svi_mod
from .black_scholes import price as bs_price


class PricingEngine:
    """Self-updating, self-persisting options pricing engine."""

    def __init__(self, model: str = "heston",
                 db_path: str = "calibrations.db",
                 risk_free_rate: float = 0.04,
                 source: str = "yfinance"):
        if model not in ("svi", "heston"):
            raise ValueError("model must be 'svi' or 'heston'.")
        self.model = model
        self.r     = risk_free_rate
        self.store = CalibrationStore(db_path)
        self._last_spot = {}

        if source == "yfinance":
            self.loader = YFinanceLoader(risk_free_rate)
        elif source == "synthetic":
            self.loader = SyntheticLoader(risk_free_rate)
        else:
            raise ValueError("source must be 'yfinance' or 'synthetic'.")

    # ── update / recalibrate ──────────────────────────────────────────────────

    def update(self, ticker: str,
               market_data: MarketData = None,
               warm_start: bool = True,
               ewma_alpha: float = None,
               **loader_kwargs) -> CalibrationResult:
        """
        Fetch fresh market data, (re)calibrate, and persist.

        Parameters
        ----------
        ticker      : symbol (key for storage and warm-start lookup)
        market_data : provide a MarketData directly (skips the loader)
        warm_start  : seed from the last stored calibration if available
        ewma_alpha  : smooth fitted params against the previous ones (0–1)
        loader_kwargs : passed through to the loader's load()

        Returns
        -------
        CalibrationResult
        """
        if market_data is None:
            if isinstance(self.loader, SyntheticLoader):
                market_data = self.loader.load(**loader_kwargs)
            else:
                market_data = self.loader.load(ticker, **loader_kwargs)

        prev = self.store.latest_params(ticker, self.model) if warm_start else None

        result = calibrate(self.model, market_data,
                           warm_start=prev, ewma_alpha=ewma_alpha)

        self.store.save(ticker, result, spot=market_data.spot, r=market_data.r)
        self._last_spot[ticker] = market_data.spot
        return result

    # ── pricing from the live model ───────────────────────────────────────────

    def _get_params_and_spot(self, ticker: str, spot: float = None):
        params = self.store.latest_params(ticker, self.model)
        if params is None:
            raise RuntimeError(
                f"No calibration for {ticker}/{self.model}. Call update() first."
            )
        if spot is None:
            spot = self._last_spot.get(ticker)
            if spot is None:
                hist = self.store.history(ticker, self.model, limit=1)
                spot = hist[0]["spot"] if hist and hist[0]["spot"] else None
            if spot is None:
                raise RuntimeError(
                    f"No spot price known for {ticker}. Pass spot= explicitly."
                )
        return params, spot

    def implied_vol(self, ticker: str, K: float, T: float,
                    spot: float = None) -> float:
        """Implied vol for (K, T) from the live calibrated model."""
        params, spot = self._get_params_and_spot(ticker, spot)

        if self.model == "svi":
            return self._svi_iv(params, K, T, spot)
        else:
            from .implied_vol import implied_vol as bs_iv
            p  = heston_mod.HestonParams.from_dict(params)
            px = heston_mod.price(spot, K, T, self.r, p, "call")
            return float(bs_iv(spot, K, T, self.r, px, "call"))

    def _svi_iv(self, params: dict, K: float, T: float, spot: float) -> float:
        """
        SVI implied vol with linear interpolation in total variance across
        the two nearest calibrated maturity slices.
        """
        slices = params["slices"]
        avail_T = sorted(float(t) for t in slices.keys())
        F = spot * np.exp(self.r * T)
        k = np.log(K / F)

        def slice_total_var(T_slice):
            p = svi_mod.SVIParams.from_dict(slices[f"{T_slice:.6f}"])
            return float(svi_mod.total_variance(np.array([k]), p)[0])

        if T <= avail_T[0]:
            w = slice_total_var(avail_T[0]) * (T / avail_T[0])
        elif T >= avail_T[-1]:
            w = slice_total_var(avail_T[-1]) * (T / avail_T[-1])
        else:
            lo = max(t for t in avail_T if t <= T)
            hi = min(t for t in avail_T if t >= T)
            if lo == hi:
                w = slice_total_var(lo)
            else:
                w_lo, w_hi = slice_total_var(lo), slice_total_var(hi)
                frac = (T - lo) / (hi - lo)
                w = w_lo + frac * (w_hi - w_lo)

        return float(np.sqrt(max(w, 1e-12) / T))

    def price_option(self, ticker: str, K: float, T: float,
                     option: str = "call", spot: float = None, q: float = 0.0) -> float:
        """
        Price an option from the live calibrated model.

        Heston prices directly via its characteristic function.
        SVI produces an IV which is then fed into Black-Scholes.
        `q` is the continuous dividend yield (default 0).
        """
        params, spot = self._get_params_and_spot(ticker, spot)

        if self.model == "heston":
            p = heston_mod.HestonParams.from_dict(params)
            return heston_mod.price(spot, K, T, self.r, p, option, q)
        else:
            iv = self.implied_vol(ticker, K, T, spot)
            return bs_price(spot, K, T, self.r, iv, option, q)

    # ── diagnostics ───────────────────────────────────────────────────────────

    def drift(self, ticker: str, param_name: str):
        """Time series of one parameter for drift analysis."""
        return self.store.param_timeseries(ticker, self.model, param_name)

    def history(self, ticker: str, limit: int = 100):
        """Full calibration history for this ticker/model."""
        return self.store.history(ticker, self.model, limit)
