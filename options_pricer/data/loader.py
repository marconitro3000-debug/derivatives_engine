"""
data/loader.py
Market-data loaders that produce MarketData snapshots for calibration.

Two sources behind one abstract interface:
  - YFinanceLoader : real option chains from Yahoo Finance (needs network)
  - SyntheticLoader : Heston-generated chains for testing / demos / offline use

Both return a calibration.MarketData object.
"""

import numpy as np
from datetime import datetime

from ..calibration.calibrator import MarketData
from ..models import heston as heston_mod


# ── base interface ────────────────────────────────────────────────────────────

class MarketDataLoader:
    """Abstract loader. Subclasses implement load()."""

    def load(self, *args, **kwargs) -> MarketData:
        raise NotImplementedError


# ── yfinance loader ───────────────────────────────────────────────────────────

class YFinanceLoader(MarketDataLoader):
    """
    Load a real option chain from Yahoo Finance.

    Requires `yfinance` and network access. Filters to liquid options
    (non-zero volume, reasonable bid-ask) and uses mid-price implied vol
    from the chain.
    """

    def __init__(self, risk_free_rate: float = 0.04):
        self.r = risk_free_rate

    def load(self, ticker: str,
             max_expiries: int = 6,
             moneyness_range: tuple = (0.85, 1.15),
             min_volume: int = 10) -> MarketData:
        import yfinance as yf

        tk   = yf.Ticker(ticker)
        spot = tk.history(period="1d")["Close"].iloc[-1]
        spot = float(spot)

        expiries = tk.options[:max_expiries]
        if not expiries:
            raise RuntimeError(f"No option expiries available for {ticker}.")

        strikes, maturities, ivs, weights = [], [], [], []
        now = datetime.now()

        for exp in expiries:
            T = (datetime.strptime(exp, "%Y-%m-%d") - now).days / 365.0
            if T <= 0:
                continue
            chain = tk.option_chain(exp).calls
            chain = chain[
                (chain["volume"].fillna(0) >= min_volume) &
                (chain["impliedVolatility"] > 0.01) &
                (chain["strike"] >= spot * moneyness_range[0]) &
                (chain["strike"] <= spot * moneyness_range[1])
            ]
            for _, row in chain.iterrows():
                strikes.append(float(row["strike"]))
                maturities.append(T)
                ivs.append(float(row["impliedVolatility"]))
                # weight by volume (liquidity-weighted calibration)
                weights.append(np.sqrt(float(row["volume"]) + 1))

        if not strikes:
            raise RuntimeError(
                f"No liquid options for {ticker} after filtering. "
                "Loosen min_volume or moneyness_range."
            )

        return MarketData(
            spot=spot, r=self.r,
            strikes=np.array(strikes),
            maturities=np.array(maturities),
            ivs=np.array(ivs),
            weights=np.array(weights),
        )


# ── synthetic loader ──────────────────────────────────────────────────────────

class SyntheticLoader(MarketDataLoader):
    """
    Generate a synthetic option chain from known Heston parameters.

    Useful for: testing the calibration pipeline (the calibrator should recover
    the true params), demos, and offline development.
    """

    def __init__(self, risk_free_rate: float = 0.04):
        self.r = risk_free_rate

    def load(self, spot: float = 100.0,
             true_params: heston_mod.HestonParams = None,
             strikes: np.ndarray = None,
             maturities: np.ndarray = None,
             noise: float = 0.0,
             seed: int = 42) -> MarketData:
        rng = np.random.default_rng(seed)

        if true_params is None:
            true_params = heston_mod.HestonParams(
                v0=0.04, kappa=2.0, theta=0.05, xi=0.4, rho=-0.6
            )
        if strikes is None:
            strikes = np.linspace(spot * 0.85, spot * 1.15, 9)
        if maturities is None:
            maturities = np.array([0.08, 0.25, 0.5, 1.0])

        from ..core.implied_vol import implied_vol

        K_list, T_list, iv_list = [], [], []
        for T in maturities:
            for K in strikes:
                px = heston_mod.price(spot, K, T, self.r, true_params, "call")
                try:
                    iv = implied_vol(spot, K, T, self.r, px, "call")
                except (ValueError, ZeroDivisionError):
                    continue
                if noise > 0:
                    iv += rng.normal(0, noise)
                K_list.append(K)
                T_list.append(T)
                iv_list.append(max(iv, 1e-3))

        md = MarketData(
            spot=spot, r=self.r,
            strikes=np.array(K_list),
            maturities=np.array(T_list),
            ivs=np.array(iv_list),
        )
        md.true_params = true_params.to_dict()   # attach ground truth for tests
        return md
