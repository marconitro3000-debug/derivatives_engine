"""
volsurface/data/synthetic.py
The offline chain generator: a chain sampled from a known SSVI surface.

This is what `use_live_data = False` and the whole test suite run against, and
it earns its place for a reason beyond convenience. The ground-truth surface is
arbitrage-free by construction, so any violation the diagnostics report on a fit
of this data is a defect in the *model*, not a feature of the market -- which is
exactly the question a live chain cannot answer.

Worth knowing about the results on it: per-slice SVI edges the network out on
RMSE here and violates nothing. That is the expected outcome and not a
disappointment -- the network earns its keep on real quotes, not on data a
parametric model already describes exactly.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from volsurface.data.snapshot import ChainSnapshot
from volsurface.pricing.blackscholes import greeks, price as bs_price


def synthetic_snapshot(
    *,
    ticker: str = "SYN",
    spot: float = 100.0,
    rate: float = 0.03,
    n_expiries: int = 6,
    n_strikes: int = 21,
    rho: float = -0.65,
    eta: float = 1.0,
    gamma: float = 0.35,
    atm_vol: float = 0.20,
    noise_bps: float = 0.0,
    seed: int = 0,
) -> ChainSnapshot:
    """A chain generated from a known arbitrage-free SSVI surface.

    Used by the tests and by ``use_live_data = False``: because the
    ground-truth surface is known and provably arbitrage-free, any violation the
    diagnostics report on a fit of this data is a defect in the *model*, not a
    feature of the market.
    """
    from volsurface.surfaces.svi import SSVIParams

    rng = np.random.default_rng(seed)
    ssvi = SSVIParams(rho=rho, eta=eta, gamma=gamma)
    asof = date(2025, 1, 2)

    maturities = np.linspace(0.08, 1.5, n_expiries)
    k_grid = np.linspace(-0.45, 0.35, n_strikes)

    k_all, T_all, iv_all, K_all, call_all, mid_all, spr_all, vega_all = ([] for _ in range(8))
    forwards, discounts = {}, {}

    for T in maturities:
        T = float(round(T, 6))
        DF = float(np.exp(-rate * T))
        F = spot / DF
        forwards[T], discounts[T] = F, DF

        theta = atm_vol ** 2 * T
        w = ssvi.total_var(k_grid, theta)
        iv = np.sqrt(np.maximum(w, 1e-12) / T)
        if noise_bps:
            iv = iv + rng.normal(0.0, noise_bps / 10_000.0, size=iv.shape)

        for k, sigma in zip(k_grid, iv):
            K = float(F * np.exp(k))
            is_call = bool(k >= 0)
            option = "call" if is_call else "put"
            fwd_price = bs_price(F, K, T, 0.0, float(sigma), option)
            v = greeks(F, K, T, 0.0, float(sigma))["vega"] * 100.0 * DF

            k_all.append(float(k)); T_all.append(T); iv_all.append(float(sigma))
            K_all.append(K); call_all.append(is_call)
            mid_all.append(fwd_price * DF); spr_all.append(0.02); vega_all.append(float(v))

    vega = np.array(vega_all)
    raw_w = vega / np.maximum(np.array(spr_all), 0.01)
    return ChainSnapshot(
        ticker=ticker, asof=asof, spot=spot,
        k=np.array(k_all), T=np.array(T_all), iv=np.array(iv_all),
        iv_european=np.array(iv_all),      # generated as European by construction
        weight=raw_w / raw_w.mean(),
        strike=np.array(K_all), is_call=np.array(call_all, dtype=bool),
        mid=np.array(mid_all), mid_european=np.array(mid_all),
        spread=np.array(spr_all), vega=vega,
        forwards=forwards, discounts=discounts,
    )
