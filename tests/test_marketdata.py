"""Chain cleaning: the forward fit, the filters, and the IVs that come out."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from marketdata.chain import build_snapshot, implied_forward
from options.black_scholes import price as bs_price


# -- put-call parity forward fit ----------------------------------------------

def test_implied_forward_recovers_a_known_forward():
    F, DF = 103.25, 0.985
    strikes = np.array([95.0, 100.0, 105.0, 110.0])
    calls = DF * np.array([bs_price(F, K, 0.5, 0.0, 0.2, "call") for K in strikes])
    puts = DF * np.array([bs_price(F, K, 0.5, 0.0, 0.2, "put") for K in strikes])

    F_hat, DF_hat = implied_forward(strikes, calls, puts)
    assert F_hat == pytest.approx(F, rel=1e-9)
    assert DF_hat == pytest.approx(DF, rel=1e-9)


def test_implied_forward_needs_enough_strikes():
    with pytest.raises(ValueError, match="at least 3"):
        implied_forward(np.array([100.0]), np.array([5.0]), np.array([4.0]))


def test_implied_forward_rejects_an_implausible_discount_factor():
    """A garbage regression must fail loudly, not return a nonsense forward."""
    strikes = np.array([90.0, 100.0, 110.0])
    with pytest.raises(ValueError, match="discount factor"):
        implied_forward(strikes, np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))


# -- end-to-end cleaning ------------------------------------------------------

def _frame(strikes, prices, half_spread=0.05, oi=500):
    return pd.DataFrame({
        "strike": strikes,
        "bid": np.maximum(np.asarray(prices) - half_spread, 0.01),
        "ask": np.asarray(prices) + half_spread,
        "openInterest": oi,
        "volume": 100,
    })


def _synthetic_expiry(F, DF, T, sigma, strikes):
    calls = [DF * bs_price(F, K, T, 0.0, sigma, "call") for K in strikes]
    puts = [DF * bs_price(F, K, T, 0.0, sigma, "put") for K in strikes]
    return _frame(strikes, calls), _frame(strikes, puts)


def test_build_snapshot_recovers_the_generating_vol():
    """Flat 22% vol in, ~22% vol out -- through parity, OTM selection and inversion."""
    asof, expiry, sigma = date(2025, 1, 2), "2025-07-02", 0.22
    T = 181 / 365.0
    DF = float(np.exp(-0.04 * T))
    F = 100.0 / DF
    strikes = np.arange(80.0, 125.0, 5.0)

    snap = build_snapshot(
        "TEST", 100.0, asof, {expiry: _synthetic_expiry(F, DF, T, sigma, strikes)},
        max_rel_spread=1.0,
    )

    assert len(snap) > 0
    assert snap.forwards[T] == pytest.approx(F, rel=1e-3)
    assert snap.discounts[T] == pytest.approx(DF, rel=1e-3)
    assert np.allclose(snap.iv, sigma, atol=2e-3)


def test_build_snapshot_takes_otm_options_only():
    """Calls above the forward, puts below -- never both at one strike."""
    asof, expiry = date(2025, 1, 2), "2025-07-02"
    T = 181 / 365.0
    DF = float(np.exp(-0.04 * T))
    F = 100.0 / DF
    strikes = np.arange(80.0, 125.0, 5.0)

    snap = build_snapshot(
        "TEST", 100.0, asof, {expiry: _synthetic_expiry(F, DF, T, 0.22, strikes)},
        max_rel_spread=1.0,
    )

    assert np.all(snap.strike[snap.is_call] >= F)
    assert np.all(snap.strike[~snap.is_call] < F)
    assert len(np.unique(snap.strike)) == len(snap.strike)


def test_wide_and_one_sided_quotes_are_dropped():
    asof, expiry = date(2025, 1, 2), "2025-07-02"
    T = 181 / 365.0
    DF = float(np.exp(-0.04 * T))
    F = 100.0 / DF
    strikes = np.arange(80.0, 125.0, 5.0)
    calls, puts = _synthetic_expiry(F, DF, T, 0.22, strikes)

    tight = build_snapshot("TEST", 100.0, asof, {expiry: (calls, puts)},
                           max_rel_spread=1.0)

    calls.loc[0, "bid"] = 0.0                       # one-sided
    calls.loc[1, "ask"] = calls.loc[1, "bid"] * 10  # absurdly wide
    loose = build_snapshot("TEST", 100.0, asof, {expiry: (calls, puts)},
                           max_rel_spread=0.25)

    assert len(loose) < len(tight)


def test_expiries_outside_the_maturity_window_are_skipped():
    asof = date(2025, 1, 2)
    T = 5 / 365.0                                    # inside min_T=0.02? no: 0.0137
    DF, F = 0.999, 100.1
    strikes = np.arange(90.0, 115.0, 5.0)
    with pytest.raises(RuntimeError, match="survived cleaning"):
        build_snapshot("TEST", 100.0, asof,
                       {"2025-01-07": _synthetic_expiry(F, DF, T, 0.22, strikes)})


def test_weights_favour_atm_over_the_wings(clean_chain):
    """Vega weighting must put more mass at the money than in the far wings."""
    for T in clean_chain.maturities:
        sl = clean_chain.slice_at(T)
        atm = np.argmin(np.abs(sl.k))
        wing = np.argmax(np.abs(sl.k))
        assert sl.weight[atm] > sl.weight[wing]
