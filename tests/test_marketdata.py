"""Chain cleaning: the forward fit, the filters, and the IVs that come out."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from volsurface.american import binomial_price, carry_from_forward
from volsurface.chain import build_snapshot, implied_forward
from volsurface.blackscholes import price as bs_price


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
        max_rel_spread=1.0, de_americanize=False,
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
        max_rel_spread=1.0, de_americanize=False,
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
                           max_rel_spread=1.0, de_americanize=False)

    calls.loc[0, "bid"] = 0.0                       # one-sided
    calls.loc[1, "ask"] = calls.loc[1, "bid"] * 10  # absurdly wide
    loose = build_snapshot("TEST", 100.0, asof, {expiry: (calls, puts)},
                           max_rel_spread=0.25, de_americanize=False)

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


# -- de-Americanisation -------------------------------------------------------

def _american_expiry(S, F, DF, T, sigma, strikes):
    """Quotes priced as genuine American options on a lattice."""
    r, q = carry_from_forward(S, F, DF, T)
    px = lambda K, opt: binomial_price(S, K, T, r, sigma, opt, style="american",
                                       n_steps=400, q=q)["price"]
    return (_frame(strikes, [px(K, "call") for K in strikes]),
            _frame(strikes, [px(K, "put") for K in strikes]))


def test_de_americanisation_recovers_the_vol_from_american_quotes():
    """Price American at 25% vol; the chain must give back 25%, not more.

    Without the correction the European inversion has nowhere to put the
    early-exercise premium and reports a higher volatility -- the bias this
    whole mechanism exists to remove.
    """
    asof, expiry, sigma = date(2025, 1, 2), "2026-07-02", 0.25
    T = 546 / 365.0
    DF = float(np.exp(-0.045 * T))
    S = 100.0
    F = S * np.exp((0.045 - 0.01) * T)
    strikes = np.arange(70.0, 136.0, 5.0)
    chains = {expiry: _american_expiry(S, F, DF, T, sigma, strikes)}

    corrected = build_snapshot("TEST", S, asof, chains, max_rel_spread=1.0,
                               de_americanize=True, lattice_steps=200)
    naive = build_snapshot("TEST", S, asof, chains, max_rel_spread=1.0,
                           de_americanize=False)

    err_corrected = np.abs(corrected.iv - sigma).max()
    err_naive = np.abs(naive.iv - sigma).max()
    assert err_corrected < 2e-3                       # under 20bp on every quote
    assert err_corrected < err_naive / 2              # and less than half the naive error


def test_de_americanisation_also_fixes_the_forward():
    """Put-call parity is a theorem about European options, so the forward fitted
    from raw American mids is biased -- by close to 1% here. The iterated fit
    must remove most of that."""
    asof, expiry, sigma = date(2025, 1, 2), "2026-07-02", 0.25
    T = 546 / 365.0
    r_true, q_true = 0.045, 0.01
    DF_true = float(np.exp(-r_true * T))
    S = 100.0
    F_true = S * np.exp((r_true - q_true) * T)
    strikes = np.arange(70.0, 136.0, 5.0)
    chains = {expiry: _american_expiry(S, F_true, DF_true, T, sigma, strikes)}

    corrected = build_snapshot("TEST", S, asof, chains, max_rel_spread=1.0,
                               de_americanize=True, lattice_steps=200)
    naive = build_snapshot("TEST", S, asof, chains, max_rel_spread=1.0,
                           de_americanize=False)

    Tk = corrected.maturities[0]
    err_corrected = abs(corrected.forwards[Tk] - F_true)
    err_naive = abs(naive.forwards[Tk] - F_true)

    assert err_naive > 0.5                            # the bias is real, not noise
    assert err_corrected < err_naive / 3
    assert abs(corrected.discounts[Tk] - DF_true) < abs(naive.discounts[Tk] - DF_true) / 3


def test_the_bias_lands_on_puts_not_calls():
    """Early exercise of an American call is worthless unless q > r; puts differ."""
    asof, expiry = date(2025, 1, 2), "2026-07-02"
    T = 546 / 365.0
    DF = float(np.exp(-0.045 * T))
    S = 100.0
    F = S * np.exp((0.045 - 0.01) * T)
    strikes = np.arange(70.0, 136.0, 5.0)

    snap = build_snapshot("TEST", S, asof,
                          {expiry: _american_expiry(S, F, DF, T, 0.25, strikes)},
                          max_rel_spread=1.0, de_americanize=True, lattice_steps=200)

    bias = snap.early_exercise_bp
    assert bias[~snap.is_call].mean() > bias[snap.is_call].mean()
    assert np.all(bias >= -1e-9)


def test_synthetic_chain_reports_no_early_exercise_bias(clean_chain):
    """The generated chain is European by construction, so the bias must be zero."""
    assert np.allclose(clean_chain.early_exercise_bp, 0.0)
    assert "off" in clean_chain.early_exercise_summary()
