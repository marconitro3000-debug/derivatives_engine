"""Black-Scholes pricing, Greeks and implied-vol inversion."""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.blackscholes import greeks, price, put_call_parity_check
from volsurface.impliedvol import implied_vol


def test_put_call_parity_holds_with_dividends():
    S, K, T, r, q, sigma = 100.0, 105.0, 0.75, 0.03, 0.02, 0.25
    c = price(S, K, T, r, sigma, "call", q)
    p = price(S, K, T, r, sigma, "put", q)
    assert put_call_parity_check(S, K, T, r, c, p, q)["error"] < 1e-10


def test_price_is_bounded_by_no_arbitrage_limits():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.04, 0.2
    c = price(S, K, T, r, sigma, "call")
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    assert intrinsic < c < S


def test_deep_itm_call_delta_approaches_one():
    d = greeks(100.0, 10.0, 1.0, 0.03, 0.2)
    assert d["delta_call"] == pytest.approx(np.exp(-0.0), abs=1e-6)
    assert d["delta_put"] == pytest.approx(0.0, abs=1e-6)


def test_vega_matches_finite_difference():
    S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.02, 0.3
    h = 1e-5
    fd = (price(S, K, T, r, sigma + h) - price(S, K, T, r, sigma - h)) / (2 * h)
    assert greeks(S, K, T, r, sigma)["vega"] * 100 == pytest.approx(fd, rel=1e-6)


@pytest.mark.parametrize(("K", "sigma"), [
    (100.0, 0.05), (100.0, 0.15), (100.0, 0.40), (100.0, 1.20),
    (140.0, 0.15), (140.0, 0.40), (140.0, 1.20),
    (70.0, 0.40), (70.0, 1.20),
    # (140, 0.05) and (70, 0.05) are deliberately absent: those prices carry no
    # recoverable volatility -- see the two identifiability tests below.
])
def test_implied_vol_roundtrips(K, sigma):
    """Price at a known vol, invert, and get the vol back."""
    S, T, r = 100.0, 0.6, 0.03
    px = price(S, K, T, r, sigma, "call")
    assert implied_vol(S, K, T, r, px, "call") == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_prices_below_intrinsic():
    with pytest.raises(ValueError, match="intrinsic"):
        implied_vol(100.0, 50.0, 1.0, 0.05, 1.0, "call")


def test_implied_vol_refuses_when_the_price_does_not_identify_a_vol():
    """A deep ITM call at low vol is worth intrinsic to the last bit of a float64.

    Every sigma below ~7% prices S=100, K=70, T=0.6 to the identical double, so
    no root finder can recover the input. The inversion must say so rather than
    return whichever value it landed on.
    """
    px = price(100.0, 70.0, 0.6, 0.03, 0.05, "call")
    with pytest.raises(ValueError, match="not identifiable"):
        implied_vol(100.0, 70.0, 0.6, 0.03, px, "call")


def test_deep_itm_is_still_invertible_at_a_high_vol():
    """The refusal is about vega, not about moneyness -- 40% vol inverts fine."""
    px = price(100.0, 70.0, 0.6, 0.03, 0.40, "call")
    assert implied_vol(100.0, 70.0, 0.6, 0.03, px, "call") == pytest.approx(0.40, abs=1e-6)
