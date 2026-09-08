"""Pricing a named option off a fitted surface.

These run against a joint SSVI surface rather than the network on purpose: the
pricer is supposed to work for anything implementing `VolSurface`, and fitting
SSVI to the synthetic chain costs a fraction of a training run.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pytest

from volsurface.pricing.blackscholes import price as bs_price
from volsurface.pricing.option import OptionQuote, price_option, resolve_maturity
from volsurface.surfaces.svi import SSVISurface


@pytest.fixture(scope="module")
def surface(clean_chain):
    return SSVISurface.fit(clean_chain)


@pytest.fixture(scope="module")
def mid_expiry(clean_chain):
    """A listed maturity in the middle of the term structure, as a date."""
    T = float(clean_chain.maturities[len(clean_chain.maturities) // 2])
    return clean_chain.asof + timedelta(days=round(T * 365)), T


# -- naming the expiry --------------------------------------------------------

def test_a_date_is_measured_from_the_chain_not_from_today(clean_chain):
    """A surface fitted last week prices a December expiry at last week's T.

    Measuring against `date.today()` would silently shorten every maturity by
    the age of the chain -- and would make the same call return a different
    number tomorrow with no new data.
    """
    expiry = clean_chain.asof + timedelta(days=180)
    T, resolved = resolve_maturity(expiry, clean_chain.asof)

    assert resolved == expiry
    assert T == pytest.approx(180 / 365.0)


def test_the_four_ways_of_naming_an_expiry_agree(clean_chain):
    asof = clean_chain.asof
    expiry = asof + timedelta(days=105)
    years = 105 / 365.0

    forms = [
        expiry,
        datetime(expiry.year, expiry.month, expiry.day, 13, 30),
        expiry.isoformat(),
        "105d",
        years,
        str(years),
    ]
    got = [resolve_maturity(f, asof)[0] for f in forms]
    assert all(t == pytest.approx(years) for t in got)


def test_a_year_fraction_carries_no_expiry_date(clean_chain):
    """`expiry` is None when the caller never gave a calendar date to report."""
    T, resolved = resolve_maturity(0.25, clean_chain.asof)
    assert T == 0.25 and resolved is None


def test_a_four_digit_year_is_not_a_maturity(clean_chain):
    """'2026' parses as a float, so dates must be tried first."""
    with pytest.raises(ValueError):
        resolve_maturity("2026", clean_chain.asof)


def test_unreadable_expiry_says_what_it_accepts(clean_chain):
    with pytest.raises(ValueError, match="year fraction"):
        resolve_maturity("next friday", clean_chain.asof)


def test_unsupported_expiry_type_is_a_type_error(clean_chain):
    with pytest.raises(TypeError):
        resolve_maturity([2026, 12, 19], clean_chain.asof)


# -- guards -------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["c", "CALL", "Calls", "p", "put", "PUTS"])
def test_side_spelling_is_forgiving(clean_chain, surface, kind, mid_expiry):
    expiry, _ = mid_expiry
    quote = price_option(surface, clean_chain, clean_chain.spot, expiry, kind)
    assert quote.kind in ("call", "put")


def test_a_nonsense_side_is_refused(clean_chain, surface, mid_expiry):
    expiry, _ = mid_expiry
    with pytest.raises(ValueError, match="call.*put"):
        price_option(surface, clean_chain, clean_chain.spot, expiry, "banana")


def test_non_positive_strike_is_refused(clean_chain, surface, mid_expiry):
    expiry, _ = mid_expiry
    with pytest.raises(ValueError, match="strike"):
        price_option(surface, clean_chain, -10.0, expiry, "call")


def test_an_expired_option_is_refused_rather_than_priced(clean_chain, surface):
    """There is no implied vol at T <= 0, and returning intrinsic value here
    would hand back a number that looks like a surface price."""
    with pytest.raises(ValueError, match="maturity must be positive"):
        price_option(surface, clean_chain, clean_chain.spot,
                     clean_chain.asof - timedelta(days=1), "call")


# -- the number ---------------------------------------------------------------

def test_put_call_parity_holds_off_the_surface(clean_chain, surface, mid_expiry):
    """C - P = DF*(F - K) is the check that the discounting is applied once and
    to both sides. A parity break here means the forward or the discount factor
    disagrees between the two calls."""
    expiry, _ = mid_expiry
    K = clean_chain.spot * 1.02

    call = price_option(surface, clean_chain, K, expiry, "call")
    put = price_option(surface, clean_chain, K, expiry, "put")

    assert call.price - put.price == pytest.approx(
        call.discount * (call.forward - K), abs=1e-10
    )


def test_price_is_black76_at_the_surface_vol(clean_chain, surface, mid_expiry):
    """The pricer must not quietly substitute a forward of its own."""
    expiry, T = mid_expiry
    K = clean_chain.spot * 0.97
    quote = price_option(surface, clean_chain, K, expiry, "put")

    F, DF = clean_chain.forward_at(quote.T)
    expected = DF * bs_price(F, K, quote.T, 0.0, quote.implied_vol, "put")

    assert quote.forward == F and quote.discount == DF
    assert quote.price == pytest.approx(expected, rel=1e-12)


def test_implied_vol_is_the_surface_at_that_exact_point(clean_chain, surface, mid_expiry):
    expiry, _ = mid_expiry
    K = clean_chain.spot * 1.05
    quote = price_option(surface, clean_chain, K, expiry, "call")

    k = np.array([np.log(K / quote.forward)])
    T = np.array([quote.T])
    assert quote.implied_vol == pytest.approx(float(surface.implied_vol(k, T)[0]))


def test_calls_get_cheaper_as_the_strike_rises(clean_chain, surface, mid_expiry):
    expiry, _ = mid_expiry
    strikes = clean_chain.spot * np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    prices = [price_option(surface, clean_chain, K, expiry, "call").price for K in strikes]
    assert all(a > b for a, b in zip(prices, prices[1:]))


def test_spot_delta_rescales_the_forward_delta(clean_chain, surface, mid_expiry):
    expiry, _ = mid_expiry
    quote = price_option(surface, clean_chain, clean_chain.spot, expiry, "call")
    assert quote.delta_spot == pytest.approx(
        quote.delta * quote.forward / clean_chain.spot
    )


def test_greeks_all_carry_the_discount_factor(clean_chain, surface, mid_expiry):
    """Mixing discounted and undiscounted sensitivities in one row is a ~2%
    error on delta at an 18-month expiry, and it is invisible in the output."""
    expiry, _ = mid_expiry
    quote = price_option(surface, clean_chain, clean_chain.spot, expiry, "call")

    from volsurface.pricing.blackscholes import greeks

    raw = greeks(quote.forward, quote.strike, quote.T, 0.0, quote.implied_vol)
    assert quote.delta == pytest.approx(quote.discount * raw["delta_call"])
    assert quote.gamma == pytest.approx(quote.discount * raw["gamma"])
    assert quote.vega == pytest.approx(quote.discount * raw["vega"])
    assert quote.theta == pytest.approx(quote.discount * raw["theta_call"])


# -- whether to believe it ----------------------------------------------------

def test_a_quoted_option_is_not_flagged_as_extrapolated(clean_chain, surface, mid_expiry):
    expiry, _ = mid_expiry
    K = float(np.median(clean_chain.strike))
    quote = price_option(surface, clean_chain, K, expiry, "call")

    assert quote.in_quoted_strikes and quote.in_quoted_maturities
    assert not quote.is_extrapolated
    assert "EXTRAPOLATED" not in quote.summary()


def test_a_far_strike_and_a_far_expiry_are_both_flagged(clean_chain, surface):
    """The number is still bounded to the prior out here -- but it is the
    model's opinion, not the market's, and the quote has to say so."""
    far = clean_chain.asof + timedelta(days=round(clean_chain.T.max() * 365) + 400)
    quote = price_option(surface, clean_chain, clean_chain.spot * 0.2, far, "put")

    assert not quote.in_quoted_strikes
    assert not quote.in_quoted_maturities
    assert quote.is_extrapolated
    assert "EXTRAPOLATED" in quote.summary()


def test_a_clean_surface_reports_no_local_violation(clean_chain, surface, mid_expiry):
    """`clean_chain` comes from an arbitrage-free SSVI surface, so an SSVI fit
    to it must satisfy both conditions at any point asked about."""
    expiry, _ = mid_expiry
    for mult in (0.85, 1.0, 1.15):
        quote = price_option(surface, clean_chain, clean_chain.spot * mult, expiry, "call")
        assert quote.dw_dT >= 0.0
        assert quote.butterfly_g >= 0.0
        assert quote.arbitrage_free


def test_the_quote_is_immutable(clean_chain, surface, mid_expiry):
    """A priced quote is a record of an answer, not a mutable scratchpad."""
    expiry, _ = mid_expiry
    quote = price_option(surface, clean_chain, clean_chain.spot, expiry, "call")
    assert isinstance(quote, OptionQuote)
    with pytest.raises(Exception):
        quote.price = 0.0


def test_summary_names_the_option_it_priced(clean_chain, surface, mid_expiry):
    expiry, _ = mid_expiry
    quote = price_option(surface, clean_chain, 123.5, expiry, "put")
    text = quote.summary()

    assert "123.5" in text
    assert "put" in text
    assert expiry.isoformat() in text
    assert clean_chain.ticker in text
