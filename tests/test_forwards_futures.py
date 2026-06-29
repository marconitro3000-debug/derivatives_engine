import math

import pytest

from forwards import ForwardContract, forward_price, forward_value, implied_carry_rate
from futures import FuturesContract, annualized_basis, futures_price, mark_to_market_pnl


def test_forward_price_with_dividend_yield():
    fwd = forward_price(spot=100, maturity=1.0, rate=0.05, income_yield=0.02)

    assert fwd == pytest.approx(100 * math.exp(0.03))


def test_forward_value_is_zero_at_fair_delivery_price():
    delivery = forward_price(100, 0.5, 0.04)
    value = forward_value(100, delivery, 0.5, 0.04)

    assert value == pytest.approx(0.0)


def test_short_forward_value_is_opposite_long():
    long_value = forward_value(110, 100, 1.0, 0.05, position="long")
    short_value = forward_value(110, 100, 1.0, 0.05, position="short")

    assert short_value == pytest.approx(-long_value)


def test_implied_carry_rate_round_trip():
    fwd = forward_price(100, 2.0, 0.05, income_yield=0.01, storage_cost=0.02)

    assert implied_carry_rate(100, fwd, 2.0) == pytest.approx(0.06)


def test_forward_contract_value_scales_by_notional():
    contract = ForwardContract("OIL", delivery_price=75, maturity=1.0, notional=1000)

    assert contract.value(spot=80, rate=0.03) == pytest.approx(
        1000 * forward_value(80, 75, 1.0, 0.03)
    )


def test_futures_price_matches_forward_with_deterministic_rates():
    assert futures_price(100, 1.0, 0.05, income_yield=0.02) == pytest.approx(
        forward_price(100, 1.0, 0.05, income_yield=0.02)
    )


def test_mark_to_market_pnl_long_and_short():
    assert mark_to_market_pnl(100, 103, contracts=2, multiplier=50, position="long") == 300
    assert mark_to_market_pnl(100, 103, contracts=2, multiplier=50, position="short") == -300


def test_annualized_basis():
    assert annualized_basis(100, 105, 1.0) == pytest.approx(math.log(1.05))


def test_futures_contract_mtm_uses_multiplier():
    contract = FuturesContract("ES", price=5000, maturity=0.25, contracts=2, multiplier=50)

    assert contract.mtm_pnl(5010) == 1000
