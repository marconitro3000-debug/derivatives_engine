import math

import pytest

from rates import (
    DiscountCurve,
    InterestRateSwap,
    bootstrap_deposit_swap_curve,
    fixed_leg_pv,
    floating_leg_pv,
    present_value,
    swap_pv,
)


def test_flat_curve_discount_factor():
    curve = DiscountCurve.flat(0.05, max_maturity=10.0)

    assert curve.discount_factor(2.0) == pytest.approx(math.exp(-0.10))


def test_flat_curve_zero_and_forward_rates():
    curve = DiscountCurve.flat(0.05, max_maturity=10.0)

    assert curve.zero_rate(3.0) == pytest.approx(0.05)
    assert curve.forward_rate(1.0, 2.0) == pytest.approx(0.05)


def test_present_value_cashflows():
    curve = DiscountCurve.flat(0.05, max_maturity=10.0)
    pv = present_value([(1.0, 100.0), (2.0, 100.0)], curve)

    assert pv == pytest.approx(100 * math.exp(-0.05) + 100 * math.exp(-0.10))


def test_bootstrap_deposit_quote():
    curve = bootstrap_deposit_swap_curve(deposit_quotes={0.5: 0.04})

    assert curve.discount_factor(0.5) == pytest.approx(1 / (1 + 0.04 * 0.5))


def test_bootstrap_swap_reprices_par_rate():
    curve = bootstrap_deposit_swap_curve(
        deposit_quotes={0.5: 0.04, 1.0: 0.042},
        swap_quotes={2.0: 0.045, 3.0: 0.047},
        fixed_freq=1,
    )

    assert curve.par_swap_rate(2.0, pay_freq=1) == pytest.approx(0.045)
    assert curve.par_swap_rate(3.0, pay_freq=1) == pytest.approx(0.047)


def test_par_swap_has_zero_value():
    curve = DiscountCurve.flat(0.05, max_maturity=10.0)
    fixed_rate = curve.par_swap_rate(5.0, pay_freq=1)

    assert swap_pv(1_000_000, fixed_rate, 5.0, curve, pay_freq=1) == pytest.approx(0.0, abs=1e-6)


def test_payer_swap_benefits_when_fixed_below_market():
    curve = DiscountCurve.flat(0.05, max_maturity=10.0)
    fixed_rate = curve.par_swap_rate(5.0, pay_freq=1) - 0.01

    assert swap_pv(1_000_000, fixed_rate, 5.0, curve, pay_freq=1, position="payer") > 0


def test_swap_class_matches_functions():
    curve = DiscountCurve.flat(0.05, max_maturity=10.0)
    swap = InterestRateSwap(1_000_000, 0.04, 5.0, pay_freq=1)

    assert swap.fixed_leg_pv(curve) == pytest.approx(fixed_leg_pv(1_000_000, 0.04, 5.0, curve))
    assert swap.floating_leg_pv(curve) == pytest.approx(floating_leg_pv(1_000_000, 5.0, curve))
    assert swap.pv(curve) == pytest.approx(swap_pv(1_000_000, 0.04, 5.0, curve))
