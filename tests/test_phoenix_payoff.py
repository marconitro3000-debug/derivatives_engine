from __future__ import annotations

from dataclasses import replace

from structured.phoenix_autocall import PhoenixAutocallSpec, price_phoenix_autocall


def test_phoenix_autocalls_when_trigger_is_reached():
    spec = PhoenixAutocallSpec(
        spot=100,
        risk_free_rate=0.0,
        volatility=0.0,
        autocall_level=0.99,
        n_paths=1_000,
        steps_per_year=12,
        seed=1,
    )
    result = price_phoenix_autocall(spec)
    assert result["autocall_probability"] > 0.99
    assert result["expected_life_years"] <= 0.25


def test_lower_barrier_reduces_loss_probability():
    base = PhoenixAutocallSpec(spot=100, volatility=0.35, n_paths=5_000, steps_per_year=80, seed=7)
    high_barrier = price_phoenix_autocall(replace(base, capital_barrier=0.85, coupon_barrier=0.85))
    low_barrier = price_phoenix_autocall(replace(base, capital_barrier=0.60, coupon_barrier=0.60))
    assert high_barrier["prob_final_capital_loss"] >= low_barrier["prob_final_capital_loss"]


def test_memory_coupon_increases_expected_coupon():
    base = PhoenixAutocallSpec(
        spot=100,
        coupon_barrier=1.05,
        capital_barrier=0.70,
        autocall_level=1.50,
        volatility=0.20,
        n_paths=5_000,
        steps_per_year=80,
        seed=3,
    )
    no_memory = price_phoenix_autocall(replace(base, memory=False))
    memory = price_phoenix_autocall(replace(base, memory=True))
    assert memory["expected_coupon_pct"] >= no_memory["expected_coupon_pct"]


def test_higher_volatility_increases_barrier_touch_probability():
    base = PhoenixAutocallSpec(spot=100, capital_barrier=0.70, coupon_barrier=0.70, n_paths=5_000, steps_per_year=80, seed=11)
    low_vol = price_phoenix_autocall(replace(base, volatility=0.10))
    high_vol = price_phoenix_autocall(replace(base, volatility=0.45))
    assert high_vol["barrier_touch_probability"] >= low_vol["barrier_touch_probability"]
