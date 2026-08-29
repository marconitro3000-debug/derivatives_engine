"""The numerical pricers, anchored to the closed form they must converge to.

The binomial tree and the Monte Carlo engine are not used to build the surface —
Black-Scholes and its inversion are. They earn their place as the independent
check on that closed form: if all three disagree, the analytic formula is where
to look first.
"""

from __future__ import annotations

import numpy as np
import pytest

from options.binomial_tree import binomial_price
from options.black_scholes import price
from options.monte_carlo import mc_price


S, K, T, R, SIGMA = 100.0, 105.0, 1.0, 0.04, 0.25


@pytest.mark.parametrize("option", ["call", "put"])
def test_binomial_converges_to_black_scholes(option):
    """The CRR error must actually decay with the step count.

    Note what is *not* asserted: that the error is monotone. CRR error
    oscillates in sign as the strike moves between lattice nodes -- here it runs
    +3.2e-2, -2.2e-2, +1.0e-2, -4.3e-3, +1.2e-3 for n = 50, 100, 200, 400, 800.
    Any test pinning a single step count to a tight tolerance is testing where
    the strike happens to fall in the lattice, not convergence. The envelope is
    the real property, so that is what is checked.
    """
    exact = price(S, K, T, R, SIGMA, option)
    coarse = abs(binomial_price(S, K, T, R, SIGMA, option, n_steps=50)["price"] - exact)
    fine = abs(binomial_price(S, K, T, R, SIGMA, option, n_steps=800)["price"] - exact)

    assert fine < 5e-3
    assert fine < coarse / 10


def test_american_call_on_a_non_dividend_payer_has_no_early_exercise_premium():
    """Textbook result: never optimal to exercise such a call early."""
    res = binomial_price(S, K, T, R, SIGMA, "call", style="american", n_steps=400)
    assert res["early_exercise"] == pytest.approx(0.0, abs=1e-8)


def test_american_put_is_worth_more_than_the_european_put():
    res = binomial_price(S, K, T, R, SIGMA, "put", style="american", n_steps=400)
    assert res["early_exercise"] > 0.0
    assert res["price"] > price(S, K, T, R, SIGMA, "put")


def test_binomial_rejects_bad_arguments():
    with pytest.raises(ValueError, match="option"):
        binomial_price(S, K, T, R, SIGMA, "straddle")
    with pytest.raises(ValueError, match="style"):
        binomial_price(S, K, T, R, SIGMA, "call", style="bermudan")


@pytest.mark.parametrize("option_type", ["european_call", "european_put"])
def test_monte_carlo_brackets_the_closed_form(option_type):
    """The analytic price must sit inside the MC 95% confidence interval."""
    exact = price(S, K, T, R, SIGMA, option_type.split("_")[1])
    res = mc_price(S, K, T, R, SIGMA, option_type, n_sims=60_000, n_steps=1, seed=7)
    assert res["conf_95_lo"] <= exact <= res["conf_95_hi"]


def test_antithetic_variates_reduce_the_standard_error():
    common = dict(n_sims=40_000, n_steps=1, seed=3)
    plain = mc_price(S, K, T, R, SIGMA, "european_call", antithetic=False, **common)
    anti = mc_price(S, K, T, R, SIGMA, "european_call", antithetic=True, **common)
    assert anti["std_error"] < plain["std_error"]


def test_a_knocked_out_barrier_is_worth_less_than_the_vanilla():
    vanilla = mc_price(S, K, T, R, SIGMA, "european_call",
                       n_sims=40_000, n_steps=50, seed=11)["price"]
    knockout = mc_price(S, K, T, R, SIGMA, "barrier_call", barrier=80.0,
                        n_sims=40_000, n_steps=50, seed=11)["price"]
    assert 0.0 < knockout < vanilla


def test_monte_carlo_is_reproducible():
    a = mc_price(S, K, T, R, SIGMA, "european_call", n_sims=5_000, seed=1)["price"]
    b = mc_price(S, K, T, R, SIGMA, "european_call", n_sims=5_000, seed=1)["price"]
    assert a == b
