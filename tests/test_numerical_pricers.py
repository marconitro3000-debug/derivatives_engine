"""The numerical pricers, anchored to the closed form they must converge to.

The binomial tree and the Monte Carlo engine are not used to build the surface —
Black-Scholes and its inversion are. They earn their place as the independent
check on that closed form: if all three disagree, the analytic formula is where
to look first.
"""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.american import binomial_price
from volsurface.blackscholes import price
from volsurface.montecarlo import mc_price


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


@pytest.mark.parametrize("option", ["call", "put"])
def test_binomial_matches_black_scholes_with_a_dividend_yield(option):
    """The yield must enter the drift, not be folded into the spot.

    Both routes give the same European price, which is why the shortcut is
    tempting; only the drift version puts the American exercise boundary in the
    right place, so this test guards the European leg and
    `test_dividend_yield_changes_the_american_premium` guards the rest.
    """
    q = 0.03
    exact = price(S, K, T, R, SIGMA, option, q)
    tree = binomial_price(S, K, T, R, SIGMA, option, n_steps=800, q=q)["price"]
    assert tree == pytest.approx(exact, abs=5e-3)


def test_dividend_yield_changes_the_american_premium():
    """A dividend yield makes early exercise of a call possible at all.

    With q = 0 the American call premium is exactly zero; raising q above the
    rate makes holding the call costly and the premium turns positive. Folding
    the yield into the spot would leave it at zero and hide the effect.
    """
    no_div = binomial_price(S, K, T, R, SIGMA, "call", style="american",
                            n_steps=400, q=0.0)["early_exercise"]
    with_div = binomial_price(S, K, T, R, SIGMA, "call", style="american",
                              n_steps=400, q=0.10)["early_exercise"]
    assert no_div == pytest.approx(0.0, abs=1e-8)
    assert with_div > 1e-3


def test_de_americanised_iv_recovers_the_lattice_vol():
    """Price American on a fine lattice, invert with the coarse one, get the vol back.

    The point of the construction: the premium is a difference of two prices off
    the *same* coarse lattice, so its discretisation error cancels and 150 steps
    suffice even though pricing to this accuracy would need far more.
    """
    from volsurface.american import de_americanised_iv
    from volsurface.impliedvol import implied_vol

    S, K, T, r, q, sigma = 100.0, 85.0, 1.5, 0.045, 0.01, 0.25
    F = S * np.exp((r - q) * T)
    DF = float(np.exp(-r * T))
    mkt = binomial_price(S, K, T, r, sigma, "put", style="american",
                         n_steps=600, q=q)["price"]

    naive = implied_vol(F, K, T, 0.0, mkt / DF, "put")
    fixed, euro_price = de_americanised_iv(S, K, T, r, q, F, DF, mkt, "put",
                                           naive, n_steps=150)

    assert abs(fixed - sigma) < 2e-3
    assert abs(fixed - sigma) < abs(naive - sigma) / 3
    assert euro_price < mkt                    # the premium is stripped out


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
