"""
core/models/merton.py
Merton (1976) jump-diffusion model — closed-form as a Poisson-weighted sum of
Black-Scholes prices, one term per possible jump count.

Dynamics under the risk-neutral measure:
    dS/S = (r - lambda*k) dt + sigma dW + dJ
where jumps arrive as a Poisson process with intensity `lambda`, and each
jump's log-size is Normal(mu_j, delta_j^2); k = E[e^jump] - 1 is the mean
relative jump size (subtracted from the drift to keep S risk-neutral).

Reference: Merton, R. (1976) "Option pricing when underlying stock returns
are discontinuous", Journal of Financial Economics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

from options.black_scholes import price as bs_price


@dataclass
class MertonParams:
    jump_intensity: float   # lambda: expected number of jumps per year
    jump_mean: float        # mu_j: mean log jump size
    jump_vol: float         # delta_j: std dev of log jump size


def price(S: float, K: float, T: float, r: float, sigma: float,
          p: MertonParams, option: str = "call", q: float = 0.0,
          max_terms: int = 50) -> float:
    """European option price under Merton jump-diffusion.

    Sums the Poisson-weighted Black-Scholes price conditional on exactly n
    jumps occurring before expiry, for n = 0, 1, 2, ... — each term is an
    ordinary Black-Scholes price with a jump-adjusted vol and drift.
    """
    k = np.exp(p.jump_mean + 0.5 * p.jump_vol**2) - 1.0
    lam_prime = p.jump_intensity * (1.0 + k)

    total = 0.0
    for n in range(max_terms):
        poisson_weight = np.exp(-lam_prime * T) * (lam_prime * T) ** n / factorial(n)
        if n > 5 and poisson_weight < 1e-14:
            break
        sigma_n = np.sqrt(sigma**2 + n * p.jump_vol**2 / T)
        r_n = r - p.jump_intensity * k + n * np.log(1.0 + k) / T
        total += poisson_weight * bs_price(S, K, T, r_n, sigma_n, option, q)
    return float(total)


def default_params() -> MertonParams:
    """Reasonable seed: modest tail risk (2 jumps/year, -5% mean size, 10% jump vol)."""
    return MertonParams(jump_intensity=2.0, jump_mean=-0.05, jump_vol=0.10)
