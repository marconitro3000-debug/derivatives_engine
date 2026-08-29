"""
structured/autocall.py
Autocallable structured note pricing via Monte Carlo.

Payoff structure (typical European barrier autocall):
─────────────────────────────────────────────────────
At each observation date tᵢ (i=1,…,n):
  • If S(tᵢ) ≥ S₀ × autocall_level:
      → Bond redeems at par + coupon_rate × tᵢ × notional  (early redemption)
      → Product terminates

At maturity T (if never called):
  • If S(T) ≥ S₀ × ki_barrier (knock-in NOT triggered):
      → Pay notional (capital protected, no coupon)
  • If S(T) < S₀ × ki_barrier (knock-in triggered):
      → Pay notional × S(T)/S₀  (1:1 equity downside, full capital loss)

Knock-in observation can be:
  • "european": only at final maturity T
  • "continuous": daily monitoring (discrete approximation)

Optional memory coupon: if coupon_memory=True, unpaid coupons accumulate and
are all paid at the first autocall date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class AutocallResult:
    price:          float          # PV as fraction of notional
    price_notional: float          # PV in currency
    prob_call:      np.ndarray     # probability of call at each observation
    prob_no_call:   float          # probability of reaching maturity uncalled
    prob_ki:        float          # P(KI triggered | not called)
    prob_ki_loss:   float          # P(KI loss at maturity, full product)
    expected_life:  float          # expected life (years)
    call_date_dist: pd.Series      # distribution of call dates
    payoff_hist:    np.ndarray     # histogram of payoffs
    notional:       float


def price_autocall(S: float, r: float, sigma: float,
                   T: float = 1.0,
                   obs_dates: list[float] = None,
                   autocall_level: float = 1.0,
                   ki_barrier: float = 0.70,
                   coupon_rate: float = 0.08,
                   coupon_memory: bool = False,
                   ki_type: str = "european",
                   notional: float = 1_000.0,
                   n_sims: int = 50_000,
                   n_steps_per_year: int = 252,
                   seed: int = None) -> AutocallResult:
    """
    Price an autocallable note by Monte Carlo.

    Parameters
    ----------
    S             : initial spot
    r             : risk-free rate (continuous)
    sigma         : equity volatility (annualized)
    T             : maturity in years
    obs_dates     : autocall observation dates (default: quarterly in [0, T])
    autocall_level: autocall trigger as fraction of S₀ (default 1.0 = par)
    ki_barrier    : knock-in barrier as fraction of S₀ (default 0.70)
    coupon_rate   : annual coupon rate (fraction)
    coupon_memory : if True, accumulate missed coupons
    ki_type       : 'european' (at maturity only) or 'continuous' (daily)
    notional      : face value
    n_sims        : number of MC paths
    n_steps_per_year: discrete time steps per year

    Returns
    -------
    AutocallResult
    """
    rng = np.random.default_rng(seed)

    if obs_dates is None:
        # Quarterly by default
        n_obs   = max(1, int(T * 4))
        obs_dates = [round((i + 1) * T / n_obs, 6) for i in range(n_obs)]

    obs_dates = sorted(obs_dates)
    T_final   = float(max(obs_dates[-1], T))

    # Build time grid
    n_steps = max(int(T_final * n_steps_per_year), len(obs_dates) * 4)
    dt      = T_final / n_steps
    t_grid  = np.linspace(dt, T_final, n_steps)

    # Map observation dates to grid indices
    obs_idx = [int(round(t / dt)) - 1 for t in obs_dates]
    obs_idx = [min(max(i, 0), n_steps - 1) for i in obs_idx]

    # GBM simulation
    Z     = rng.standard_normal((n_sims, n_steps))
    drift = (r - 0.5 * sigma ** 2) * dt
    diff  = sigma * np.sqrt(dt) * Z
    lret  = drift + diff
    paths = S * np.exp(np.cumsum(lret, axis=1))   # shape: (n_sims, n_steps)

    AC_level = S * autocall_level
    KI_level = S * ki_barrier

    payoffs       = np.zeros(n_sims)
    call_date_idx = np.full(n_sims, -1, dtype=int)   # -1 = no call
    ki_hit        = np.zeros(n_sims, dtype=bool)
    alive         = np.ones(n_sims, dtype=bool)

    # Check continuous KI during path
    if ki_type == "continuous":
        # path min
        path_min = paths.min(axis=1)   # shape: (n_sims,)

    for i_obs, t_idx in enumerate(obs_idx):
        t_obs = obs_dates[i_obs]
        s_obs = paths[:, t_idx]

        called  = alive & (s_obs >= AC_level)
        coupon  = coupon_rate * t_obs
        if coupon_memory:
            coupon = coupon_rate * t_obs  # full accumulated coupon at this date
        payoffs[called]       = notional * (1.0 + coupon)
        call_date_idx[called] = i_obs
        alive[called]         = False

    # At maturity: surviving paths
    if alive.any():
        s_T = paths[alive, -1]

        # Check knock-in
        if ki_type == "european":
            ki = s_T < KI_level
        else:
            ki = path_min[alive] < KI_level

        ki_hit[alive] = ki

        # KI triggered: equity downside (no coupon)
        ki_mask = alive.copy()
        ki_mask[alive] = ki
        payoffs[ki_mask] = notional * paths[ki_mask, -1] / S

        # KI not triggered: capital returned (no coupon for missed calls)
        no_ki_mask = alive.copy()
        no_ki_mask[alive] = ~ki
        payoffs[no_ki_mask] = notional

    # Discount each payoff to present value
    # Time of payment: obs_dates[i] for called, T for uncalled
    pay_times = np.full(n_sims, T_final)
    for i_obs, t_obs in enumerate(obs_dates):
        mask = call_date_idx == i_obs
        pay_times[mask] = t_obs

    disc_payoffs = payoffs * np.exp(-r * pay_times)
    price_frac   = disc_payoffs.mean() / notional

    # Diagnostics
    prob_call   = np.array([(call_date_idx == i).mean() for i in range(len(obs_dates))])
    prob_no_call = alive.mean()
    prob_ki     = ki_hit[alive].mean() if alive.any() else 0.0
    prob_ki_loss = ki_hit.mean()

    exp_life = np.where(call_date_idx >= 0,
                        np.array([obs_dates[i] if i >= 0 else T_final
                                  for i in call_date_idx]),
                        T_final).mean()

    call_dates_series = pd.Series(prob_call,
                                  index=[f"t={t:.2f}" for t in obs_dates],
                                  name="P(call)")

    return AutocallResult(
        price=float(price_frac),
        price_notional=float(price_frac * notional),
        prob_call=prob_call,
        prob_no_call=float(prob_no_call),
        prob_ki=float(prob_ki),
        prob_ki_loss=float(prob_ki_loss),
        expected_life=float(exp_life),
        call_date_dist=call_dates_series,
        payoff_hist=disc_payoffs,
        notional=float(notional),
    )


def autocall_greeks(S: float, r: float, sigma: float,
                    T: float = 1.0, **kwargs) -> dict:
    """
    Finite-difference greeks for the autocall.

    Returns delta, gamma, vega, rho.
    """
    dS    = S * 0.01
    dsig  = 0.01
    dr    = 0.0001

    kwargs.setdefault("n_sims", 20_000)
    kwargs.setdefault("seed",   0)

    base = price_autocall(S,     r,      sigma,      T, **kwargs).price
    up_S = price_autocall(S+dS,  r,      sigma,      T, **kwargs).price
    dn_S = price_autocall(S-dS,  r,      sigma,      T, **kwargs).price
    up_v = price_autocall(S,     r,      sigma+dsig, T, **kwargs).price
    up_r = price_autocall(S,     r+dr,   sigma,      T, **kwargs).price

    delta = (up_S - dn_S) / (2 * dS)
    gamma = (up_S - 2 * base + dn_S) / (dS ** 2)
    vega  = (up_v - base) / dsig
    rho_  = (up_r - base) / dr

    return {
        "price": float(base),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega":  float(vega),
        "rho":   float(rho_),
    }
