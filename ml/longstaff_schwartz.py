"""
ml/longstaff_schwartz.py
Longstaff-Schwartz (2001) least-squares Monte Carlo for American/Bermudan options.

Reference: Longstaff & Schwartz, "Valuing American Options by Simulation:
A Simple Least-Squares Approach", RFS 2001.

Algorithm
---------
1. Simulate N GBM paths: shape (N, n_steps+1)
2. For each ITM path at exercise date t, regress discounted future cash-flows
   onto basis functions of S_t  (Laguerre polynomials or power basis)
3. If immediate payoff > estimated continuation → exercise
4. Price = mean of discounted optimal cash-flows

Basis functions (degree 3 by default):
    Laguerre : L₀(x) = 1, L₁(x) = 1−x, L₂(x) = 1−2x+x²/2
    Power    : 1, x, x², x³  (normalized by K)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class LSMResult:
    price: float
    std_error: float
    conf_95_lo: float
    conf_95_hi: float
    n_sims: int
    n_steps: int
    exercise_boundary: np.ndarray   # shape (n_steps,): S* at each step (NaN if not computed)
    early_exercise_premium: float   # vs European Black-Scholes

    def __str__(self) -> str:
        return (
            f"LSM American Option\n"
            f"  Price        : {self.price:>10.4f}\n"
            f"  Std error    : {self.std_error:>10.4f}\n"
            f"  95% CI       : [{self.conf_95_lo:.4f}, {self.conf_95_hi:.4f}]\n"
            f"  EE premium   : {self.early_exercise_premium:>10.4f}\n"
            f"  Sims × steps : {self.n_sims} × {self.n_steps}"
        )


def _laguerre_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """Laguerre polynomial basis on x (normalized, x >= 0)."""
    cols = [np.exp(-x / 2)]
    if degree >= 1:
        cols.append(np.exp(-x / 2) * (1 - x))
    if degree >= 2:
        cols.append(np.exp(-x / 2) * (1 - 2*x + 0.5*x**2))
    if degree >= 3:
        cols.append(np.exp(-x / 2) * (1 - 3*x + 1.5*x**2 - x**3/6))
    return np.column_stack(cols)


def _power_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """Power polynomial basis: [1, x, x², ...]"""
    return np.column_stack([x**i for i in range(degree + 1)])


def _bs_price(S, K, T, r, sigma, option_type="put") -> float:
    """European BS price for EE premium calculation."""
    from scipy.stats import norm as _norm
    if T <= 0:
        return max(K - S, 0.0) if option_type == "put" else max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "put":
        return K * np.exp(-r * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1)
    return S * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d2)


def price_american_lsm(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "put",
    n_sims: int = 50_000,
    n_steps: int = 100,
    degree: int = 3,
    basis: str = "laguerre",
    antithetic: bool = True,
    seed: int = 42,
) -> LSMResult:
    """
    American option price via Longstaff-Schwartz LSM.

    Parameters
    ----------
    S, K, T, r, sigma : standard Black-Scholes parameters
    option_type : "put" or "call"
    n_sims      : number of Monte Carlo paths
    n_steps     : number of time steps
    degree      : polynomial degree for regression basis
    basis       : "laguerre" or "power"
    antithetic  : use antithetic variates for variance reduction

    Returns
    -------
    LSMResult dataclass
    """
    if n_sims % 2 != 0 and antithetic:
        n_sims += 1

    dt       = T / n_steps
    discount = np.exp(-r * dt)
    rng      = np.random.default_rng(seed)

    # ── Simulate paths ────────────────────────────────────────────────────────
    half  = n_sims // 2 if antithetic else n_sims
    Z     = rng.standard_normal((half, n_steps))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)

    log_S  = np.log(S) + np.cumsum(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1
    )
    paths  = np.exp(log_S)                        # shape (n_sims, n_steps)
    paths  = np.hstack([np.full((n_sims, 1), S), paths])  # include t=0

    # ── Payoff function ───────────────────────────────────────────────────────
    if option_type == "put":
        payoff_fn = lambda x: np.maximum(K - x, 0.0)
    elif option_type == "call":
        payoff_fn = lambda x: np.maximum(x - K, 0.0)
    else:
        raise ValueError(f"option_type must be 'put' or 'call', got {option_type!r}")

    # ── Basis function ────────────────────────────────────────────────────────
    if basis == "laguerre":
        basis_fn = lambda x: _laguerre_basis(x / K, degree)
    else:
        basis_fn = lambda x: _power_basis(x / K, degree)

    # ── Backward induction ────────────────────────────────────────────────────
    # cash_flow[i] = optimal cash flow for path i (undiscounted)
    # exercise_time[i] = when path i exercises (in years)
    cf       = payoff_fn(paths[:, -1])
    ex_time  = np.full(n_sims, T)

    for step in range(n_steps - 1, 0, -1):
        t_step = step * dt
        S_t    = paths[:, step]
        itm    = payoff_fn(S_t) > 0

        if itm.sum() < degree + 2:
            continue

        # Discount existing cash flows to current time
        itm_idx   = np.where(itm)[0]
        disc_cf   = cf[itm_idx] * np.exp(-r * (ex_time[itm_idx] - t_step))

        # Regression: continuation value estimate
        X         = basis_fn(S_t[itm_idx])
        try:
            coeffs, *_ = np.linalg.lstsq(X, disc_cf, rcond=None)
            cont       = X @ coeffs
        except np.linalg.LinAlgError:
            continue

        # Exercise decision
        imm_payoff = payoff_fn(S_t[itm_idx])
        exercise   = imm_payoff > cont

        ex_idx = itm_idx[exercise]
        cf[ex_idx]      = imm_payoff[exercise]
        ex_time[ex_idx] = t_step

    # ── Price ─────────────────────────────────────────────────────────────────
    discounted = cf * np.exp(-r * ex_time)
    price      = float(np.mean(discounted))
    se         = float(np.std(discounted) / np.sqrt(n_sims))
    european   = _bs_price(S, K, T, r, sigma, option_type)

    # ── Exercise boundary (rough estimate) ────────────────────────────────────
    boundary = np.full(n_steps, np.nan)
    for step in range(1, n_steps):
        ex_mask = (ex_time == step * dt)
        if ex_mask.sum() > 5:
            boundary[step] = np.median(paths[ex_mask, step])

    return LSMResult(
        price=price,
        std_error=se,
        conf_95_lo=price - 1.96 * se,
        conf_95_hi=price + 1.96 * se,
        n_sims=n_sims,
        n_steps=n_steps,
        exercise_boundary=boundary,
        early_exercise_premium=max(price - european, 0.0),
    )


def price_bermudan_lsm(
    S: float, K: float, T: float, r: float, sigma: float,
    exercise_dates: list[float],
    option_type: str = "put",
    n_sims: int = 50_000,
    n_steps: int = 200,
    degree: int = 3,
    seed: int = 42,
) -> LSMResult:
    """
    Bermudan option: exercise only allowed at specified dates.

    Parameters
    ----------
    exercise_dates : sorted list of exercise times in years (e.g. [0.25, 0.5, 0.75, 1.0])
    """
    exercise_dates = sorted(exercise_dates)
    T = exercise_dates[-1]
    rng = np.random.default_rng(seed)

    dt = T / n_steps
    Z  = rng.standard_normal((n_sims, n_steps))
    log_S = np.log(S) + np.cumsum(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1
    )
    paths = np.exp(log_S)
    paths = np.hstack([np.full((n_sims, 1), S), paths])

    payoff_fn = (lambda x: np.maximum(K - x, 0.0) if option_type == "put"
                 else lambda x: np.maximum(x - K, 0.0))

    # Map exercise dates to step indices
    ex_steps = sorted(set(max(1, round(d / dt)) for d in exercise_dates))

    cf      = payoff_fn(paths[:, -1])
    ex_time = np.full(n_sims, T)

    for step in reversed(ex_steps[:-1]):   # last date = maturity (no regression needed)
        t_step = step * dt
        S_t    = paths[:, step]
        itm    = payoff_fn(S_t) > 0

        if itm.sum() < degree + 2:
            continue

        itm_idx  = np.where(itm)[0]
        disc_cf  = cf[itm_idx] * np.exp(-r * (ex_time[itm_idx] - t_step))
        X        = _laguerre_basis(S_t[itm_idx] / K, degree)
        coeffs, *_ = np.linalg.lstsq(X, disc_cf, rcond=None)
        cont     = X @ coeffs

        imm     = payoff_fn(S_t[itm_idx])
        ex_here = imm > cont

        idx = itm_idx[ex_here]
        cf[idx]      = imm[ex_here]
        ex_time[idx] = t_step

    discounted = cf * np.exp(-r * ex_time)
    price      = float(np.mean(discounted))
    se         = float(np.std(discounted) / np.sqrt(n_sims))
    european   = _bs_price(S, K, T, r, sigma, option_type)

    return LSMResult(
        price=price,
        std_error=se,
        conf_95_lo=price - 1.96 * se,
        conf_95_hi=price + 1.96 * se,
        n_sims=n_sims,
        n_steps=n_steps,
        exercise_boundary=np.full(n_steps, np.nan),
        early_exercise_premium=max(price - european, 0.0),
    )
