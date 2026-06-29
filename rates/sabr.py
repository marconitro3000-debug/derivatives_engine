"""
rates/sabr.py
SABR stochastic volatility model (Hagan, Kumar, Lesniewski, Woodward 2002).

Dynamics:
    dF = σ · F^β · dW₁
    dσ = ν · σ · dW₂
    E[dW₁ dW₂] = ρ dt

Parameters:
    α   (alpha) : initial vol level  (> 0)
    β   (beta)  : CEV exponent       ∈ [0, 1]; β=1 → log-normal, β=0 → normal
    ρ   (rho)   : correlation         ∈ (−1, 1)
    ν   (nu)    : vol of vol          (≥ 0)

Hagan et al. (2002) approximation for implied Black vol σ_B(K, F, T):

For K ≠ F:
    z       = (ν/α) × (FK)^{(1−β)/2} × ln(F/K)
    χ(z)    = ln[(√(1−2ρz+z²) + z − ρ) / (1−ρ)]
    A       = α / [(FK)^{(1−β)/2} × {1 + (1−β)²/24 × ln²(F/K) + (1−β)⁴/1920 × ln⁴(F/K)}]
    σ_B     = A × (z/χ(z)) × {1 + [(1−β)²α²/(24(FK)^{1−β}) + ραβν/(4(FK)^{(1−β)/2}) + (2−3ρ²)ν²/24] × T}

ATM (K → F):
    σ_ATM = (α / F^{1−β}) × {1 + [(1−β)²α²/(24F^{2−2β}) + ραβν/(4F^{1−β}) + (2−3ρ²)ν²/24] × T}
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class SABRParams:
    alpha: float    # initial vol
    beta:  float    # CEV exponent ∈ [0,1]
    rho:   float    # correlation ∈ (-1,1)
    nu:    float    # vol of vol

    def __post_init__(self):
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if self.nu < 0:
            raise ValueError("nu must be >= 0")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")


# ── core formula ──────────────────────────────────────────────────────────────

def implied_vol_sabr(F: float, K: float, T: float,
                      params: SABRParams) -> float:
    """
    Hagan et al. (2002) Black-vol approximation for SABR.

    Parameters
    ----------
    F : forward rate
    K : strike
    T : option expiry (years)

    Returns
    -------
    Implied Black vol σ_B (annualized)
    """
    alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu

    # ATM special case (K very close to F)
    if abs(F - K) < 1e-10 * F:
        FK_mid = F ** (1.0 - beta)
        term1  = (1.0 - beta) ** 2 * alpha ** 2 / (24.0 * FK_mid ** 2)
        term2  = rho * beta * nu * alpha / (4.0 * FK_mid)
        term3  = (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0
        return float(alpha / FK_mid * (1.0 + (term1 + term2 + term3) * T))

    log_FK = np.log(F / K)
    FK_avg = np.sqrt(F * K)
    FK_mid = FK_avg ** (1.0 - beta)

    # z and chi
    z   = (nu / alpha) * FK_mid * log_FK
    rho2 = rho ** 2
    chi = np.log((np.sqrt(1.0 - 2.0 * rho * z + z ** 2) + z - rho) / (1.0 - rho))

    if abs(chi) < 1e-12:
        z_chi = 1.0
    else:
        z_chi = z / chi

    # Series expansion in log(F/K)
    log2 = log_FK ** 2
    log4 = log_FK ** 4
    denom_expansion = 1.0 + (1.0 - beta) ** 2 / 24.0 * log2 \
                         + (1.0 - beta) ** 4 / 1920.0 * log4

    A = alpha / (FK_mid * denom_expansion)

    # Time expansion
    term1 = (1.0 - beta) ** 2 * alpha ** 2 / (24.0 * FK_avg ** (2.0 * (1.0 - beta)))
    term2 = rho * beta * nu * alpha / (4.0 * FK_mid)
    term3 = (2.0 - 3.0 * rho2) * nu ** 2 / 24.0
    time_factor = 1.0 + (term1 + term2 + term3) * T

    return float(A * z_chi * time_factor)


def implied_vol_grid(F: float, strikes: np.ndarray, T: float,
                      params: SABRParams) -> np.ndarray:
    """SABR vol for an array of strikes."""
    return np.array([implied_vol_sabr(F, K, T, params) for K in strikes])


# ── calibration ───────────────────────────────────────────────────────────────

def calibrate(F: float, T: float,
              strikes: np.ndarray,
              market_vols: np.ndarray,
              beta: float = 0.5,
              initial_guess: tuple[float, float, float] = None) -> SABRParams:
    """
    Calibrate α, ρ, ν to market Black vols, holding β fixed.

    Minimises sum of squared errors between SABR and market vols.

    Parameters
    ----------
    F            : forward rate
    T            : option expiry
    strikes      : market strike array
    market_vols  : market Black vols at each strike
    beta         : fixed CEV exponent (0.5 is a common choice)
    initial_guess: (alpha, rho, nu) starting point (optional)

    Returns
    -------
    SABRParams
    """
    K    = np.asarray(strikes, dtype=float)
    mkt  = np.asarray(market_vols, dtype=float)
    atm  = np.interp(F, K, mkt) if F >= K[0] and F <= K[-1] else mkt[len(mkt)//2]

    if initial_guess is None:
        alpha0 = atm * F ** (1.0 - beta)
        rho0   = -0.30
        nu0    = 0.40
    else:
        alpha0, rho0, nu0 = initial_guess

    def objective(x):
        try:
            p = SABRParams(alpha=np.exp(x[0]), beta=beta,
                           rho=np.tanh(x[1]),
                           nu=np.exp(x[2]))
            sabr_vols = implied_vol_grid(F, K, T, p)
            return float(np.sum((sabr_vols - mkt) ** 2))
        except Exception:
            return 1e8

    x0 = np.array([np.log(alpha0), np.arctanh(rho0), np.log(nu0)])
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 10_000, "xatol": 1e-8, "fatol": 1e-10})

    best = res.x
    return SABRParams(
        alpha=float(np.exp(best[0])),
        beta=float(beta),
        rho=float(np.tanh(best[1])),
        nu=float(np.exp(best[2])),
    )


def calibrate_beta(F: float, T: float,
                    strikes: np.ndarray,
                    market_vols: np.ndarray) -> SABRParams:
    """
    Calibrate all four SABR parameters (α, β, ρ, ν) jointly.
    More numerically challenging; prefer fixing β when possible.
    """
    K   = np.asarray(strikes, dtype=float)
    mkt = np.asarray(market_vols, dtype=float)

    def objective(x):
        try:
            p = SABRParams(alpha=np.exp(x[0]),
                           beta=1.0 / (1.0 + np.exp(-x[1])),
                           rho=np.tanh(x[2]),
                           nu=np.exp(x[3]))
            sabr_vols = implied_vol_grid(F, K, T, p)
            return float(np.sum((sabr_vols - mkt) ** 2))
        except Exception:
            return 1e8

    atm = np.interp(F, K, mkt) if K[0] <= F <= K[-1] else mkt[len(mkt)//2]
    x0  = np.array([np.log(atm * F ** 0.5), 0.0, -0.3, np.log(0.4)])
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 20_000, "xatol": 1e-8, "fatol": 1e-10})
    x   = res.x
    return SABRParams(
        alpha=float(np.exp(x[0])),
        beta=float(1.0 / (1.0 + np.exp(-x[1]))),
        rho=float(np.tanh(x[2])),
        nu=float(np.exp(x[3])),
    )


# ── ATM vol and greeks ────────────────────────────────────────────────────────

def atm_vol(F: float, T: float, params: SABRParams) -> float:
    """ATM SABR vol (closed form, no approximation needed for K=F)."""
    return implied_vol_sabr(F, F, T, params)


def vol_sensitivity(F: float, T: float, params: SABRParams,
                     d_alpha: float = 1e-4) -> dict:
    """
    Numerical sensitivities of ATM vol to SABR parameters.
    Useful for hedging books of caplets/swaptions.
    """
    base   = atm_vol(F, T, params)
    d_rho  = 1e-4
    d_nu   = 1e-4

    p_a  = SABRParams(params.alpha + d_alpha, params.beta, params.rho, params.nu)
    p_r  = SABRParams(params.alpha, params.beta, params.rho + d_rho, params.nu)
    p_n  = SABRParams(params.alpha, params.beta, params.rho, params.nu + d_nu)

    return {
        "atm_vol":    base,
        "d_vol/d_alpha": (atm_vol(F, T, p_a) - base) / d_alpha,
        "d_vol/d_rho":   (atm_vol(F, T, p_r) - base) / d_rho,
        "d_vol/d_nu":    (atm_vol(F, T, p_n) - base) / d_nu,
    }
