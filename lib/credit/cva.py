"""
credit/cva.py
Credit Valuation Adjustment (CVA) and Debit Valuation Adjustment (DVA).

CVA = (1−R) × Σᵢ DF(tᵢ) × EE(tᵢ) × [Q(tᵢ₋₁) − Q(tᵢ)]

where:
  EE(tᵢ)  = Expected Positive Exposure at tᵢ  = E[max(V(tᵢ), 0)]
  Q(tᵢ)   = counterparty survival probability
  DF(tᵢ)  = risk-free discount factor
  R        = counterparty recovery rate

DVA uses the bank's own credit curve and Expected Negative Exposure:
DVA = (1−R_own) × Σᵢ DF(tᵢ) × ENE(tᵢ) × [Q_own(tᵢ₋₁) − Q_own(tᵢ)]

BCVA (bilateral CVA) = CVA − DVA
"""

from __future__ import annotations

import numpy as np


def cva(hazard_curve, discount_curve,
        exposure_times, expected_exposures,
        recovery: float = None) -> dict:
    """
    CVA from a pre-computed expected exposure (EE) profile.

    Parameters
    ----------
    hazard_curve        : HazardCurve (counterparty)
    discount_curve      : DiscountCurve (risk-free)
    exposure_times      : array-like of time points (years)
    expected_exposures  : array-like of EE(tᵢ) ≥ 0

    Returns
    -------
    dict:
      'cva'           — total CVA (same units as EE)
      'contributions' — per-period CVA contributions
      'exposure_times'
    """
    R     = recovery if recovery is not None else hazard_curve.recovery
    times = np.asarray(exposure_times, dtype=float)
    EE    = np.asarray(expected_exposures, dtype=float)

    Q_prev        = 1.0
    contributions = np.zeros(len(times))

    for i, (t, ee) in enumerate(zip(times, EE)):
        Q_t  = hazard_curve.survival_prob(t)
        DF_t = discount_curve.discount_factor(t)
        contributions[i] = (1.0 - R) * DF_t * ee * (Q_prev - Q_t)
        Q_prev = Q_t

    total = float(np.sum(contributions))
    return {
        "cva":            total,
        "contributions":  contributions,
        "exposure_times": times,
    }


def cva_option(S: float, K: float, T: float, r_rate: float, sigma: float,
               hazard_curve, discount_curve,
               option_type: str = "call",
               recovery: float = None,
               n_steps: int = 252) -> dict:
    """
    CVA for a vanilla European option using the re-pricing approach.

    The expected exposure at time t is approximated by the Black-Scholes
    option value with remaining life (T − t):
        EE(t) ≈ BSM(S, K, T−t, r, σ, option_type)

    This is the "counterparty has no collateral" scenario.

    Returns
    -------
    dict with keys: 'cva', 'vanilla_price', 'cva_adjusted_price',
                    'cva_pct_of_vanilla', 'exposure_profile', 'contributions',
                    'exposure_times'
    """
    from options.black_scholes import price as bs_price

    R     = recovery if recovery is not None else hazard_curve.recovery
    dt    = T / n_steps
    # exclude t=T so that remaining life (T-t) is always > 0
    times = np.linspace(dt, T - dt, n_steps - 1)

    EE = np.array([
        max(bs_price(S, K, T - t, r_rate, sigma, option=option_type), 0.0)
        for t in times
    ])

    result  = cva(hazard_curve, discount_curve, times, EE, R)
    vanilla = bs_price(S, K, T, r_rate, sigma, option=option_type)

    result["vanilla_price"]       = float(vanilla)
    result["cva_adjusted_price"]  = float(vanilla - result["cva"])
    result["cva_pct_of_vanilla"]  = (float(result["cva"] / vanilla * 100)
                                     if vanilla > 1e-10 else np.nan)
    result["exposure_profile"]    = EE
    return result


def dva(hazard_curve_own, discount_curve,
        exposure_times, expected_negative_exposures,
        recovery_own: float = None) -> dict:
    """
    DVA: benefit from own credit risk using Expected Negative Exposure (ENE).

    ENE(tᵢ) = E[max(−V(tᵢ), 0)]  — how much the bank owes the counterparty.
    """
    R     = recovery_own if recovery_own is not None else hazard_curve_own.recovery
    times = np.asarray(exposure_times, dtype=float)
    ENE   = np.asarray(expected_negative_exposures, dtype=float)

    Q_prev        = 1.0
    contributions = np.zeros(len(times))

    for i, (t, ene) in enumerate(zip(times, ENE)):
        Q_t  = hazard_curve_own.survival_prob(t)
        DF_t = discount_curve.discount_factor(t)
        contributions[i] = (1.0 - R) * DF_t * ene * (Q_prev - Q_t)
        Q_prev = Q_t

    total = float(np.sum(contributions))
    return {
        "dva":            total,
        "contributions":  contributions,
        "exposure_times": times,
    }


def bilateral_cva(cva_result: dict, dva_result: dict) -> dict:
    """
    BCVA = CVA − DVA
    Positive BCVA means the bank should charge the counterparty a credit premium.
    """
    return {
        "cva":  cva_result["cva"],
        "dva":  dva_result["dva"],
        "bcva": cva_result["cva"] - dva_result["dva"],
    }
