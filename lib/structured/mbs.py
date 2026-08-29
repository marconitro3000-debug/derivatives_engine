"""
structured/mbs.py
Mortgage-Backed Security (MBS) pricing using the PSA prepayment model.

PSA (Public Securities Association) Prepayment Model:
─────────────────────────────────────────────────────
PSA speed expressed as a percentage of the benchmark (100 PSA = 100%).
Benchmark assumes:
  CPR_t = 6% × (t/30)    for t ≤ 30 months
  CPR_t = 6%              for t > 30 months

At PSA speed p (%):
  CPR_t = 6% × (p/100) × min(t/30, 1)
  SMM_t = 1 − (1 − CPR_t)^(1/12)   (Single Monthly Mortality)

Monthly cash flows:
─────────────────────────────────────────────────────
Let:
  B_t = outstanding balance at beginning of month t
  r_m = WAC / 12  (monthly coupon rate; WAC = Weighted Average Coupon)
  N   = WAM       (Weighted Average Maturity, months)

  Scheduled payment:  P̄_t = B_t × r_m(1+r_m)^(N-t+1) / ((1+r_m)^(N-t+1) − 1)
  Interest:           I_t = B_t × r_m
  Sched. principal:   SP_t = P̄_t − I_t
  Prepayment:         PP_t = SMM_t × (B_t − SP_t)
  Total principal:    TP_t = SP_t + PP_t
  Total cash flow:    CF_t = I_t + TP_t
  End balance:        B_{t+1} = B_t − TP_t

WAL (Weighted Average Life):
  WAL = Σ_t (t/12) × TP_t / Face

Price given flat yield y (monthly):
  P = Σ_t CF_t / (1 + y/12)^t / Face × 100

OAS (Option-Adjusted Spread): z s.t. P_market = Σ CF_t / (1 + (r_t + z)/12)^t
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq


# ── PSA prepayment ────────────────────────────────────────────────────────────

def psa_smm(month: int, psa_speed: float = 100.0) -> float:
    """
    Single Monthly Mortality rate for a given month and PSA speed.

    Parameters
    ----------
    month     : loan age in months (1-indexed)
    psa_speed : PSA speed as percentage of benchmark (100 = 6% CPR at seasoning)
    """
    speed  = psa_speed / 100.0
    cpr    = 0.06 * speed * min(month / 30.0, 1.0)
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def psa_schedule(n_months: int, psa_speed: float = 100.0) -> np.ndarray:
    """SMM for each month 1..n_months."""
    return np.array([psa_smm(t, psa_speed) for t in range(1, n_months + 1)])


# ── cash flow engine ──────────────────────────────────────────────────────────

def mbs_cashflows(face: float = 1_000_000.0,
                  wac: float = 0.065,
                  wam: int = 360,
                  psa_speed: float = 100.0,
                  service_fee: float = 0.0025) -> pd.DataFrame:
    """
    Compute monthly MBS cash flows under the PSA prepayment model.

    Parameters
    ----------
    face        : original pool face value ($)
    wac         : weighted average coupon (annual, e.g. 0.065 = 6.5%)
    wam         : weighted average maturity (months, e.g. 360 = 30-year)
    psa_speed   : PSA benchmark speed (100 = standard)
    service_fee : annual servicing fee (subtracted from WAC for pass-through)

    Returns
    -------
    DataFrame with columns:
      month, balance_beg, smm, scheduled_payment, interest, sched_principal,
      prepayment, total_principal, total_cf, net_interest, net_cf, balance_end
    """
    r_m    = wac / 12.0           # monthly WAC rate
    r_pt   = (wac - service_fee) / 12.0   # pass-through rate
    smm_arr= psa_schedule(wam, psa_speed)

    records = []
    B       = float(face)
    N       = wam

    for t in range(1, wam + 1):
        if B < 0.01:
            break
        smm_t = smm_arr[t - 1]
        rem   = N - t + 1                   # remaining months

        # Fully amortising scheduled payment on remaining balance
        if rem > 0 and r_m > 0:
            sched_pmt = B * r_m * (1 + r_m) ** rem / ((1 + r_m) ** rem - 1)
        else:
            sched_pmt = B

        interest   = B * r_m
        sched_prin = sched_pmt - interest
        prepay     = max(B - sched_prin, 0.0) * smm_t
        total_prin = sched_prin + prepay
        total_cf   = interest + total_prin
        net_int    = B * r_pt          # investor receives pass-through rate
        net_cf     = net_int + total_prin

        B_end = max(B - total_prin, 0.0)

        records.append({
            "month":            t,
            "balance_beg":      B,
            "smm":              smm_t,
            "scheduled_payment": sched_pmt,
            "interest":         interest,
            "sched_principal":  sched_prin,
            "prepayment":       prepay,
            "total_principal":  total_prin,
            "total_cf":         total_cf,
            "net_interest":     net_int,
            "net_cf":           net_cf,
            "balance_end":      B_end,
        })
        B = B_end

    return pd.DataFrame(records)


# ── analytics ─────────────────────────────────────────────────────────────────

def weighted_average_life(cf_df: pd.DataFrame) -> float:
    """
    WAL in years.
    WAL = Σ_t (t/12) × total_principal_t / face
    """
    face = cf_df["balance_beg"].iloc[0]
    wal  = (cf_df["month"] / 12.0 * cf_df["total_principal"]).sum() / face
    return float(wal)


def mbs_price(cf_df: pd.DataFrame, yield_: float,
              face: float = None) -> float:
    """
    MBS price as a percentage of face value given flat yield.

    P = 100 × Σ_t net_CF_t / (1 + y/12)^t / Face

    Parameters
    ----------
    yield_ : annual yield (e.g. 0.065)
    face   : original face (if None, uses balance_beg of first row)
    """
    if face is None:
        face = float(cf_df["balance_beg"].iloc[0])
    r_m     = yield_ / 12.0
    months  = cf_df["month"].values
    cfs     = cf_df["net_cf"].values
    disc    = 1.0 / (1.0 + r_m) ** months
    return float(100.0 * (cfs * disc).sum() / face)


def mbs_yield(market_price_pct: float, cf_df: pd.DataFrame,
              face: float = None) -> float:
    """
    Solve for yield given market price (% of face).

    Uses Brent root-finding on mbs_price(y) = market_price_pct.
    """
    f = lambda y: mbs_price(cf_df, y, face) - market_price_pct
    return float(brentq(f, 0.0001, 0.30, xtol=1e-10, maxiter=200))


def oas(market_price_pct: float, cf_df: pd.DataFrame,
        discount_curve, face: float = None) -> float:
    """
    Option-Adjusted Spread (OAS) in bps.

    Find z (bps) such that:
        P_market = 100 × Σ_t CF_t / (1 + (r(t/12) + z/10000)/12)^t / Face

    where r(t) is the zero rate from the discount curve.
    """
    if face is None:
        face = float(cf_df["balance_beg"].iloc[0])

    months = cf_df["month"].values
    cfs    = cf_df["net_cf"].values

    def price_at_oas(z_bps):
        z    = z_bps / 10_000.0
        pv   = 0.0
        for t_m, cf in zip(months, cfs):
            t      = t_m / 12.0
            r_t    = discount_curve.zero_rate(max(t, 0.01))
            r_m_t  = (r_t + z) / 12.0
            disc   = 1.0 / (1.0 + r_m_t) ** t_m
            pv    += cf * disc
        return 100.0 * pv / face

    f = lambda z: price_at_oas(z) - market_price_pct
    return float(brentq(f, -500.0, 2000.0, xtol=1e-6, maxiter=200))


def mbs_summary(cf_df: pd.DataFrame, yield_: float = None,
                market_price_pct: float = 100.0,
                psa_speed: float = 100.0,
                discount_curve=None) -> str:
    """Print a clean MBS summary."""
    face  = float(cf_df["balance_beg"].iloc[0])
    wal   = weighted_average_life(cf_df)
    n_m   = len(cf_df)
    total_int  = cf_df["net_interest"].sum()
    total_prin = cf_df["total_principal"].sum()
    total_prep = cf_df["prepayment"].sum()

    lines = [
        "MBS Summary",
        f"  Face value      : ${face:,.0f}",
        f"  Remaining months: {n_m}",
        f"  PSA speed       : {psa_speed}%",
        f"  WAL             : {wal:.2f} years",
        f"  Total interest  : ${total_int:,.0f}",
        f"  Total principal : ${total_prin:,.0f}",
        f"  Total prepayment: ${total_prep:,.0f}  ({total_prep/face*100:.1f}% of face)",
    ]
    if yield_ is not None:
        px = mbs_price(cf_df, yield_, face)
        lines.append(f"  Price @ {yield_*100:.2f}% yield: {px:.3f}")
    if discount_curve is not None:
        y_ = mbs_yield(market_price_pct, cf_df, face)
        z  = oas(market_price_pct, cf_df, discount_curve, face)
        lines.append(f"  Market price    : {market_price_pct:.3f}")
        lines.append(f"  MBS yield       : {y_*100:.4f}%")
        lines.append(f"  OAS             : {z:.1f} bps")
    return "\n".join(lines)
