"""
structured/cdo.py
CDO tranche pricing via Gaussian copula (Large Homogeneous Portfolio).

Model assumptions (Vasicek LHP):
  - N identical credits, uniform PD p, recovery R, pairwise correlation ρ
  - Single common factor Z ~ N(0,1); defaults conditionally independent
  - Conditional default probability given Z=z:
        p(z) = Φ((Φ⁻¹(p) − √ρ · z) / √(1−ρ))
  - Portfolio loss rate (conditional on z):
        L(z) = (1−R) · p(z)   [LHP: law of large numbers kicks in]

Tranche [A, D] (0 ≤ A < D ≤ 1):
    Tranche loss = (max(L−A, 0) − max(L−D, 0)) / (D−A)
    Expected tranche loss:
        ETL = (E[max(L−A,0)] − E[max(L−D,0)]) / (D−A)
    where E[max(L−K, 0)] = ∫ max(L(z)−K, 0) φ(z) dz   [1-D Gaussian integral]

Fair tranche spread:
    Protection leg: PL = Σᵢ DF(tᵢ) · (ETL(tᵢ) − ETL(tᵢ₋₁)) / (D−A)
    Premium RPV01 : = Σᵢ Δtᵢ · DF(tᵢ) · (1 − ETL(tᵢ))
    s* = PL / RPV01

Cumulative PD at horizon T: p(T) = 1 − exp(−h·T), h = −ln(1−p_annual)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import brentq

_Phi   = norm.cdf
_PhiI  = norm.ppf
_phi   = norm.pdf


# ── core LHP formulas ─────────────────────────────────────────────────────────

def _cond_pd(z: float, pd_1y: float, rho: float) -> float:
    """P(default | Z=z) in the Gaussian copula LHP."""
    return float(_Phi((_PhiI(pd_1y) - np.sqrt(rho) * z) / np.sqrt(1.0 - rho)))


def _expected_shortfall_lhp(K: float, pd_1y: float, rho: float,
                              recovery: float, n_quad: int = 200) -> float:
    """
    E[max(L − K, 0)] using Gauss-Hermite quadrature.
    L = (1 − R) · p(z),  z ~ N(0,1)
    """
    LGD = 1.0 - recovery

    def integrand(z):
        loss = LGD * _cond_pd(z, pd_1y, rho)
        return max(loss - K, 0.0) * _phi(z)

    val, _ = quad(integrand, -6.0, 6.0, limit=n_quad)
    return float(val)


def expected_tranche_loss(attachment: float, detachment: float,
                           pd_1y: float, rho: float,
                           recovery: float = 0.40) -> float:
    """
    Expected tranche loss as a fraction of tranche notional.

    Parameters
    ----------
    attachment : tranche lower bound (e.g. 0.03 = 3%)
    detachment : tranche upper bound (e.g. 0.07 = 7%)
    pd_1y      : annual default probability of each name
    rho        : pairwise asset-return correlation
    recovery   : recovery rate

    Returns
    -------
    ETL ∈ [0, 1]
    """
    if attachment >= detachment:
        raise ValueError("attachment must be < detachment")

    ea = _expected_shortfall_lhp(attachment, pd_1y, rho, recovery)
    ed = _expected_shortfall_lhp(detachment, pd_1y, rho, recovery)
    return float((ea - ed) / (detachment - attachment))


def tranche_fair_spread(attachment: float, detachment: float,
                         pd_1y: float, rho: float,
                         discount_curve,
                         recovery: float = 0.40,
                         maturity: float = 5.0,
                         pay_freq: int = 4) -> dict:
    """
    Fair CDO tranche spread (bps per annum).

    Uses piecewise-constant hazard approximation for cumulative PD:
        PD(T) = 1 − exp(−h · T),  h = −ln(1 − pd_1y)

    Parameters
    ----------
    attachment, detachment : tranche bounds [0, 1]
    pd_1y     : annual reference entity PD
    rho       : pairwise correlation
    discount_curve : DiscountCurve (from rates module)
    recovery  : recovery rate
    maturity  : tenor in years
    pay_freq  : premium payment frequency (4 = quarterly)

    Returns
    -------
    dict: fair_spread_bps, protection_leg, rpv01, etl_at_maturity,
          expected_loss_pct, attachment, detachment
    """
    h     = -np.log(1.0 - pd_1y)
    dt    = 1.0 / pay_freq
    times = np.arange(dt, maturity + 1e-9, dt)

    # Expected tranche loss at each payment date
    etl_t = []
    for t in times:
        pd_t = 1.0 - np.exp(-h * t)
        etl_t.append(expected_tranche_loss(attachment, detachment, pd_t, rho, recovery))
    etl_t = np.array(etl_t)

    # Protection leg: PV of loss increments
    prot_leg = 0.0
    for i, t in enumerate(times):
        df   = discount_curve.discount_factor(t)
        dETL = etl_t[i] - (etl_t[i - 1] if i > 0 else 0.0)
        prot_leg += df * dETL

    # RPV01: PV of 1 bps × (1 − tranche loss outstanding)
    rpv01 = 0.0
    for i, t in enumerate(times):
        df = discount_curve.discount_factor(t)
        rpv01 += dt * df * (1.0 - etl_t[i])

    fair_spread = (prot_leg / rpv01) if rpv01 > 1e-12 else np.nan

    return {
        "fair_spread_bps":    float(fair_spread * 10_000),
        "protection_leg":     float(prot_leg),
        "rpv01":              float(rpv01),
        "etl_at_maturity":    float(etl_t[-1]),
        "expected_loss_pct":  float(etl_t[-1] * 100),
        "attachment":         float(attachment),
        "detachment":         float(detachment),
    }


def cdo_structure(attachment_points: list[float],
                   pd_1y: float, rho: float,
                   discount_curve,
                   recovery: float = 0.40,
                   maturity: float = 5.0,
                   pay_freq: int = 4) -> list[dict]:
    """
    Price all tranches in a CDO capital structure.

    Parameters
    ----------
    attachment_points : list of tranche boundaries, e.g. [0, 0.03, 0.07, 0.12, 0.22, 1.0]

    Returns
    -------
    List of dicts, one per tranche, each with fair_spread_bps, etl, name, ...
    """
    results = []
    for i in range(len(attachment_points) - 1):
        A = attachment_points[i]
        D = attachment_points[i + 1]
        res = tranche_fair_spread(A, D, pd_1y, rho, discount_curve, recovery,
                                   maturity, pay_freq)

        # Name convention
        if i == 0:
            name = "Equity"
        elif i == len(attachment_points) - 2:
            name = "Senior"
        elif i == 1:
            name = "Mezzanine 1"
        elif i == 2:
            name = "Mezzanine 2"
        else:
            name = f"Tranche {i+1}"

        res["name"] = name
        results.append(res)
    return results


def loss_distribution(pd_1y: float, rho: float, recovery: float = 0.40,
                       n_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    LHP portfolio loss distribution f(L) via change of variables.

    Returns (loss_grid, density) where density is the PDF of portfolio loss rate L.
    Note: there is a point mass at L=0 (probability all survive).
    """
    LGD  = 1.0 - recovery
    Lmax = LGD
    L    = np.linspace(1e-6, Lmax - 1e-6, n_points)

    # L = LGD * Phi(a - b*z) → z = (Phi^{-1}(L/LGD) - a) / (-b)
    a   = _PhiI(pd_1y) / np.sqrt(1.0 - rho)
    b   = np.sqrt(rho  / (1.0 - rho))

    # dP/dL via Jacobian
    eps = _PhiI(L / LGD)
    z   = (eps - _PhiI(pd_1y)) / np.sqrt(rho)    # factor value
    # Jacobian: dz/dL = 1 / (LGD · phi(eps) · sqrt(rho/(1-rho)))
    dz_dL  = 1.0 / (LGD * _phi(eps) * np.sqrt(rho / (1.0 - rho)))
    density = _phi(z) * np.abs(dz_dL)

    return L, density
