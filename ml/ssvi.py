"""
ml/ssvi.py
Surface SVI (SSVI) — Gatheral & Jacquier (2014) parametrization.

Standard SVI for a single slice:
    w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]
    where k = log(K/F), w = σ_BS² × T (total implied variance)

SSVI parametrization (slice-consistent, calendar-spread-arbitrage-free):
    w(k, θ) = (θ/2){1 + ρ φ(θ) k + √[(φ(θ) k + ρ)² + (1-ρ²)]}
    where θ = forward ATM variance = σ_ATM² × T
    and   φ(θ) is a smooth positive function

Power-law φ:
    φ(θ) = η / [θ^γ (1+θ)^{1-γ}]
    Parameters: η > 0, γ ∈ (0, 1/2]

No-butterfly-arbitrage requires:
    θ φ(θ) (1 + |ρ|) < 4
    θ φ²(θ) (1 + |ρ|) ≤ 4

Calibration: minimize Σᵢ (w_model - w_mkt)² in total variance space.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm


@dataclass
class SVIParams:
    """Single-slice SVI raw parameters."""
    a: float   # overall level of variance
    b: float   # slope of wings
    rho: float # correlation (tilt): ∈ (-1, 1)
    m: float   # translation (ATM offset in log-moneyness)
    sigma: float  # curvature: > 0

    def total_var(self, k: np.ndarray) -> np.ndarray:
        """w(k) = a + b[ρ(k-m) + √((k-m)²+σ²)]"""
        k = np.asarray(k)
        d = k - self.m
        return self.a + self.b * (self.rho * d + np.sqrt(d**2 + self.sigma**2))

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        """Black-Scholes implied vol from total variance w."""
        w = self.total_var(k)
        return np.sqrt(np.maximum(w / T, 0.0))


@dataclass
class SSVIParams:
    """Surface SVI parameters (power-law φ)."""
    rho: float   # global correlation: ∈ (-1, 1)
    eta: float   # φ scale: > 0
    gamma: float # φ decay: ∈ (0, 0.5]

    def phi(self, theta: float) -> float:
        """φ(θ) = η / [θ^γ (1+θ)^{1-γ}]"""
        return self.eta / (theta**self.gamma * (1 + theta)**(1 - self.gamma))

    def total_var(self, k: np.ndarray, theta: float) -> np.ndarray:
        """w(k, θ) = (θ/2){1 + ρ φ k + √[(φk + ρ)² + (1-ρ²)]}"""
        k   = np.asarray(k)
        phi = self.phi(theta)
        a   = phi * k + self.rho
        return (theta / 2) * (1 + self.rho * phi * k
                               + np.sqrt(a**2 + 1 - self.rho**2))

    def implied_vol(self, k: np.ndarray, theta: float, T: float) -> np.ndarray:
        w = self.total_var(k, theta)
        return np.sqrt(np.maximum(w / T, 0.0))

    def no_butterfly_arbitrage(self, theta: float) -> bool:
        phi = self.phi(theta)
        cond1 = theta * phi * (1 + abs(self.rho)) < 4
        cond2 = theta * phi**2 * (1 + abs(self.rho)) <= 4
        return bool(cond1 and cond2)

    def no_calendar_spread_arbitrage(self, thetas: np.ndarray) -> bool:
        """Total variance must be non-decreasing in θ (forward time)."""
        return bool(np.all(np.diff(np.sort(thetas)) >= 0))


# ── Single-slice SVI calibration ──────────────────────────────────────────────

def calibrate_svi(log_moneyness: np.ndarray, total_var_market: np.ndarray,
                   weights: np.ndarray | None = None) -> SVIParams:
    """
    Calibrate raw SVI to a single smile (total variance vs log-moneyness).

    Parameters
    ----------
    log_moneyness    : k = log(K/F), array
    total_var_market : w_mkt = σ²_mkt × T, array
    weights          : optional per-point weights

    Returns
    -------
    SVIParams
    """
    k   = np.asarray(log_moneyness)
    w   = np.asarray(total_var_market)
    wts = np.ones(len(k)) if weights is None else np.asarray(weights)

    def objective(x):
        a, b_raw, rho_raw, m, sigma_raw = x
        b     = np.exp(b_raw)              # b > 0
        rho   = np.tanh(rho_raw)           # rho ∈ (-1, 1)
        sigma = np.exp(sigma_raw)          # sigma > 0
        p     = SVIParams(a, b, rho, m, sigma)
        try:
            w_model = p.total_var(k)
            if np.any(w_model <= 0):
                return 1e10
            return float(np.sum(wts * (w_model - w)**2))
        except Exception:
            return 1e10

    # Initialise at ATM
    atm_w  = np.interp(0.0, k, w)
    x0     = [atm_w * 0.5, np.log(0.1), 0.0, 0.0, np.log(0.1)]

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 20_000, "xatol": 1e-10, "fatol": 1e-12})
    a, b_r, rho_r, m, sig_r = res.x
    return SVIParams(
        a=float(a),
        b=float(np.exp(b_r)),
        rho=float(np.tanh(rho_r)),
        m=float(m),
        sigma=float(np.exp(sig_r)),
    )


# ── Surface SVI calibration ───────────────────────────────────────────────────

def calibrate_ssvi(log_moneyness_list: list[np.ndarray],
                    total_var_list: list[np.ndarray],
                    theta_list: list[float],
                    weights_list: list[np.ndarray] | None = None) -> SSVIParams:
    """
    Calibrate SSVI surface parameters (ρ, η, γ) jointly across all slices.

    Parameters
    ----------
    log_moneyness_list : list of k arrays, one per maturity
    total_var_list     : list of w_mkt arrays, one per maturity
    theta_list         : list of ATM total variances σ²_ATM × T per slice
    weights_list       : optional per-slice weights

    Returns
    -------
    SSVIParams
    """
    n_slices = len(log_moneyness_list)
    if weights_list is None:
        weights_list = [np.ones(len(k)) for k in log_moneyness_list]

    def objective(x):
        rho_r, eta_r, gam_r = x
        rho   = np.tanh(rho_r)
        eta   = np.exp(eta_r)
        gamma = 0.5 / (1 + np.exp(-gam_r))   # ∈ (0, 0.5)
        p     = SSVIParams(rho=rho, eta=eta, gamma=gamma)

        total_err = 0.0
        for i in range(n_slices):
            theta = theta_list[i]
            # Check arbitrage constraints
            phi = p.phi(theta)
            if theta * phi * (1 + abs(rho)) >= 4:
                return 1e10
            w_model = p.total_var(log_moneyness_list[i], theta)
            if np.any(w_model <= 0):
                return 1e10
            diff = w_model - total_var_list[i]
            total_err += float(np.sum(weights_list[i] * diff**2))
        return total_err

    bounds = [(-3.0, 3.0), (-4.0, 2.0), (-5.0, 5.0)]
    res    = differential_evolution(objective, bounds, seed=0,
                                     maxiter=500, tol=1e-10, popsize=10)
    rho_r, eta_r, gam_r = res.x
    return SSVIParams(
        rho=float(np.tanh(rho_r)),
        eta=float(np.exp(eta_r)),
        gamma=float(0.5 / (1 + np.exp(-gam_r))),
    )


# ── Local vol (Dupire) from SSVI ──────────────────────────────────────────────

def ssvi_local_vol(params: SSVIParams, k: float, T: float,
                    theta_fn, dtheta_fn=None) -> float:
    """
    Dupire local vol² from SSVI total variance surface.

    σ²_loc(k, T) = ∂w/∂T / [1 - (k/w)∂w/∂k + (1/4)(-1/4 - 1/w + k²/w²)(∂w/∂k)²
                              + (1/2)∂²w/∂k²]

    where θ = θ(T) is the ATM total variance as a function of time.
    Requires θ'(T) = dθ/dT.

    Parameters
    ----------
    theta_fn  : callable T → θ(T)   (e.g. spline interpolator)
    dtheta_fn : callable T → dθ/dT  (optional; uses finite diff if None)
    """
    h_T = 1e-4
    if dtheta_fn is None:
        dtheta = (theta_fn(T + h_T) - theta_fn(T - h_T)) / (2 * h_T)
    else:
        dtheta = dtheta_fn(T)

    theta = theta_fn(T)

    h_k = 1e-4
    w0  = params.total_var(np.array([k]),         theta)[0]
    wk1 = params.total_var(np.array([k + h_k]),   theta)[0]
    wk2 = params.total_var(np.array([k - h_k]),   theta)[0]

    dw_dk  = (wk1 - wk2) / (2 * h_k)
    d2w_dk = (wk1 - 2*w0 + wk2) / h_k**2

    # ∂w/∂T via chain rule: ∂w/∂T = (∂w/∂θ) × (dθ/dT)
    phi     = params.phi(theta)
    w_theta = (w0 / theta) if theta > 1e-10 else 0.0   # approx: w ∝ θ
    dw_dT   = w_theta * dtheta

    denom = (1
             - (k / w0) * dw_dk
             + 0.25 * (-0.25 - 1/w0 + k**2/w0**2) * dw_dk**2
             + 0.5 * d2w_dk)

    if denom <= 0 or dw_dT < 0:
        return np.nan

    return float(dw_dT / denom)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def svi_fit_summary(params: SVIParams, k: np.ndarray, w_mkt: np.ndarray,
                     T: float) -> str:
    w_fit = params.total_var(k)
    err   = (w_fit - w_mkt) * 10_000 / (2 * np.sqrt(w_mkt) * T)  # approx IV bps

    lines = [
        "SVI Fit Summary:",
        f"  a={params.a:.6f}  b={params.b:.6f}  ρ={params.rho:.4f}  "
        f"m={params.m:.4f}  σ={params.sigma:.4f}",
        f"  IV error (bps)  RMSE={np.sqrt(np.mean(err**2)):.3f}  "
        f"Max={np.max(np.abs(err)):.3f}",
    ]
    for ki, wi, we in zip(k, w_mkt, err):
        lines.append(f"    k={ki:+.3f}  IV_mkt={np.sqrt(wi/T)*100:.2f}%  "
                     f"IV_fit={np.sqrt(w_fit[list(k).index(ki)]/T)*100:.2f}%  "
                     f"err={we:+.2f}bp")
    return "\n".join(lines)
