"""
volatility/local_vol.py
Dupire local volatility surface from an SSVI parametrisation.

Theory
------
Dupire (1994) showed that any consistent set of European call prices implies a
unique diffusion process  dS = r·S·dt + σ_local(S,t)·S·dW.

In total-variance coordinates  w(k,T) = σ²_BS(k,T)·T  (Gatheral 2006, p.11):

    σ²_local(k, T) = ∂w/∂T / g(k, T)

where  k = log(K/F(T))  is log-moneyness and

    g(k, T) = (1 − k·∂w/∂k/(2w))² − (∂w/∂k)²·(¼ + 1/w)/4 + ∂²w/∂k²/2

We parametrize w via SSVI (Gatheral & Jacquier 2014):

    w(k, θ) = θ/2 · {1 + ρ·φ(θ)·k + √[(φ(θ)·k + ρ)² + 1 − ρ²]}
    φ(θ)    = η / [θ^γ · (1+θ)^{1−γ}]

This gives *exact* analytical derivatives (no finite-differences on the strike
axis), a globally butterfly-arbitrage-free surface, and a well-defined ATM
term structure  θ(T)  that is interpolated via cubic spline.

Key properties compared with other approaches
----------------------------------------------
• vs Heston LV : SSVI LV is exactly consistent with the calibrated smile;
                 Heston LV can violate no-arbitrage in practice.
• vs non-parametric LV : more stable; no need to smooth noisy market data.
• vs Heston MC : simpler dynamics, no correlation parameter between S and V.

References
----------
Dupire (1994) "Pricing with a Smile", Risk.
Gatheral (2006) "The Volatility Surface", Wiley.
Gatheral & Jacquier (2014) "Arbitrage-Free SVI Volatility Surfaces", Quant Finance.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from scipy.interpolate import CubicSpline
from scipy.stats import norm

from ml.ssvi import SSVIParams


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class LocalVolSlice:
    """Local vol and implied vol evaluated on a log-moneyness grid."""
    k: np.ndarray             # log-moneyness
    T: float
    local_vol: np.ndarray     # σ_local(k, T)
    implied_vol: np.ndarray   # σ_BS(k, T) from SSVI
    density: np.ndarray       # risk-neutral density proxy g(k)

    @property
    def atm_local_vol(self) -> float:
        """σ_local at k=0 by linear interpolation."""
        return float(np.interp(0.0, self.k, self.local_vol))

    @property
    def atm_implied_vol(self) -> float:
        return float(np.interp(0.0, self.k, self.implied_vol))


@dataclass
class ModelComparisonRow:
    K: float
    T: float
    iv_bs: float              # flat-vol BS input
    price_bs: float
    price_local_vol: float
    price_local_vol_se: float
    diff_bps: float           # (LV - BS) in IV basis points (approx)


# ── Local volatility surface ──────────────────────────────────────────────────

class LocalVolSurface:
    """
    Dupire local volatility surface backed by an SSVI parametrisation.

    Parameters
    ----------
    ssvi        : calibrated SSVIParams (ρ, η, γ)
    T_nodes     : sorted maturity array [T₁ < T₂ < … < Tₙ]
    theta_nodes : ATM total variances  θᵢ = σ²_ATM(Tᵢ)·Tᵢ
    """

    def __init__(self, ssvi: SSVIParams,
                 T_nodes: np.ndarray, theta_nodes: np.ndarray):
        self.ssvi = ssvi
        T_nodes     = np.asarray(T_nodes,     dtype=float)
        theta_nodes = np.asarray(theta_nodes, dtype=float)

        if np.any(np.diff(T_nodes) <= 0):
            raise ValueError("T_nodes must be strictly increasing.")
        if np.any(np.diff(theta_nodes) < 0):
            raise ValueError(
                "theta_nodes must be non-decreasing (calendar no-arb violated)."
            )

        # Cubic spline through (T, θ) — 1st derivative gives dθ/dT
        self._theta_spline = CubicSpline(T_nodes, theta_nodes, extrapolate=True)

    # ── ATM term structure ────────────────────────────────────────────────────

    def theta(self, T: float) -> float:
        """θ(T) = σ²_ATM(T)·T, interpolated by cubic spline."""
        return float(np.maximum(self._theta_spline(T), 1e-10))

    def dtheta_dT(self, T: float) -> float:
        """dθ/dT from spline — positive if term structure is upward sloping."""
        return float(self._theta_spline(T, 1))          # 1st derivative

    def atm_iv(self, T: float) -> float:
        """σ_ATM(T) = √(θ(T)/T)."""
        return float(np.sqrt(self.theta(T) / max(T, 1e-10)))

    # ── SSVI analytical derivatives ───────────────────────────────────────────

    def _total_var(self, k: np.ndarray, theta: float) -> np.ndarray:
        return self.ssvi.total_var(k, theta)

    def _dw_dk(self, k: np.ndarray, theta: float) -> np.ndarray:
        """∂w/∂k — closed form for SSVI."""
        phi = self.ssvi.phi(theta)
        rho = self.ssvi.rho
        D   = np.sqrt((phi * k + rho) ** 2 + 1.0 - rho ** 2)
        return (theta * phi / 2.0) * (rho + (phi * k + rho) / D)

    def _d2w_dk2(self, k: np.ndarray, theta: float) -> np.ndarray:
        """∂²w/∂k² — closed form for SSVI."""
        phi = self.ssvi.phi(theta)
        rho = self.ssvi.rho
        D   = np.sqrt((phi * k + rho) ** 2 + 1.0 - rho ** 2)
        return (theta * phi ** 2 / 2.0) * (1.0 - rho ** 2) / D ** 3

    def _dw_dT(self, k: np.ndarray, T: float) -> np.ndarray:
        """
        ∂w/∂T via chain rule:  ∂w/∂T = (∂w/∂θ) · (dθ/dT)

        For SSVI:
            ∂φ/∂θ  = −φ · [γ/θ + (1−γ)/(1+θ)]
            ∂w/∂θ  = w/θ + (θ/2)·k·φ'·[ρ + (φk+ρ)/D]
        """
        theta  = self.theta(T)
        dtheta = self.dtheta_dT(T)

        phi   = self.ssvi.phi(theta)
        rho   = self.ssvi.rho
        gamma = self.ssvi.gamma

        D     = np.sqrt((phi * k + rho) ** 2 + 1.0 - rho ** 2)
        w     = self._total_var(k, theta)

        dphi_dtheta = -phi * (gamma / theta + (1.0 - gamma) / (1.0 + theta))

        # ∂w/∂θ = w/θ + (θ/2)·k·(∂φ/∂θ)·[ρ + (φk+ρ)/D]
        dw_dtheta = (w / theta
                     + (theta / 2.0) * k * dphi_dtheta
                     * (rho + (phi * k + rho) / D))

        return dw_dtheta * dtheta

    # ── Dupire density and local vol ──────────────────────────────────────────

    def density(self, k: np.ndarray, T: float) -> np.ndarray:
        """
        Risk-neutral density proxy (Gatheral 2006, Eq. 1.4):

            g(k,T) = (1 − k·∂w_k/(2w))² − (∂w_k)²·(¼ + 1/w)/4 + ∂²w_k/2

        g ≥ 0  ⟺  no butterfly arbitrage at (k, T).
        """
        k     = np.asarray(k, dtype=float)
        theta = self.theta(T)
        w     = np.maximum(self._total_var(k, theta), 1e-12)
        dw    = self._dw_dk(k, theta)
        d2w   = self._d2w_dk2(k, theta)

        return ((1.0 - k * dw / (2.0 * w)) ** 2
                - dw ** 2 * (0.25 + 1.0 / w) / 4.0
                + d2w / 2.0)

    def local_var(self, k: np.ndarray, T: float) -> np.ndarray:
        """σ²_local(k, T).  NaN where denominator ≤ 0 (would be arbitrage)."""
        k      = np.asarray(k, dtype=float)
        dw_dT  = self._dw_dT(k, T)
        g      = self.density(k, T)
        return np.where(g > 1e-10, dw_dT / g, np.nan)

    def __call__(self, k, T: float) -> np.ndarray:
        """σ_local(k, T) = √max(σ²_local, 0)."""
        lv2 = self.local_var(np.atleast_1d(np.asarray(k, dtype=float)), T)
        return np.sqrt(np.maximum(lv2, 0.0))

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        """BS implied vol from SSVI: σ_BS = √(w/T)."""
        theta = self.theta(T)
        w     = self._total_var(np.asarray(k, dtype=float), theta)
        return np.sqrt(np.maximum(w / T, 0.0))

    # ── Slice evaluation ──────────────────────────────────────────────────────

    def slice(
        self, T: float, k_grid: Optional[np.ndarray] = None
    ) -> LocalVolSlice:
        """Full local-vol / implied-vol / density slice at maturity T."""
        if k_grid is None:
            k_grid = np.linspace(-0.45, 0.45, 201)
        k_grid = np.asarray(k_grid, dtype=float)
        return LocalVolSlice(
            k           = k_grid,
            T           = T,
            local_vol   = self(k_grid, T),
            implied_vol = self.implied_vol(k_grid, T),
            density     = self.density(k_grid, T),
        )

    def surface(
        self,
        T_grid: np.ndarray,
        k_grid: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate local vol surface on a (T, k) grid.

        Returns
        -------
        K_mesh : 2-D array of log-moneyness  (n_T × n_k)
        T_mesh : 2-D array of maturities
        LV     : local vol surface  σ_local(k, T)
        """
        if k_grid is None:
            k_grid = np.linspace(-0.40, 0.40, 101)
        K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
        LV = np.zeros_like(K_mesh)
        for i, T in enumerate(T_grid):
            LV[i] = self(K_mesh[i], T)
        return K_mesh, T_mesh, LV

    # ── Smile dynamics: LV vs IV comparison ──────────────────────────────────

    def lv_iv_ratio(self, T: float, k_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        σ_local / σ_BS at each k.

        Classic result: for pure SV models the ratio σ_local/σ_BS depends on
        the correlation structure.  For a flat smile (ρ=0, symmetric) the
        local vol is ≈ twice the implied vol slope (Derman-Kani).
        """
        sl = self.slice(T, k_grid)
        iv = np.maximum(sl.implied_vol, 1e-8)
        return sl.local_vol / iv


# ── Monte Carlo pricer under local vol ───────────────────────────────────────

def mc_price_local_vol(
    S: float,
    K: float,
    T: float,
    r: float,
    lv: LocalVolSurface,
    option: str = "call",
    n_sims: int = 50_000,
    n_steps: int = 100,
    antithetic: bool = True,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Price a European option by Monte Carlo under local volatility dynamics:

        dS/S = r·dt + σ_local(log(S/F(t)), t)·dW

    where F(t) = S₀·e^{r·t} is the forward price.

    Parameters
    ----------
    antithetic : use antithetic variates for variance reduction

    Returns
    -------
    (price, std_error)
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps

    if antithetic:
        n_half = n_sims // 2
        S_pos  = np.full(n_half, float(S))
        S_neg  = np.full(n_half, float(S))
    else:
        S_t = np.full(n_sims, float(S))

    for step in range(n_steps):
        t   = step * dt
        F_t = S * np.exp(r * t)           # rolling forward for log-moneyness

        if antithetic:
            Z       = rng.standard_normal(n_half)
            k_pos   = np.log(np.maximum(S_pos, 1e-10) / F_t)
            k_neg   = np.log(np.maximum(S_neg, 1e-10) / F_t)
            lv_pos  = _safe_lv(lv, k_pos, t + dt)
            lv_neg  = _safe_lv(lv, k_neg, t + dt)
            S_pos   = S_pos * np.exp((r - 0.5 * lv_pos**2) * dt + lv_pos * np.sqrt(dt) * Z)
            S_neg   = S_neg * np.exp((r - 0.5 * lv_neg**2) * dt - lv_neg * np.sqrt(dt) * Z)
        else:
            Z    = rng.standard_normal(n_sims)
            k    = np.log(np.maximum(S_t, 1e-10) / F_t)
            lv_t = _safe_lv(lv, k, t + dt)
            S_t  = S_t * np.exp((r - 0.5 * lv_t**2) * dt + lv_t * np.sqrt(dt) * Z)

    if antithetic:
        S_T    = np.concatenate([S_pos, S_neg])
    else:
        S_T    = S_t

    payoff = (np.maximum(S_T - K, 0.0) if option == "call"
              else np.maximum(K - S_T, 0.0))
    df     = np.exp(-r * T)
    price  = df * float(np.mean(payoff))
    se     = df * float(np.std(payoff)) / np.sqrt(len(payoff))
    return price, se


def _safe_lv(lv: LocalVolSurface, k: np.ndarray, T: float,
             fallback: float = 0.20) -> np.ndarray:
    """Evaluate local vol with NaN → fallback and clipping to [1%, 300%]."""
    sigma = lv(k, T)
    sigma = np.where(np.isnan(sigma) | (sigma <= 0), fallback, sigma)
    return np.clip(sigma, 0.01, 3.0)


# ── Model comparison utility ──────────────────────────────────────────────────

def compare_with_bs(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    lv: LocalVolSurface,
    n_sims: int = 30_000,
    seed: int = 42,
) -> list[ModelComparisonRow]:
    """
    Price a strip of calls under local vol and compare with Black-Scholes.

    Black-Scholes uses the SSVI implied vol at each strike (same smile
    calibration), so any price difference is purely from the dynamic
    correction introduced by the local vol process.

    Returns a list of ModelComparisonRow.
    """
    from options.black_scholes import price as bs_price
    from options.implied_vol import implied_vol as extract_iv

    F    = S * np.exp(r * T)
    rows = []

    for K in strikes:
        k        = np.log(K / F)
        iv_ssvi  = float(lv.implied_vol(np.array([k]), T)[0])
        px_bs    = bs_price(S, K, T, r, iv_ssvi, "call")
        px_lv, se = mc_price_local_vol(S, K, T, r, lv,
                                        option="call", n_sims=n_sims, seed=seed)

        # Approximate IV difference in bps
        try:
            iv_lv   = extract_iv(S, K, T, r, px_lv, "call")
            diff_bps = (iv_lv - iv_ssvi) * 1e4
        except Exception:
            diff_bps = float("nan")

        rows.append(ModelComparisonRow(
            K=K, T=T,
            iv_bs=iv_ssvi,
            price_bs=px_bs,
            price_local_vol=px_lv,
            price_local_vol_se=se,
            diff_bps=diff_bps,
        ))

    return rows


# ── Builder from calibrated SSVI + ATM vols ──────────────────────────────────

def from_ssvi_and_atm(
    ssvi: SSVIParams,
    maturities: np.ndarray,
    atm_ivs: np.ndarray,
) -> LocalVolSurface:
    """
    Construct a LocalVolSurface from a calibrated SSVI + ATM implied vols.

    Parameters
    ----------
    ssvi       : fitted SSVIParams
    maturities : sorted array of maturities  [T₁ < … < Tₙ]
    atm_ivs    : ATM implied vols  σ_ATM(Tᵢ)  at each maturity
    """
    maturities  = np.asarray(maturities, dtype=float)
    atm_ivs     = np.asarray(atm_ivs,    dtype=float)
    theta_nodes = atm_ivs ** 2 * maturities     # θᵢ = σ²_ATM·Tᵢ
    return LocalVolSurface(ssvi, maturities, theta_nodes)
