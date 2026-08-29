"""
models/heston.py
Heston stochastic volatility model with semi-analytical pricing via the
characteristic function (Lewis / Carr-Madan style integration).

Dynamics under the risk-neutral measure:

    dS_t = r S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v
    d<W^S, W^v>_t = rho dt

Parameters
----------
    v0    : initial variance              (v0 > 0)
    kappa : mean-reversion speed           (kappa > 0)
    theta : long-run variance              (theta > 0)
    xi    : vol-of-vol                      (xi > 0)
    rho   : correlation spot/vol           (-1 < rho < 1)

Feller condition for v_t > 0:  2*kappa*theta >= xi^2
"""

import numpy as np
from dataclasses import dataclass, asdict
from scipy.integrate import quad


# ── parameter container ───────────────────────────────────────────────────────

@dataclass
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def as_array(self) -> np.ndarray:
        return np.array([self.v0, self.kappa, self.theta, self.xi, self.rho])

    @classmethod
    def from_array(cls, arr) -> "HestonParams":
        return cls(*[float(x) for x in arr])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HestonParams":
        return cls(v0=d["v0"], kappa=d["kappa"], theta=d["theta"],
                   xi=d["xi"], rho=d["rho"])

    def feller_satisfied(self) -> bool:
        return 2 * self.kappa * self.theta >= self.xi ** 2


# ── characteristic function ───────────────────────────────────────────────────

def _char_func(u: complex, S: float, T: float, r: float, q: float,
               p: HestonParams) -> complex:
    """
    Heston characteristic function (the "little Heston trap" formulation of
    Albrecher et al., which is numerically stable). `q` is the continuous
    dividend yield — the risk-neutral drift is (r - q).
    """
    x0 = np.log(S)
    kappa, theta, xi, rho, v0 = p.kappa, p.theta, p.xi, p.rho, p.v0

    d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + (xi ** 2) * (1j * u + u ** 2))
    g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

    exp_dt = np.exp(-d * T)
    C = ((r - q) * 1j * u * T
         + (kappa * theta / xi ** 2)
         * ((kappa - rho * xi * 1j * u - d) * T
            - 2 * np.log((1 - g * exp_dt) / (1 - g))))
    D = ((kappa - rho * xi * 1j * u - d) / xi ** 2) * ((1 - exp_dt) / (1 - g * exp_dt))

    return np.exp(C + D * v0 + 1j * u * x0)


# ── pricing via Gil-Pelaez / Lewis integration ────────────────────────────────

def price(S: float, K: float, T: float, r: float,
          p: HestonParams, option: str = "call", q: float = 0.0) -> float:
    """
    European option price under Heston via the two-integral Gil-Pelaez formula.
    `q` is the continuous dividend yield (default 0 — preserves prior behavior
    for callers that don't pass it).
    """
    def integrand(u, num):
        if num == 1:
            cf = _char_func(u - 1j, S, T, r, q, p) / _char_func(-1j, S, T, r, q, p)
        else:
            cf = _char_func(u, S, T, r, q, p)
        return np.real(np.exp(-1j * u * np.log(K)) * cf / (1j * u))

    P1 = 0.5 + (1 / np.pi) * quad(lambda u: integrand(u, 1), 1e-8, 200, limit=200)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda u: integrand(u, 2), 1e-8, 200, limit=200)[0]

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    call = S * disc_q * P1 - K * disc_r * P2
    if option == "call":
        return float(max(call, 0.0))
    elif option == "put":
        return float(max(call - S * disc_q + K * disc_r, 0.0))
    else:
        raise ValueError("option must be 'call' or 'put'.")


# ── default seed ──────────────────────────────────────────────────────────────

def default_params(atm_var: float = 0.04) -> HestonParams:
    """Reasonable seed for calibration when no warm start is available."""
    return HestonParams(v0=atm_var, kappa=2.0, theta=atm_var, xi=0.3, rho=-0.5)
