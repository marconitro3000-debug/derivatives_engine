"""
options/pde.py
Crank-Nicolson finite-difference PDE solver for European/American options, in
log-price space, supporting both flat volatility and local volatility (a
callable sigma(x, t) with x = ln(S)).

Solves, marching forward in tau = T - t (time-to-maturity, so tau=0 is
expiry), the Black-Scholes PDE transformed to log-price coordinates:

    dV/dtau = (r - q - 0.5*sigma(x,t)^2) dV/dx + 0.5*sigma(x,t)^2 d2V/dx2 - r*V

American exercise is enforced by projecting onto the intrinsic value after
each time step — a standard, simple approximation to the free-boundary
problem (accurate to the grid's resolution; not a full penalty/PSOR solve).
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy.linalg import solve_banded

SigmaInput = Union[float, Callable[[np.ndarray, float], np.ndarray]]  # flat sigma, or vectorized sigma(x_array, t)


def _sigma_fn(sigma: SigmaInput) -> Callable[[np.ndarray, float], np.ndarray]:
    """Normalizes `sigma` into a vectorized `f(x_array, t) -> array` — the PDE
    grid has hundreds of points per time step, so a per-point Python callback
    (rather than one array call) makes local-vol pricing orders of magnitude
    slower. Callable `sigma` inputs (e.g. from a local-vol surface) must
    already accept/return arrays; `LocalVolSurface.__call__` does."""
    if callable(sigma):
        return sigma
    flat = float(sigma)
    return lambda x, t: np.full(np.shape(x), flat)


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve `lower[i]*x[i-1] + diag[i]*x[i] + upper[i]*x[i+1] = rhs[i]` via
    scipy's banded solver (safer than a hand-rolled Thomas algorithm)."""
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


def price(
    S: float, K: float, T: float, r: float, sigma: SigmaInput,
    option: str = "call", style: str = "european", q: float = 0.0,
    n_space: int = 300, n_time: int = 300, x_width: float | None = None,
) -> dict:
    """
    Parameters
    ----------
    sigma   : a flat float, or a callable sigma(x, t) for local vol (x = ln(S), t = calendar time)
    style   : 'european' or 'american'
    x_width : half-width of the log-price grid (grid spans ln(S) +/- x_width). If
              None, sized automatically from a representative vol so the grid
              covers the realistic range of ln(S_T) regardless of maturity.

    Returns
    -------
    dict with keys: price, style, n_space, n_time
    """
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'.")
    if style not in ("european", "american"):
        raise ValueError("style must be 'european' or 'american'.")

    sig = _sigma_fn(sigma)
    x0 = np.log(S)
    if x_width is None:
        sigma_est = float(np.asarray(sig(np.array([x0]), T * 0.5))[0])
        x_width = max(0.8, 6.0 * sigma_est * np.sqrt(max(T, 1e-4)))
    xs = np.linspace(x0 - x_width, x0 + x_width, n_space + 1)
    dx = xs[1] - xs[0]
    dtau = T / n_time

    intrinsic = np.maximum((np.exp(xs) - K) if option == "call" else (K - np.exp(xs)), 0.0)
    V = intrinsic.copy()  # at tau = 0 (t = T), V = payoff

    def coeffs(t: float):
        sig2 = np.asarray(sig(xs, t), dtype=float) ** 2
        mu = r - q - 0.5 * sig2
        alpha = -mu / (2 * dx) + 0.5 * sig2 / dx**2   # coefficient of V[i-1]
        beta = -sig2 / dx**2 - r                       # coefficient of V[i]
        gamma = mu / (2 * dx) + 0.5 * sig2 / dx**2      # coefficient of V[i+1]
        return alpha, beta, gamma

    for step in range(n_time):
        tau_n, tau_np1 = step * dtau, (step + 1) * dtau
        t_n, t_np1 = T - tau_n, T - tau_np1

        alpha_n, beta_n, gamma_n = coeffs(t_n)
        alpha_1, beta_1, gamma_1 = coeffs(t_np1)

        lower = -0.5 * dtau * alpha_1[1:-1]
        diag = 1 - 0.5 * dtau * beta_1[1:-1]
        upper = -0.5 * dtau * gamma_1[1:-1]

        rhs = (
            0.5 * dtau * alpha_n[1:-1] * V[:-2]
            + (1 + 0.5 * dtau * beta_n[1:-1]) * V[1:-1]
            + 0.5 * dtau * gamma_n[1:-1] * V[2:]
        )

        disc_r = np.exp(-r * tau_np1)
        disc_q = np.exp(-q * tau_np1)
        if option == "call":
            v0_new, vn_new = 0.0, np.exp(xs[-1]) * disc_q - K * disc_r
        else:
            v0_new, vn_new = K * disc_r - np.exp(xs[0]) * disc_q, 0.0

        rhs[0] -= lower[0] * v0_new
        rhs[-1] -= upper[-1] * vn_new

        v_interior = _solve_tridiagonal(lower, diag, upper, rhs)
        V = np.concatenate(([v0_new], v_interior, [vn_new]))

        if style == "american":
            V = np.maximum(V, intrinsic)

    px = float(np.interp(x0, xs, V))
    return {"price": max(px, 0.0), "style": style, "n_space": n_space, "n_time": n_time}
