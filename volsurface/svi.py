"""
volsurface/svi.py
The parametric surfaces the neural model has to beat: raw SVI and SSVI.

**Raw SVI** (Gatheral 2004) fits one smile at a time:

    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]

Five parameters per expiry, and it fits an individual smile extremely well. Its
weakness is structural: each slice is calibrated independently, so nothing ties
adjacent maturities together and interpolating between slices routinely produces
calendar-spread arbitrage. This is the baseline that looks good on RMSE and bad
on the diagnostics.

**SSVI** (Gatheral & Jacquier 2014) fixes that by making every slice a function
of a single ATM variance parameter:

    w(k, theta) = (theta / 2) { 1 + rho phi(theta) k
                                + sqrt[(phi(theta) k + rho)^2 + 1 - rho^2] }
    phi(theta)  = eta / [ theta^gamma (1 + theta)^(1 - gamma) ]

Three global parameters (rho, eta, gamma) plus one theta per expiry. With theta
non-decreasing the surface is provably free of calendar-spread arbitrage, and
butterfly arbitrage is excluded by two explicit parameter inequalities. The
price is rigidity: three global shape parameters cannot follow a real smile that
changes character between the front week and the one-year point.

That trade-off -- flexible and unsafe, or safe and rigid -- is the gap the
penalised neural surface in `volsurface/neural/` is built to close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from volsurface.surface import VolSurface


# -- raw SVI ------------------------------------------------------------------

@dataclass
class SVIParams:
    """Single-slice raw SVI parameters."""

    a: float        # overall variance level
    b: float        # wing slope, b >= 0
    rho: float      # tilt / skew, in (-1, 1)
    m: float        # smile centre in log-moneyness
    sigma: float    # curvature, > 0

    def total_var(self, k: np.ndarray) -> np.ndarray:
        d = np.asarray(k, dtype=float) - self.m
        return self.a + self.b * (self.rho * d + np.sqrt(d ** 2 + self.sigma ** 2))

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        return np.sqrt(np.maximum(self.total_var(k), 0.0) / T)

    def is_butterfly_free(self) -> bool:
        """Lee's wing bound plus positivity -- necessary, not sufficient."""
        if self.b < 0 or self.sigma <= 0 or abs(self.rho) >= 1:
            return False
        if self.b * (1 + abs(self.rho)) > 4.0:
            return False
        w_min = self.a + self.b * self.sigma * np.sqrt(1 - self.rho ** 2)
        return bool(w_min >= 0)


def _svi_inner_solve(y: np.ndarray, w: np.ndarray, wts: np.ndarray,
                     s: float) -> tuple[np.ndarray, float]:
    """Solve the *linear* sub-problem of the quasi-explicit SVI calibration.

    With ``y = (k - m) / sigma`` fixed, raw SVI is linear in a reparametrisation:

        w = a + d * y + c * sqrt(y^2 + 1),     c = b*sigma,  d = rho*b*sigma

    so for given ``(m, sigma)`` the remaining fit is a small constrained least
    squares in ``(a, d, c)``. The domain and Lee's wing bound become the linear
    constraints

        0 <= c <= 4*sigma,   |d| <= c,   |d| <= 4*sigma - c,   0 <= a <= max(w)

    which SLSQP handles exactly. Returns ``((a, d, c), residual)``.
    """
    basis = np.stack([np.ones_like(y), y, np.sqrt(y ** 2 + 1.0)], axis=1)
    sw = np.sqrt(wts)
    A, b_vec = basis * sw[:, None], w * sw

    def resid(x):
        return float(np.sum((A @ x - b_vec) ** 2))

    def grad(x):
        return 2.0 * A.T @ (A @ x - b_vec)

    w_max = float(np.max(w))
    # The 1e-6 margin keeps |rho| strictly below 1: rho = d/c is undefined in
    # the model's own domain at the boundary, and `is_butterfly_free` rejects it.
    constraints = [
        {"type": "ineq", "fun": lambda x: (1 - 1e-6) * x[2] - abs(x[1])},  # |d| <= c
        {"type": "ineq", "fun": lambda x: 4 * s - x[2] - abs(x[1])},       # |d| <= 4s - c
    ]
    bounds = [(0.0, max(w_max, 1e-8)), (-4 * s, 4 * s), (0.0, 4 * s)]

    # Seed from the unconstrained solution, clipped into the box.
    x0 = np.clip(np.linalg.lstsq(A, b_vec, rcond=None)[0],
                 [b[0] for b in bounds], [b[1] for b in bounds])

    res = minimize(resid, x0, jac=grad, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 200, "ftol": 1e-14})
    x = res.x if res.success else x0
    return x, resid(x)


def calibrate_svi(log_moneyness: np.ndarray, total_var: np.ndarray,
                  weights: np.ndarray | None = None) -> SVIParams:
    """Quasi-explicit calibration of raw SVI to one smile (Zeliade Systems, 2009).

    A direct five-parameter search over ``(a, b, rho, m, sigma)`` is not a good
    idea, and the failure is not subtle: the objective has long flat valleys
    where ``rho -> -1`` and ``sigma -> 0`` trade off against each other, and a
    simplex method started at a fixed point walks into them and returns a
    degenerate slice with a large negative ``a``. It looks like a converged fit
    and prices nonsense.

    The fix exploits the structure of the model. For fixed ``(m, sigma)`` the
    smile is *linear* in the remaining parameters, so the problem splits into a
    constrained linear least squares solved exactly (`_svi_inner_solve`) inside a
    two-dimensional search over ``(m, sigma)``. Two dimensions is small enough
    that a multi-start simplex covers the space reliably, and the inner solve
    cannot leave the arbitrage-free region because the constraints are imposed
    there directly.
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_var, dtype=float)
    wts = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)
    wts = wts / wts.mean()

    def outer(x) -> float:
        m, log_s = float(x[0]), float(x[1])
        s = float(np.exp(log_s))
        if not (1e-4 < s < 10.0) or abs(m) > 5.0:
            return 1e10
        return _svi_inner_solve((k - m) / s, w, wts, s)[1]

    # Multi-start: the (m, sigma) surface is smooth but not unimodal, and the
    # right basin depends on how skewed the smile is. Nine starts over a
    # two-dimensional space is enough to find it; the tolerances are loose
    # because the inner solve, not the outer search, sets the final accuracy.
    k_span = max(float(k.max() - k.min()), 1e-3)
    best_x, best_val = None, np.inf
    for m0 in (-0.3 * k_span, 0.0, 0.3 * k_span):
        for s0 in (0.05, 0.2, 0.6):
            res = minimize(outer, [m0, np.log(s0)], method="Nelder-Mead",
                           options={"maxiter": 400, "xatol": 1e-6, "fatol": 1e-12})
            if res.fun < best_val:
                best_val, best_x = float(res.fun), res.x

    # Polish the winning basin at full tolerance.
    res = minimize(outer, best_x, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-9, "fatol": 1e-14})
    if res.fun < best_val:
        best_x = res.x

    m, s = float(best_x[0]), float(np.exp(best_x[1]))
    (a, d, c), _ = _svi_inner_solve((k - m) / s, w, wts, s)

    b = c / s
    rho = float(np.clip(d / c, -1.0, 1.0)) if c > 1e-12 else 0.0
    return SVIParams(a=float(a), b=float(b), rho=rho, m=m, sigma=s)


# -- SSVI ---------------------------------------------------------------------

@dataclass
class SSVIParams:
    """Global SSVI shape parameters with the power-law ``phi``."""

    rho: float      # global tilt, in (-1, 1)
    eta: float      # phi scale, > 0
    gamma: float    # phi decay, in (0, 0.5]

    def phi(self, theta: float | np.ndarray) -> float | np.ndarray:
        theta = np.asarray(theta, dtype=float)
        return self.eta / (theta ** self.gamma * (1 + theta) ** (1 - self.gamma))

    def total_var(self, k: np.ndarray, theta: float | np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        phi = self.phi(theta)
        z = phi * k + self.rho
        return (theta / 2) * (
            1 + self.rho * phi * k + np.sqrt(z ** 2 + 1 - self.rho ** 2)
        )

    def no_butterfly_arbitrage(self, theta: float) -> bool:
        """Gatheral-Jacquier sufficient conditions for a single slice."""
        phi = float(self.phi(theta))
        return bool(
            theta * phi * (1 + abs(self.rho)) < 4
            and theta * phi ** 2 * (1 + abs(self.rho)) <= 4
        )


def calibrate_ssvi(k_list: list[np.ndarray], w_list: list[np.ndarray],
                   theta_list: list[float],
                   weights_list: list[np.ndarray] | None = None) -> SSVIParams:
    """Fit ``(rho, eta, gamma)`` jointly across every slice.

    Differential evolution rather than a local method: the SSVI objective has
    well-separated local minima in ``(rho, eta)`` and a gradient method seeded
    badly will happily return an inverted skew. Slices that would violate the
    butterfly condition are rejected inside the objective, so the returned
    parameters are arbitrage-free by construction rather than by inspection.
    """
    n = len(k_list)
    if weights_list is None:
        weights_list = [np.ones_like(k) for k in k_list]

    def objective(x):
        rho, eta, gamma = np.tanh(x[0]), np.exp(x[1]), 0.5 / (1 + np.exp(-x[2]))
        p = SSVIParams(rho=rho, eta=eta, gamma=gamma)
        total = 0.0
        for i in range(n):
            theta = theta_list[i]
            if not p.no_butterfly_arbitrage(theta):
                return 1e10
            w_model = p.total_var(k_list[i], theta)
            if np.any(w_model <= 0):
                return 1e10
            total += float(np.sum(weights_list[i] * (w_model - w_list[i]) ** 2))
        return total

    res = differential_evolution(
        objective, [(-3.0, 3.0), (-4.0, 2.0), (-5.0, 5.0)],
        seed=0, maxiter=500, tol=1e-10, popsize=10,
    )
    return SSVIParams(
        rho=float(np.tanh(res.x[0])),
        eta=float(np.exp(res.x[1])),
        gamma=float(0.5 / (1 + np.exp(-res.x[2]))),
    )


# -- surface wrappers ---------------------------------------------------------

class SVISliceSurface(VolSurface):
    """Per-expiry raw SVI, linearly interpolated in total variance across T.

    Deliberately the naive desk construction: fit each smile as well as
    possible, then join the slices. Nothing enforces consistency between them,
    which is precisely why `volsurface.diagnostics.scan_arbitrage` finds calendar
    violations in the gaps between listed expiries.
    """

    name = "SVI (per-slice)"

    def __init__(self, params: dict[float, SVIParams]):
        if not params:
            raise ValueError("SVISliceSurface needs at least one calibrated slice")
        self.params = dict(sorted(params.items()))
        self.T_nodes = np.array(list(self.params.keys()))
        self._slices = list(self.params.values())

    @classmethod
    def fit(cls, snapshot) -> "SVISliceSurface":
        out = {}
        for T in snapshot.maturities:
            sl = snapshot.slice_at(T)
            if len(sl) < 5:                     # 5 free parameters
                continue
            out[float(T)] = calibrate_svi(sl.k, sl.total_variance, sl.weight)
        return cls(out)

    def total_variance(self, k: np.ndarray, T: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        T = np.broadcast_to(np.asarray(T, dtype=float), k.shape)

        # Total variance from every slice, then interpolate along the T axis.
        w_nodes = np.stack([p.total_var(k) for p in self._slices], axis=0)
        w_nodes = np.maximum(w_nodes, 1e-12)

        if len(self.T_nodes) == 1:
            # Single slice: scale by T so w -> 0 at expiry rather than staying flat.
            return w_nodes[0] * (T / self.T_nodes[0])

        idx = np.clip(np.searchsorted(self.T_nodes, T) - 1, 0, len(self.T_nodes) - 2)
        T0, T1 = self.T_nodes[idx], self.T_nodes[idx + 1]
        frac = (T - T0) / (T1 - T0)             # extrapolates linearly outside
        rows = np.arange(k.size)
        w0 = w_nodes[idx, rows]
        w1 = w_nodes[idx + 1, rows]
        return np.maximum(w0 + frac * (w1 - w0), 1e-12)


class SSVISurface(VolSurface):
    """Joint SSVI with a monotone ``theta(T)``: calendar-arbitrage-free by design.

    ``theta(T)`` is the ATM total variance term structure. It is forced
    non-decreasing (running maximum of the calibrated nodes) and interpolated
    piecewise-linearly through the origin, so ``dw/dT >= 0`` holds everywhere
    rather than only at the nodes. Beyond the last listed expiry it extrapolates
    at the final slope, which preserves monotonicity where a spline would turn
    over. Linear interpolation is also what lets `volsurface.neural.prior.TorchSSVIPrior`
    reproduce this surface exactly in torch -- a smoother scheme would leave the
    neural model correcting a prior subtly different from the one reported here.
    """

    name = "SSVI (joint)"

    def __init__(self, params: SSVIParams, T_nodes: np.ndarray, theta_nodes: np.ndarray):
        self.params = params
        self.T_nodes = np.asarray(T_nodes, dtype=float)
        # Enforce the monotonicity the no-calendar-arbitrage proof requires.
        self.theta_nodes = np.maximum.accumulate(np.asarray(theta_nodes, dtype=float))

        self._T_aug = np.concatenate([[0.0], self.T_nodes])
        self._th_aug = np.concatenate([[0.0], self.theta_nodes])
        self._T_max = float(self.T_nodes[-1])
        self._theta_max = float(self.theta_nodes[-1])
        self._slope_end = self._theta_max / self._T_max if self._T_max > 0 else 0.0
        if len(self.T_nodes) >= 2:
            dT = self.T_nodes[-1] - self.T_nodes[-2]
            if dT > 0:
                self._slope_end = max(
                    (self.theta_nodes[-1] - self.theta_nodes[-2]) / dT, 0.0
                )

    @classmethod
    def fit(cls, snapshot) -> "SSVISurface":
        T_nodes, k_list, w_list, wt_list, theta_list = [], [], [], [], []
        for T in snapshot.maturities:
            sl = snapshot.slice_at(T)
            if len(sl) < 3:
                continue
            order = np.argsort(sl.k)
            k, w = sl.k[order], sl.total_variance[order]
            T_nodes.append(float(T))
            k_list.append(k)
            w_list.append(w)
            wt_list.append(sl.weight[order])
            theta_list.append(max(float(np.interp(0.0, k, w)), 1e-6))

        if len(T_nodes) < 2:
            raise ValueError("SSVI needs at least two usable expiries")

        params = calibrate_ssvi(k_list, w_list, theta_list, wt_list)
        return cls(params, np.array(T_nodes), np.array(theta_list))

    def theta(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        inside = np.interp(np.clip(T, 0.0, self._T_max), self._T_aug, self._th_aug)
        beyond = self._theta_max + self._slope_end * (T - self._T_max)
        return np.maximum(np.where(T <= self._T_max, inside, beyond), 1e-10)

    def total_variance(self, k: np.ndarray, T: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        T = np.broadcast_to(np.asarray(T, dtype=float), k.shape)
        return np.maximum(self.params.total_var(k, self.theta(T)), 1e-12)
