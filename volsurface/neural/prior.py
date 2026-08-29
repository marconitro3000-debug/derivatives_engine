"""
volsurface/neural/prior.py
The SSVI prior, in torch, differentiable end to end.

The neural surface does not learn ``w(k, T)`` from nothing. It learns a bounded
*multiplicative correction* to a calibrated SSVI surface, which means the shape
the network starts from is already arbitrage-free, already has the right
asymptotics in the wings, and already goes to zero at expiry. The network only
has to explain what SSVI's three global parameters cannot.

Reimplementing SSVI here rather than calling the numpy version is what makes the
arbitrage penalties possible: ``dw/dT``, ``dw/dk`` and ``d2w/dk2`` of the *full*
model, prior included, come out of `torch.autograd` exactly, with no finite
differences and no error floor to tune tolerances against.

`TorchSSVIPrior.from_surface` reproduces `volsurface.svi.SSVISurface` exactly --
both use the same monotone piecewise-linear ``theta(T)``, so the prior the
network corrects is the surface the report benchmarks against.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class TorchSSVIPrior(nn.Module):
    """SSVI total variance as a frozen torch module.

    Parameters are registered as buffers, not parameters: the prior is
    calibrated once by `volsurface.svi.calibrate_ssvi` and held fixed while the
    network trains. Letting both move at once makes the correction
    unidentifiable -- the network can always absorb a change in ``rho`` and you
    lose the interpretation of the correction as a residual.
    """

    def __init__(self, rho: float, eta: float, gamma: float,
                 T_nodes: np.ndarray, theta_nodes: np.ndarray):
        super().__init__()
        self.register_buffer("rho", torch.tensor(float(rho), dtype=torch.float64))
        self.register_buffer("eta", torch.tensor(float(eta), dtype=torch.float64))
        self.register_buffer("gamma", torch.tensor(float(gamma), dtype=torch.float64))

        T_nodes = np.asarray(T_nodes, dtype=np.float64)
        theta_nodes = np.maximum.accumulate(np.asarray(theta_nodes, dtype=np.float64))
        # Anchor at the origin: w(k, 0) = 0 for every k.
        self.register_buffer("T_nodes", torch.tensor(np.concatenate([[0.0], T_nodes])))
        self.register_buffer("theta_nodes", torch.tensor(np.concatenate([[0.0], theta_nodes])))

        dT = T_nodes[-1] - T_nodes[-2] if len(T_nodes) >= 2 else T_nodes[-1]
        dtheta = (theta_nodes[-1] - theta_nodes[-2]) if len(theta_nodes) >= 2 else theta_nodes[-1]
        slope_end = max(dtheta / dT, 0.0) if dT > 0 else 0.0
        self.register_buffer("slope_end", torch.tensor(float(slope_end), dtype=torch.float64))

    @classmethod
    def from_surface(cls, surface) -> "TorchSSVIPrior":
        """Build the torch prior from a calibrated `volsurface.svi.SSVISurface`."""
        p = surface.params
        return cls(p.rho, p.eta, p.gamma, surface.T_nodes, surface.theta_nodes)

    # -- theta term structure -------------------------------------------------

    def theta(self, T: Tensor) -> Tensor:
        """Monotone piecewise-linear ATM total variance ``theta(T)``.

        Written with `searchsorted` and explicit linear segments rather than an
        interpolation helper so it stays differentiable in ``T``: the calendar
        penalty needs ``dtheta/dT``, and the piecewise-linear form makes that
        derivative exactly the segment slope.
        """
        Tn, thn = self.T_nodes, self.theta_nodes
        T_max = Tn[-1]

        idx = torch.clamp(torch.searchsorted(Tn, T.detach().contiguous()) - 1,
                          0, len(Tn) - 2)
        T0, T1 = Tn[idx], Tn[idx + 1]
        th0, th1 = thn[idx], thn[idx + 1]
        inside = th0 + (th1 - th0) * (T - T0) / (T1 - T0)

        beyond = thn[-1] + self.slope_end * (T - T_max)
        return torch.clamp(torch.where(T <= T_max, inside, beyond), min=1e-10)

    def phi(self, theta: Tensor) -> Tensor:
        return self.eta / (theta ** self.gamma * (1.0 + theta) ** (1.0 - self.gamma))

    def total_variance(self, k: Tensor, T: Tensor) -> Tensor:
        theta = self.theta(T)
        phi = self.phi(theta)
        z = phi * k + self.rho
        return (theta / 2.0) * (
            1.0 + self.rho * phi * k
            + torch.sqrt(z ** 2 + 1.0 - self.rho ** 2)
        )

    def forward(self, k: Tensor, T: Tensor) -> Tensor:
        return self.total_variance(k, T)


class FlatPrior(nn.Module):
    """Constant-vol prior ``w = sigma^2 T``: the ablation for "does the prior matter?".

    Training against this instead of `TorchSSVIPrior` isolates how much of the
    final fit comes from the parametric shape and how much from the network.
    """

    def __init__(self, sigma: float = 0.20):
        super().__init__()
        self.register_buffer("var", torch.tensor(float(sigma) ** 2, dtype=torch.float64))

    def total_variance(self, k: Tensor, T: Tensor) -> Tensor:
        return self.var * T + 0.0 * k

    def forward(self, k: Tensor, T: Tensor) -> Tensor:
        return self.total_variance(k, T)
