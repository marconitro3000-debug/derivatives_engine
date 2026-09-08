"""
volsurface/surfaces/neural/arbitrage.py
Soft no-arbitrage constraints, differentiated exactly.

The two static conditions from `volsurface.evaluation.diagnostics` are turned into penalties the
optimiser can see:

    calendar   dw/dT      >= 0
    butterfly  g(k, T)    >= 0,
               g = (1 - k w_k / 2w)^2 - (w_k^2 / 4)(1/4 + 1/w) + w_kk / 2

Each becomes ``mean(relu(-violation)^2)``: zero wherever the condition holds, so
a compliant surface pays nothing, and quadratic in the depth of the breach, so a
gradient exists the moment one appears. A hinge (relu without the square) would
push with constant force regardless of severity and tends to oscillate around
the constraint boundary.

**Where the penalties are evaluated matters more than their weight.** They are
imposed on random *collocation points* drawn over a region deliberately wider
than the quoted data -- typically 30% beyond the extreme strikes and past the
last listed expiry. Constraining the surface only where quotes exist is nearly
free and nearly useless: the wings and the gaps between expiries are exactly
where an interpolating network invents negative densities. Resampling the
collocation points every epoch, rather than fixing a grid, keeps the network
from learning to satisfy the constraint at a finite set of locations while
violating it between them.

The derivatives come from `torch.autograd.grad` with ``create_graph=True``, so
the penalty is differentiated a second time during backpropagation. This is the
reason the model runs in float64 and uses a smooth activation: a ReLU network
has zero second derivative almost everywhere and the butterfly penalty would be
blind to curvature.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PenaltyWeights:
    """Lagrange-style weights on the two static-arbitrage conditions.

    Defaults are set so that, on a typical equity chain, the penalty terms are
    one to two orders of magnitude below the fit term at convergence: large
    enough to eliminate violations, small enough not to distort the fit where no
    violation is at stake.
    """

    calendar: float = 10.0
    butterfly: float = 1.0

    # Collocation region, as a multiple of the data's own range.
    k_margin: float = 0.3        # extend log-moneyness 30% past the extreme quotes
    T_lo_factor: float = 0.5     # sample down to half the shortest expiry
    T_hi_factor: float = 1.3     # and out to 1.3x the longest
    n_points: int = 2048


@dataclass
class CollocationRegion:
    """The (k, T) box the constraints are enforced on."""

    k_lo: float
    k_hi: float
    T_lo: float
    T_hi: float

    @classmethod
    def around(cls, k, T, weights: PenaltyWeights) -> "CollocationRegion":
        k_lo, k_hi = float(k.min()), float(k.max())
        span = max(k_hi - k_lo, 1e-3)
        T_lo, T_hi = float(T.min()), float(T.max())
        return cls(
            k_lo=k_lo - weights.k_margin * span,
            k_hi=k_hi + weights.k_margin * span,
            T_lo=max(T_lo * weights.T_lo_factor, 1e-3),
            T_hi=T_hi * weights.T_hi_factor,
        )

    def sample(self, n: int, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """Fresh draws over the box: uniform in ``k``, uniform in ``sqrt(T)``.

        Uniform in ``T`` is the wrong measure. On an eight-expiry chain running
        out to two years, a uniform draw puts about one point in a hundred below
        the shortest listed expiry -- and the short end, where total variance is
        smallest and the smile most curved, is precisely where the density goes
        negative first. Drawing uniformly in ``sqrt(T)`` gives a density
        proportional to ``1/sqrt(T)``, which concentrates the constraint where
        the surface actually bends, at no extra cost. It also matches how the
        expiries themselves are sampled in `volsurface.data`.
        """
        u = torch.rand(n, 2, dtype=torch.float64, generator=generator)
        k = self.k_lo + u[:, 0] * (self.k_hi - self.k_lo)
        root_lo, root_hi = self.T_lo ** 0.5, self.T_hi ** 0.5
        T = (root_lo + u[:, 1] * (root_hi - root_lo)) ** 2
        return k, T


# -- derivatives --------------------------------------------------------------

def surface_derivatives(model, k: Tensor, T: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``(w, dw/dk, d2w/dk2, dw/dT)`` with the autograd graph retained.

    `k` and `T` are cloned before `requires_grad_` so the caller's tensors are
    never mutated -- a subtle trap when the same collocation tensors are reused.
    """
    k = k.clone().requires_grad_(True)
    T = T.clone().requires_grad_(True)

    w = model.total_variance_torch(k, T)
    w_k, w_T = torch.autograd.grad(w.sum(), [k, T], create_graph=True)
    w_kk = torch.autograd.grad(w_k.sum(), k, create_graph=True)[0]
    return w, w_k, w_kk, w_T


def durrleman_g(w: Tensor, w_k: Tensor, w_kk: Tensor, k: Tensor) -> Tensor:
    """Durrleman's function; ``g >= 0`` iff the risk-neutral density is non-negative."""
    w_safe = torch.clamp(w, min=1e-10)
    return (
        (1.0 - k * w_k / (2.0 * w_safe)) ** 2
        - (w_k ** 2 / 4.0) * (0.25 + 1.0 / w_safe)
        + w_kk / 2.0
    )


# -- penalties ----------------------------------------------------------------

def arbitrage_penalties(model, k: Tensor, T: Tensor) -> dict[str, Tensor]:
    """Calendar and butterfly penalties at the given collocation points.

    Returns the two penalty terms plus the raw worst-case violations, which are
    what the training log reports: a penalty value of 3e-7 means nothing to a
    reader, while "worst dw/dT = -0.0002" is a number that can be judged.
    """
    w, w_k, w_kk, w_T = surface_derivatives(model, k, T)
    g = durrleman_g(w, w_k, w_kk, k)

    cal_violation = torch.relu(-w_T)
    bf_violation = torch.relu(-g)

    return {
        "calendar": (cal_violation ** 2).mean(),
        "butterfly": (bf_violation ** 2).mean(),
        "worst_calendar": w_T.detach().min(),
        "worst_butterfly": g.detach().min(),
        "calendar_frac": (cal_violation.detach() > 0).double().mean(),
        "butterfly_frac": (bf_violation.detach() > 0).double().mean(),
    }
