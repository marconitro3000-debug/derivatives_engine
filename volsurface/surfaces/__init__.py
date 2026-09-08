"""
volsurface.surfaces
===================

The models. One interface, three implementations, and the comparison between
them is the point of the project.

    base      the `VolSurface` interface -- everything is total variance
              ``w(k, T) = sigma(k, T)^2 * T``, and implementing
              `total_variance` is enough to get IVs, prices and the whole
              diagnostic suite for free
    svi       the parametric baselines: raw SVI per expiry (quasi-explicit
              calibration) and joint SSVI. SSVI is also the network's prior
    neural    **the network.** An MLP that learns a bounded multiplicative
              correction to the SSVI prior, trained under no-arbitrage
              penalties imposed on collocation points that extend past the
              quoted strikes

The tension the project exists to resolve: per-slice SVI tracks the smiles but
nothing connects the slices, so interpolating between them produces calendar
arbitrage; joint SSVI makes that impossible by construction but cannot follow a
chain whose smile changes character across the term structure. `neural` closes
the gap -- it fits like the flexible model and behaves like the safe one.

Because all three implement `base.VolSurface`, they are scored by identical code
(`volsurface.evaluation`). Without that, the comparison would not be worth
reading.
"""

from .base import VolSurface
from .neural import (
    ModelConfig,
    NeuralVolSurface,
    PenaltyWeights,
    TrainConfig,
    TrainResult,
    train_surface,
)
from .svi import SSVIParams, SSVISurface, SVIParams, SVISliceSurface

__all__ = [
    "ModelConfig",
    "NeuralVolSurface",
    "PenaltyWeights",
    "SSVIParams",
    "SSVISurface",
    "SVIParams",
    "SVISliceSurface",
    "TrainConfig",
    "TrainResult",
    "VolSurface",
    "train_surface",
]
