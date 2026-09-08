"""
volsurface.surfaces.neural
=================

The network, its prior, its constraints, and the loop that fits them.

    prior      SSVI in torch, differentiable end to end -- the shape the network
               corrects rather than replaces
    model      w(k,T) = w_prior(k,T) * (1 + alpha*tanh(net(k,T)))
    arbitrage  calendar and butterfly penalties, differentiated by autograd and
               imposed on collocation points beyond the quoted strikes
    dataset    tensors and the stratified split
    train      vega-weighted objective, penalty warm-up, early stopping
"""

from .arbitrage import (
    CollocationRegion,
    PenaltyWeights,
    arbitrage_penalties,
    durrleman_g,
    surface_derivatives,
)
from .dataset import SurfaceTensors, stratified_split, to_tensors
from .model import ModelConfig, NeuralVolSurface
from .prior import FlatPrior, TorchSSVIPrior
from .train import TrainConfig, TrainResult, train_surface

__all__ = [
    "NeuralVolSurface", "ModelConfig",
    "TorchSSVIPrior", "FlatPrior",
    "PenaltyWeights", "CollocationRegion", "arbitrage_penalties",
    "durrleman_g", "surface_derivatives",
    "SurfaceTensors", "to_tensors", "stratified_split",
    "TrainConfig", "TrainResult", "train_surface",
]
