"""
nn/
The neural implied-volatility surface: model, arbitrage penalties, training,
evaluation.

    from marketdata import fetch_chain
    from nn import train_surface, compare, comparison_table

    chain = fetch_chain("SPY")
    result = train_surface(chain)
    print(comparison_table(compare(chain, result)))
"""

from .arbitrage import CollocationRegion, PenaltyWeights, arbitrage_penalties
from .dataset import stratified_split, to_tensors
from .evaluate import compare, comparison_table, evaluate_surface, plot_arbitrage_map, plot_fit
from .model import ModelConfig, NeuralVolSurface
from .prior import FlatPrior, TorchSSVIPrior
from .train import TrainConfig, TrainResult, train_surface

__all__ = [
    "NeuralVolSurface", "ModelConfig",
    "TorchSSVIPrior", "FlatPrior",
    "PenaltyWeights", "CollocationRegion", "arbitrage_penalties",
    "to_tensors", "stratified_split",
    "TrainConfig", "TrainResult", "train_surface",
    "evaluate_surface", "compare", "comparison_table", "plot_fit", "plot_arbitrage_map",
]
