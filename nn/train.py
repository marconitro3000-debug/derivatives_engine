"""
nn/train.py
Fitting the neural surface.

Objective
---------
    L  =  sum_i weight_i * (sigma_model(k_i, T_i) - sigma_market_i)^2
       +  lambda_cal * mean( relu(-dw/dT)^2 )        on collocation points
       +  lambda_bfly * mean( relu(-g)^2 )           on collocation points

Two choices in the first term are worth defending.

**The fit is in vol space, not in total-variance space.** Total variance is the
right coordinate for the constraints, but fitting it minimises a quantity whose
scale grows with maturity, which silently reweights the one-year smile ten times
harder than the one-month. Vol-space errors are what a trader quotes and what a
model is judged on.

**Quotes are vega-weighted.** `marketdata.chain` attaches a weight of roughly
vega / spread to every quote. Unweighted least squares in vol space chases deep
wing options whose vol is barely identified by their price, at the expense of
the at-the-money region where the money is.

Optimisation
------------
Full-batch Adam with a cosine schedule. A chain is a few hundred to a few
thousand points; mini-batching would only add gradient noise, and the arbitrage
penalty is a property of the *surface* rather than of any batch, so it is
evaluated on freshly drawn collocation points at every step regardless.

The penalties are ramped in linearly over the first `warmup_epochs`. Starting at
full weight makes the network satisfy the constraints trivially by sitting on
the prior, which it can do from step zero (the output layer is zero-initialised)
and from which the fit term then struggles to move it.

Early stopping tracks the *unpenalised* validation error: the penalties are a
means to a usable surface, not part of the thing being measured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from baselines.svi import SSVISurface
from nn.arbitrage import CollocationRegion, PenaltyWeights, arbitrage_penalties
from nn.dataset import SurfaceTensors, stratified_split, to_tensors
from nn.model import ModelConfig, NeuralVolSurface
from nn.prior import FlatPrior, TorchSSVIPrior


@dataclass
class TrainConfig:
    epochs: int = 2000
    lr: float = 3e-3
    weight_decay: float = 1e-6
    warmup_epochs: int = 300      # linear ramp-in of the arbitrage penalties
    patience: int = 400           # early stopping on validation vol RMSE
    val_fraction: float = 0.2
    seed: int = 0
    log_every: int = 0            # 0 = silent; N = print every N epochs
    penalties: PenaltyWeights = field(default_factory=PenaltyWeights)
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class TrainResult:
    """Everything needed to report on a run without re-running it."""

    model: NeuralVolSurface
    prior_surface: SSVISurface | None
    history: dict[str, list[float]]
    best_epoch: int
    best_val_rmse_bps: float
    elapsed_sec: float
    n_train: int
    n_val: int

    def summary(self) -> str:
        h = self.history
        return (
            f"trained {len(h['train_rmse_bps'])} epochs in {self.elapsed_sec:.1f}s "
            f"(best epoch {self.best_epoch})\n"
            f"  train IV RMSE {h['train_rmse_bps'][self.best_epoch]:.1f}bp  "
            f"val IV RMSE {self.best_val_rmse_bps:.1f}bp  "
            f"(n_train={self.n_train}, n_val={self.n_val})\n"
            f"  final worst dw/dT {h['worst_calendar'][-1]:+.2e}  "
            f"worst g {h['worst_butterfly'][-1]:+.2e}"
        )


def _weighted_rmse_bps(model: NeuralVolSurface, data: SurfaceTensors) -> torch.Tensor:
    """Weighted RMSE of implied vol, in basis points."""
    iv = model.implied_vol_torch(data.k, data.T)
    err = (iv - data.iv) * 10_000.0
    return torch.sqrt((data.weight * err ** 2).sum() / data.weight.sum())


def train_surface(snapshot, config: TrainConfig | None = None,
                  prior: str = "ssvi") -> TrainResult:
    """Fit a `NeuralVolSurface` to a cleaned option chain.

    Parameters
    ----------
    snapshot : `marketdata.chain.ChainSnapshot`
    config   : training hyper-parameters; defaults are tuned for a liquid
               single-name or index chain of a few hundred quotes.
    prior    : ``"ssvi"`` (calibrate SSVI first and learn the correction) or
               ``"flat"`` (constant-vol prior -- the ablation that shows how much
               of the result the parametric prior is responsible for).

    Returns
    -------
    `TrainResult` holding the best model by validation error, the calibrated
    prior, and the full training history.
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    t0 = time.time()

    # -- prior ---------------------------------------------------------------
    prior_surface: SSVISurface | None = None
    if prior == "ssvi":
        prior_surface = SSVISurface.fit(snapshot)
        prior_module = TorchSSVIPrior.from_surface(prior_surface)
    elif prior == "flat":
        prior_module = FlatPrior(sigma=float(np.median(snapshot.iv)))
    else:
        raise ValueError(f"unknown prior {prior!r}; use 'ssvi' or 'flat'")

    # -- data ----------------------------------------------------------------
    train_idx, val_idx = stratified_split(snapshot, cfg.val_fraction, cfg.seed)
    train = to_tensors(snapshot, train_idx)
    val = to_tensors(snapshot, val_idx) if len(val_idx) else None

    mean, std = NeuralVolSurface.feature_stats(snapshot.k[train_idx], snapshot.T[train_idx])
    model = NeuralVolSurface(prior_module, cfg.model, mean, std)

    region = CollocationRegion.around(snapshot.k, snapshot.T, cfg.penalties)
    generator = torch.Generator().manual_seed(cfg.seed)

    opt = torch.optim.Adam(model.net.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history: dict[str, list[float]] = {
        k: [] for k in ("loss", "fit", "pen_calendar", "pen_butterfly",
                        "train_rmse_bps", "val_rmse_bps",
                        "worst_calendar", "worst_butterfly",
                        "frac_calendar", "frac_butterfly")
    }
    best_val, best_epoch, best_state = float("inf"), 0, None
    epochs_since_best = 0

    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()

        iv_model = model.implied_vol_torch(train.k, train.T)
        fit = (train.weight * (iv_model - train.iv) ** 2).sum() / train.weight.sum()

        k_col, T_col = region.sample(cfg.penalties.n_points, generator)
        pen = arbitrage_penalties(model, k_col, T_col)

        ramp = min(1.0, (epoch + 1) / max(cfg.warmup_epochs, 1))
        loss = (
            fit
            + ramp * cfg.penalties.calendar * pen["calendar"]
            + ramp * cfg.penalties.butterfly * pen["butterfly"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 10.0)
        opt.step()
        sched.step()

        # -- diagnostics ------------------------------------------------------
        model.eval()
        with torch.no_grad():
            train_rmse = float(_weighted_rmse_bps(model, train))
            val_rmse = float(_weighted_rmse_bps(model, val)) if val else train_rmse

        history["loss"].append(float(loss.detach()))
        history["fit"].append(float(fit.detach()))
        history["pen_calendar"].append(float(pen["calendar"].detach()))
        history["pen_butterfly"].append(float(pen["butterfly"].detach()))
        history["train_rmse_bps"].append(train_rmse)
        history["val_rmse_bps"].append(val_rmse)
        history["worst_calendar"].append(float(pen["worst_calendar"]))
        history["worst_butterfly"].append(float(pen["worst_butterfly"]))
        history["frac_calendar"].append(float(pen["calendar_frac"]))
        history["frac_butterfly"].append(float(pen["butterfly_frac"]))

        # Only start selecting a best model once the penalties are at full
        # strength; before that the "best" epoch is usually an under-constrained
        # surface that happens to fit well.
        if epoch >= cfg.warmup_epochs and val_rmse < best_val - 1e-6:
            best_val, best_epoch = val_rmse, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if cfg.log_every and epoch % cfg.log_every == 0:
            print(
                f"  epoch {epoch:5d}  loss={float(loss.detach()):.3e}  "
                f"train={train_rmse:7.1f}bp  val={val_rmse:7.1f}bp  "
                f"worst dw/dT={float(pen['worst_calendar']):+.2e}  "
                f"worst g={float(pen['worst_butterfly']):+.2e}"
            )

        if epochs_since_best >= cfg.patience and epoch >= cfg.warmup_epochs:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return TrainResult(
        model=model,
        prior_surface=prior_surface,
        history=history,
        best_epoch=best_epoch,
        best_val_rmse_bps=best_val if np.isfinite(best_val) else history["val_rmse_bps"][-1],
        elapsed_sec=time.time() - t0,
        n_train=len(train_idx),
        n_val=len(val_idx),
    )
