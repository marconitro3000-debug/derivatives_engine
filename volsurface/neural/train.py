"""
volsurface/neural/train.py
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

**Quotes are vega-weighted.** `volsurface.chain` attaches a weight of roughly
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
means to a usable surface, not part of the thing being measured. Its patience
counter starts when the warm-up ends, not at epoch zero -- model selection does
not begin until then, so counting earlier would silently spend most of the
budget before there was a candidate to be patient about.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from volsurface.svi import SSVISurface
from volsurface.neural.arbitrage import CollocationRegion, PenaltyWeights, arbitrage_penalties
from volsurface.neural.dataset import SurfaceTensors, stratified_split, to_tensors
from volsurface.neural.model import ModelConfig, NeuralVolSurface
from volsurface.neural.prior import FlatPrior, TorchSSVIPrior


@dataclass
class TrainConfig:
    epochs: int = 2000
    lr: float = 3e-3
    weight_decay: float = 1e-6
    warmup_epochs: int = 300      # linear ramp-in of the arbitrage penalties
    patience: int = 400           # early stopping on validation vol RMSE
    val_fraction: float = 0.2
    seed: int = 0
    split_seed: int | None = None
    """Seed for the train/validation split, if it should differ from `seed`.

    Defaults to `seed`, which is what you want for a single run. Separating them
    matters the moment you compare two configurations across several seeds: with
    one seed driving both, changing it changes *which quotes are held out*, so
    the validation numbers of two arms are not measured on the same thing and
    the scatter between seeds swamps the effect being looked for. Pin this and
    vary `seed`, and the only thing moving is the model."""

    log_every: int = 0            # 0 = silent; N = print every N epochs
    feasible_tol: float = 1e-8
    """How negative a violation may be on an epoch's collocation draw before that
    epoch is disqualified from being selected as the best model. The penalties
    are soft, so a run can converge to a low validation error while leaving a
    shallow butterfly dent in a corner of the extrapolation region; selecting on
    error alone then ships that dent. Selecting the best *feasible* epoch is what
    a constrained problem asks for, and it costs nothing -- the violations are
    already computed every step."""

    penalties: PenaltyWeights = field(default_factory=PenaltyWeights)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        # Model selection only starts after the warm-up, so a short run with the
        # default 300-epoch ramp would finish having never recorded a candidate
        # and report epoch 0 as its best. Cap the ramp at a fifth of the budget;
        # at the default 1500 epochs this is exactly the 300 it already was.
        self.warmup_epochs = min(self.warmup_epochs, max(self.epochs // 5, 1))


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
    best_is_feasible: bool = True
    """False when no epoch after the warm-up was clean on its own collocation
    draw and the run had to fall back to selecting on validation error alone.
    A run that reports this is telling you the penalties never fully bit."""

    def summary(self) -> str:
        h = self.history
        return (
            f"trained {len(h['train_rmse_bps'])} epochs in {self.elapsed_sec:.1f}s "
            f"(best epoch {self.best_epoch})\n"
            f"  train IV RMSE {h['train_rmse_bps'][self.best_epoch]:.1f}bp  "
            f"val IV RMSE {self.best_val_rmse_bps:.1f}bp  "
            f"(n_train={self.n_train}, n_val={self.n_val})\n"
            f"  selected epoch worst dw/dT {h['worst_calendar'][self.best_epoch]:+.2e}  "
            f"worst g {h['worst_butterfly'][self.best_epoch]:+.2e}"
            + ("" if self.best_is_feasible else
               "\n  !  no arbitrage-free epoch after the warm-up;"
               " selected on error alone")
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
    snapshot : `volsurface.chain.ChainSnapshot`
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
    split_seed = cfg.seed if cfg.split_seed is None else cfg.split_seed
    train_idx, val_idx = stratified_split(snapshot, cfg.val_fraction, split_seed)
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
                        "frac_calendar", "frac_butterfly", "feasible")
    }
    # Two candidates are tracked: the best *feasible* epoch, which is the one
    # that gets shipped, and the best epoch ignoring feasibility, which is only
    # a fallback for a run where the penalties never fully bit.
    best_val, best_epoch, best_state = float("inf"), 0, None
    fb_val, fb_epoch, fb_state = float("inf"), 0, None
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

        # -- diagnostics, on the weights that produced `fit` and `pen` ---------
        # Measured *before* the step, not after. Otherwise the epoch's recorded
        # violations belong to one weight vector and its recorded errors to the
        # next one, and a "best epoch" selected on feasibility would be selecting
        # on the feasibility of a model it is not saving.
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

        feasible = (float(pen["worst_calendar"]) >= -cfg.feasible_tol
                    and float(pen["worst_butterfly"]) >= -cfg.feasible_tol)
        history["feasible"].append(float(feasible))

        # Only start selecting a best model once the penalties are at full
        # strength; before that the "best" epoch is usually an under-constrained
        # surface that happens to fit well.
        improved = False
        if epoch >= cfg.warmup_epochs:
            if feasible and val_rmse < best_val - 1e-6:
                best_val, best_epoch = val_rmse, epoch
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                improved = True
            if val_rmse < fb_val - 1e-6:
                fb_val, fb_epoch = val_rmse, epoch
                fb_state = {k: v.detach().clone()
                            for k, v in model.state_dict().items()}
                # Progress on the fallback still counts as progress while no
                # feasible epoch has been seen; otherwise a run that is slowly
                # working its way into the feasible region gets cut off.
                improved = improved or best_state is None
        # The counter only means anything once model selection has started;
        # letting it run through the warm-up used to eat `warmup_epochs` of the
        # patience budget before the first candidate was even recorded.
        epochs_since_best = (0 if (improved or epoch < cfg.warmup_epochs)
                             else epochs_since_best + 1)

        # The step comes last: everything recorded above describes the weights
        # that are still in `best_state` if this epoch won.
        opt.step()
        sched.step()

        if cfg.log_every and epoch % cfg.log_every == 0:
            print(
                f"  epoch {epoch:5d}  loss={float(loss.detach()):.3e}  "
                f"train={train_rmse:7.1f}bp  val={val_rmse:7.1f}bp  "
                f"worst dw/dT={float(pen['worst_calendar']):+.2e}  "
                f"worst g={float(pen['worst_butterfly']):+.2e}"
            )

        if epochs_since_best >= cfg.patience and epoch >= cfg.warmup_epochs:
            break

    best_is_feasible = best_state is not None
    if not best_is_feasible:
        best_state, best_val, best_epoch = fb_state, fb_val, fb_epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return TrainResult(
        model=model,
        prior_surface=prior_surface,
        history=history,
        best_epoch=best_epoch,
        best_val_rmse_bps=best_val if np.isfinite(best_val) else history["val_rmse_bps"][-1],
        best_is_feasible=best_is_feasible,
        elapsed_sec=time.time() - t0,
        n_train=len(train_idx),
        n_val=len(val_idx),
    )
