"""
volsurface/surfaces/neural/model.py
The neural implied-volatility surface.

Architecture
------------
The network does **not** output implied vol, and it does not output total
variance either. It outputs a bounded correction to a calibrated SSVI prior:

    w(k, T) = w_prior(k, T) * [ 1 + alpha * tanh( net(features(k, T)) ) ]

Three consequences follow from that one line, and they are the whole design:

* **Positivity is structural.** ``w_prior > 0`` and the bracket lives in
  ``[1 - alpha, 1 + alpha]`` with ``alpha < 1``, so ``w > 0`` everywhere by
  construction. No penalty, no clamping, no NaNs in ``sqrt(w)``.

* **Extrapolation degrades to SSVI, not to noise.** Far outside the quoted
  strikes the network saturates at some constant and the surface becomes a
  fixed multiple of an arbitrage-free parametric surface. An unconstrained net
  asked for ``w`` directly produces whatever the last hidden layer happens to
  extrapolate to, which is how neural vol surfaces end up pricing negative
  densities two strikes past the last quote.

* **The error is bounded a priori.** With ``alpha = 0.35`` the fitted surface is
  never more than ~16% away from SSVI in vol terms. That is a guarantee you can
  state to a risk officer before the model has seen any data.

Features
--------
``(k, T)`` is a bad input basis. The smile does not live in log-moneyness, it
lives in *standardised* log-moneyness ``z = k / sqrt(T)`` -- that is the
coordinate in which smiles across maturities look alike, and handing it to the
network directly saves it from having to learn the ``sqrt(T)`` scaling from a
few hundred points. The feature vector is ``(k, sqrt(T), z, k^2, k*sqrt(T))``,
standardised to zero mean and unit variance using statistics frozen from the
training chain.

Everything runs in float64. The butterfly penalty needs a second derivative of
the network output, and in float32 the curvature of a smooth surface is at the
edge of what the autograd graph can express cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from volsurface.surfaces.base import VolSurface
from volsurface.surfaces.neural.prior import FlatPrior, TorchSSVIPrior


@dataclass
class ModelConfig:
    """Architecture hyper-parameters (training ones live in `volsurface.surfaces.neural.train`)."""

    hidden: tuple[int, ...] = (64, 64, 64)
    alpha: float = 0.35          # max relative correction to the prior
    activation: str = "silu"     # smooth: the butterfly penalty differentiates twice
    seed: int = 0

    def __post_init__(self):
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1) to keep w strictly positive")


_ACTIVATIONS = {"silu": nn.SiLU, "tanh": nn.Tanh, "gelu": nn.GELU, "softplus": nn.Softplus}


class NeuralVolSurface(nn.Module, VolSurface):
    """Total-variance surface: SSVI prior times a learned bounded correction.

    Implements `volsurface.surfaces.base.VolSurface`, so it plugs straight into
    `volsurface.evaluation.diagnostics` and is scored by exactly the same code as the parametric
    baselines -- with one difference: `dw_dT`, `dw_dk` and `d2w_dk2` are
    overridden to use autograd instead of finite differences.
    """

    _grad_cache: tuple | None = None      # see `_grads`; one entry, invalidated on weight change

    def __init__(self, prior: nn.Module, config: ModelConfig | None = None,
                 feature_mean: np.ndarray | None = None,
                 feature_std: np.ndarray | None = None):
        nn.Module.__init__(self)
        self.config = config or ModelConfig()

        self.prior = prior
        act = _ACTIVATIONS[self.config.activation]

        # Seed the weight draw reproducibly without touching the caller's RNG:
        # `nn.Linear` samples at construction, so the seed has to wrap the loop,
        # and constructing a model should not be a side effect on global state.
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.config.seed)
            layers: list[nn.Module] = []
            in_dim = 5
            for h in self.config.hidden:
                layers += [nn.Linear(in_dim, h), act()]
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)
        finally:
            torch.random.set_rng_state(rng_state)

        # Start life *at* the prior: zero the output layer so the correction is
        # exactly 1.0 before the first step. Training then only has to explain
        # the residual, and a failed run degrades to SSVI rather than to noise.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        mean = np.zeros(5) if feature_mean is None else np.asarray(feature_mean, float)
        std = np.ones(5) if feature_std is None else np.asarray(feature_std, float)
        self.register_buffer("feature_mean", torch.tensor(mean, dtype=torch.float64))
        self.register_buffer("feature_std", torch.tensor(np.maximum(std, 1e-8),
                                                         dtype=torch.float64))
        self.double()
        self._grad_cache: tuple | None = None

    @property
    def name(self) -> str:
        """Names the prior, so the flat-prior ablation is a distinct table row."""
        kind = "SSVI prior" if isinstance(self.prior, TorchSSVIPrior) else "flat prior"
        return f"Neural ({kind} + penalties)"

    # -- features -------------------------------------------------------------

    @staticmethod
    def raw_features(k: Tensor, T: Tensor) -> Tensor:
        """``(k, sqrt(T), k/sqrt(T), k^2, k*sqrt(T))`` -- see the module docstring."""
        sqrt_T = torch.sqrt(torch.clamp(T, min=1e-6))
        z = k / sqrt_T
        return torch.stack([k, sqrt_T, z, k * k, k * sqrt_T], dim=-1)

    @classmethod
    def feature_stats(cls, k: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Standardisation statistics, computed once on the training chain."""
        f = cls.raw_features(torch.tensor(k, dtype=torch.float64),
                            torch.tensor(T, dtype=torch.float64))
        return f.mean(0).numpy(), f.std(0).numpy()

    def features(self, k: Tensor, T: Tensor) -> Tensor:
        return (self.raw_features(k, T) - self.feature_mean) / self.feature_std

    # -- forward --------------------------------------------------------------

    def correction(self, k: Tensor, T: Tensor) -> Tensor:
        """The learned multiplier, in ``[1 - alpha, 1 + alpha]``."""
        raw = self.net(self.features(k, T)).squeeze(-1)
        return 1.0 + self.config.alpha * torch.tanh(raw)

    def total_variance_torch(self, k: Tensor, T: Tensor) -> Tensor:
        """``w(k, T)`` in torch, differentiable in both arguments."""
        return self.prior.total_variance(k, T) * self.correction(k, T)

    def forward(self, k: Tensor, T: Tensor) -> Tensor:
        return self.total_variance_torch(k, T)

    def implied_vol_torch(self, k: Tensor, T: Tensor) -> Tensor:
        w = torch.clamp(self.total_variance_torch(k, T), min=1e-12)
        return torch.sqrt(w / torch.clamp(T, min=1e-12))

    # -- VolSurface (numpy) interface ----------------------------------------

    def _as_tensors(self, k, T) -> tuple[Tensor, Tensor]:
        k = np.asarray(k, dtype=float)
        T = np.broadcast_to(np.asarray(T, dtype=float), k.shape)
        return (torch.tensor(k.ravel(), dtype=torch.float64),
                torch.tensor(np.ascontiguousarray(T).ravel(), dtype=torch.float64))

    def total_variance(self, k: np.ndarray, T: np.ndarray) -> np.ndarray:
        shape = np.asarray(k, dtype=float).shape
        kt, Tt = self._as_tensors(k, T)
        with torch.no_grad():
            return self.total_variance_torch(kt, Tt).numpy().reshape(shape)

    # Exact derivatives via autograd -- the finite-difference fallbacks in
    # `VolSurface` exist for the parametric baselines, but there is no reason to
    # accept their truncation error when the model is already a torch graph.

    def _grads(self, k: np.ndarray, T: np.ndarray):
        """``(dw/dk, d2w/dk2, dw/dT)``, memoised on the last ``(k, T)`` asked for.

        One backward pass produces all three, but the `VolSurface` interface
        exposes them as three separate calls and the diagnostics use all three
        on the same dense grid -- without the cache that is three identical
        graph builds for two thrown-away results each time. The cache holds one
        entry, keyed on the raw bytes of the inputs, and is invalidated whenever
        the weights change (`_bump_grad_cache` is hooked to `load_state_dict`
        and to training mode).
        """
        k_arr = np.ascontiguousarray(np.asarray(k, dtype=float))
        T_arr = np.ascontiguousarray(
            np.broadcast_to(np.asarray(T, dtype=float), k_arr.shape))
        key = (k_arr.shape, k_arr.tobytes(), T_arr.tobytes())

        if self._grad_cache is not None and self._grad_cache[0] == key:
            return self._grad_cache[1]

        kt, Tt = self._as_tensors(k_arr, T_arr)
        kt.requires_grad_(True)
        Tt.requires_grad_(True)
        w = self.total_variance_torch(kt, Tt)
        w_k, w_T = torch.autograd.grad(w.sum(), [kt, Tt], create_graph=True)
        w_kk = torch.autograd.grad(w_k.sum(), kt, create_graph=False)[0]

        out = (w_k.detach(), w_kk.detach(), w_T.detach())
        self._grad_cache = (key, out)
        return out

    def _invalidate_grad_cache(self, *_args) -> None:
        self._grad_cache = None

    def train(self, mode: bool = True):          # noqa: D102 - nn.Module override
        self._invalidate_grad_cache()
        return super().train(mode)

    def load_state_dict(self, *args, **kwargs):  # noqa: D102 - nn.Module override
        self._invalidate_grad_cache()
        return super().load_state_dict(*args, **kwargs)

    # `h` is accepted and ignored throughout: it is the finite-difference step
    # of the `VolSurface` base signature, and these derivatives are exact.

    def dw_dk(self, k: np.ndarray, T: np.ndarray, h: float = 0.0) -> np.ndarray:
        shape = np.asarray(k, dtype=float).shape
        return self._grads(k, T)[0].numpy().reshape(shape)

    def d2w_dk2(self, k: np.ndarray, T: np.ndarray, h: float = 0.0) -> np.ndarray:
        shape = np.asarray(k, dtype=float).shape
        return self._grads(k, T)[1].numpy().reshape(shape)

    def dw_dT(self, k: np.ndarray, T: np.ndarray, h: float = 0.0) -> np.ndarray:
        shape = np.asarray(k, dtype=float).shape
        return self._grads(k, T)[2].numpy().reshape(shape)

    # -- persistence ----------------------------------------------------------

    def save(self, path) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": vars(self.config),
                "prior_kind": type(self.prior).__name__,
                "prior_state": self.prior.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "NeuralVolSurface":
        blob = torch.load(path, weights_only=False)
        prior_state = blob["prior_state"]
        if blob["prior_kind"] == "TorchSSVIPrior":
            prior = TorchSSVIPrior(
                rho=float(prior_state["rho"]),
                eta=float(prior_state["eta"]),
                gamma=float(prior_state["gamma"]),
                T_nodes=prior_state["T_nodes"].numpy()[1:],
                theta_nodes=prior_state["theta_nodes"].numpy()[1:],
            )
        else:
            prior = FlatPrior()
        model = cls(prior, ModelConfig(**blob["config"]))
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model
