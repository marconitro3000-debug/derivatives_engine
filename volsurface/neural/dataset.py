"""
volsurface/neural/dataset.py
Chain -> tensors, and the train/validation split that actually tests something.

The split is **stratified within each expiry**, not random over the whole chain
and not by expiry. Both alternatives are misleading:

* A plain random split leaves the validation points surrounded by training
  points a few strikes away. Any interpolator scores well; you learn nothing
  about the model.
* Holding out whole expiries tests extrapolation in maturity, which is a
  genuinely different and much harder task, and it destroys the calendar
  structure the model is being trained to respect.

Stratifying within each expiry -- hold out every n-th strike of every smile --
tests interpolation across strike while keeping every maturity represented, and
that is what a surface is used for in practice: pricing a strike that is not
listed.

The held-out strikes are spread over the **whole** smile, wings included. An
earlier version of this file drew them from the interior only, on the argument
that holding out an extreme strike asks the model to extrapolate rather than
interpolate. That argument is true and the consequence was still wrong: the
wings are where the fit error lives, so keeping them permanently in the
training set made the validation RMSE come out *below* the training RMSE and
turned the reported generalisation gap into an artefact of the split. A
validation set that systematically excludes the hard points is not measuring
generalisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class SurfaceTensors:
    """Aligned float64 tensors for one split of the chain."""

    k: Tensor
    T: Tensor
    iv: Tensor
    weight: Tensor
    total_variance: Tensor

    def __len__(self) -> int:
        return len(self.k)

    def to(self, device) -> "SurfaceTensors":
        return SurfaceTensors(*[t.to(device) for t in
                                (self.k, self.T, self.iv, self.weight, self.total_variance)])


def to_tensors(snapshot, index: np.ndarray | None = None) -> SurfaceTensors:
    """Pack a `ChainSnapshot` (or a subset of it) into tensors."""
    sel = slice(None) if index is None else index
    t = lambda a: torch.tensor(np.asarray(a)[sel], dtype=torch.float64)

    weight = t(snapshot.weight)
    return SurfaceTensors(
        k=t(snapshot.k),
        T=t(snapshot.T),
        iv=t(snapshot.iv),
        # Renormalise after slicing so train and validation losses are on the
        # same scale and directly comparable.
        weight=weight / weight.mean(),
        total_variance=t(snapshot.total_variance),
    )


def stratified_split(snapshot, val_fraction: float = 0.2, seed: int = 0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a fraction of the strikes *within every expiry*.

    Returns ``(train_index, val_index)`` into the snapshot's arrays. Expiries
    with too few quotes to spare any are kept entirely in the training set --
    dropping a wing point from a 6-quote smile costs more than the validation
    signal is worth.

    Within an expiry the smile is sorted by strike and cut into ``n_val`` equal
    blocks, one held-out quote per block, at a random offset drawn once per
    expiry. That covers the full strike range including both wings, gives every
    quote the same probability of being held out, and makes a different seed
    score genuinely different strikes rather than the same ones shifted by one.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for T in np.unique(snapshot.T):
        idx = np.flatnonzero(snapshot.T == T)
        n_val = int(round(val_fraction * len(idx)))
        if len(idx) < 6 or n_val == 0:
            train_idx.append(idx)
            continue
        order = idx[np.argsort(snapshot.k[idx])]
        n = len(order)
        step = n / n_val
        picks = np.unique(
            np.clip((rng.random() * step + step * np.arange(n_val)).astype(int), 0, n - 1)
        )
        mask = np.zeros(n, dtype=bool)
        mask[picks] = True
        val_idx.append(order[mask])
        train_idx.append(order[~mask])

    return np.concatenate(train_idx), (
        np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
    )
