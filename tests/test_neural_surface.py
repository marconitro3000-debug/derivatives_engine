"""The neural surface: structural guarantees, exact derivatives, and training."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from volsurface.svi import SSVISurface
from volsurface.diagnostics import scan_arbitrage
from volsurface.neural.arbitrage import CollocationRegion, PenaltyWeights, arbitrage_penalties, durrleman_g
from volsurface.neural.dataset import stratified_split, to_tensors
from volsurface.neural.model import ModelConfig, NeuralVolSurface
from volsurface.neural.prior import FlatPrior, TorchSSVIPrior
from volsurface.neural.train import TrainConfig, train_surface


@pytest.fixture(scope="module")
def prior(clean_chain):
    return TorchSSVIPrior.from_surface(SSVISurface.fit(clean_chain))


@pytest.fixture(scope="module")
def model(clean_chain, prior):
    mean, std = NeuralVolSurface.feature_stats(clean_chain.k, clean_chain.T)
    return NeuralVolSurface(prior, ModelConfig(), mean, std)


# -- the prior ----------------------------------------------------------------

def test_torch_prior_matches_the_numpy_ssvi_surface(clean_chain, prior):
    """The prior the network corrects must be the surface the report benchmarks."""
    ssvi = SSVISurface.fit(clean_chain)
    k = np.linspace(-0.6, 0.5, 50)
    T = np.linspace(0.05, 2.0, 50)

    w_np = ssvi.total_variance(k, T)
    w_torch = prior.total_variance(torch.tensor(k), torch.tensor(T)).numpy()
    assert np.allclose(w_np, w_torch, rtol=1e-10)


def test_prior_theta_is_differentiable_in_maturity(prior):
    T = torch.linspace(0.05, 2.0, 20, dtype=torch.float64).requires_grad_(True)
    theta = prior.theta(T)
    (dtheta,) = torch.autograd.grad(theta.sum(), T)
    assert torch.all(dtheta >= -1e-12), "theta(T) must be non-decreasing"


# -- structural guarantees ----------------------------------------------------

def test_untrained_model_reproduces_the_prior_exactly(clean_chain, prior, model):
    """Zero-initialised output layer => correction is exactly 1 at step zero."""
    k = np.linspace(-0.5, 0.4, 30)
    T = np.full_like(k, 0.5)
    w_prior = prior.total_variance(torch.tensor(k), torch.tensor(T)).numpy()
    assert np.allclose(model.total_variance(k, T), w_prior, rtol=1e-12)


def test_correction_is_bounded_by_alpha(clean_chain, prior):
    """Even with absurd weights the surface cannot leave the prior's neighbourhood."""
    cfg = ModelConfig(alpha=0.35)
    mean, std = NeuralVolSurface.feature_stats(clean_chain.k, clean_chain.T)
    m = NeuralVolSurface(prior, cfg, mean, std)
    with torch.no_grad():
        for p in m.net.parameters():
            p.add_(torch.randn_like(p) * 50.0)

    k = torch.linspace(-3.0, 3.0, 200, dtype=torch.float64)
    T = torch.full_like(k, 0.5)
    c = m.correction(k, T)
    assert torch.all(c >= 1 - cfg.alpha - 1e-12)
    assert torch.all(c <= 1 + cfg.alpha + 1e-12)


def test_total_variance_is_strictly_positive_far_outside_the_data(clean_chain, prior):
    """Positivity is structural, so it must survive extreme extrapolation."""
    mean, std = NeuralVolSurface.feature_stats(clean_chain.k, clean_chain.T)
    m = NeuralVolSurface(prior, ModelConfig(), mean, std)
    with torch.no_grad():
        for p in m.net.parameters():
            p.add_(torch.randn_like(p) * 20.0)
    k = np.linspace(-5.0, 5.0, 200)
    assert np.all(m.total_variance(k, np.full_like(k, 3.0)) > 0)


# -- derivatives --------------------------------------------------------------

def test_autograd_derivatives_match_finite_differences(model):
    """The autograd overrides must agree with the base class's finite differences."""
    k = np.linspace(-0.3, 0.3, 15)
    T = np.full_like(k, 0.6)

    from volsurface.surface import VolSurface

    assert np.allclose(model.dw_dk(k, T), VolSurface.dw_dk(model, k, T), atol=1e-6)
    assert np.allclose(model.d2w_dk2(k, T), VolSurface.d2w_dk2(model, k, T), atol=1e-4)
    assert np.allclose(model.dw_dT(k, T), VolSurface.dw_dT(model, k, T), atol=1e-6)


def test_derivative_calls_do_not_mutate_their_inputs(model):
    k = np.array([0.1, 0.2])
    T = np.array([0.5, 0.5])
    model.dw_dk(k, T)
    assert np.array_equal(k, [0.1, 0.2]) and np.array_equal(T, [0.5, 0.5])


# -- penalties ----------------------------------------------------------------

def _grid(n=48):
    k, T = torch.meshgrid(
        torch.linspace(-0.6, 0.5, n, dtype=torch.float64),
        torch.linspace(0.05, 1.5, n, dtype=torch.float64),
        indexing="ij",
    )
    return k.reshape(-1), T.reshape(-1)


def test_penalties_are_zero_on_an_arbitrage_free_surface(model):
    pen = arbitrage_penalties(model, *_grid())
    assert float(pen["calendar"].detach()) == pytest.approx(0.0, abs=1e-14)
    assert float(pen["butterfly"].detach()) == pytest.approx(0.0, abs=1e-14)


class _RippledSurface(torch.nn.Module):
    """A deliberately broken surface: SSVI total variance with a ripple in k.

    The ripple leaves the surface positive and smooth, so nothing else complains
    about it, but it drives the density negative between the crests -- which is
    exactly the failure mode the butterfly penalty exists to catch, and exactly
    what an unconstrained network does when it interpolates noisy quotes.
    """

    def __init__(self, prior, amplitude=0.3, frequency=25.0):
        super().__init__()
        self.prior = prior
        self.amplitude = amplitude
        self.frequency = frequency
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def total_variance_torch(self, k, T):
        ripple = 1.0 + self.amplitude * torch.sin(self.frequency * k)
        return self.prior.total_variance(k, T) * ripple * self.scale


def test_penalties_detect_a_negative_density_and_give_a_gradient(prior):
    """The penalty must fire on a broken surface *and* be differentiable."""
    broken = _RippledSurface(prior)
    pen = arbitrage_penalties(broken, *_grid())

    assert float(pen["butterfly"].detach()) > 0.0
    assert float(pen["worst_butterfly"]) < 0.0
    assert float(pen["butterfly_frac"]) > 0.0

    pen["butterfly"].backward()
    assert broken.scale.grad is not None and float(broken.scale.grad.abs()) > 0.0


def test_penalties_detect_calendar_violations(prior):
    """Total variance shrinking with maturity must show up in the calendar term."""

    class _Inverted(torch.nn.Module):
        def __init__(self, prior):
            super().__init__()
            self.prior = prior

        def total_variance_torch(self, k, T):
            # A surface that decays with maturity: w = w_prior(k, T) * exp(-2T).
            return self.prior.total_variance(k, T) * torch.exp(-2.0 * T)

    pen = arbitrage_penalties(_Inverted(prior), *_grid())
    assert float(pen["calendar"].detach()) > 0.0
    assert float(pen["worst_calendar"]) < 0.0


def test_durrleman_g_is_one_for_a_flat_at_the_money_surface():
    """Sanity anchor: at k=0 with a flat smile, g reduces to 1."""
    w = torch.tensor([0.04], dtype=torch.float64)
    zero = torch.zeros(1, dtype=torch.float64)
    assert float(durrleman_g(w, zero, zero, zero)) == pytest.approx(1.0)


def test_collocation_region_extends_past_the_data(clean_chain):
    region = CollocationRegion.around(clean_chain.k, clean_chain.T, PenaltyWeights())
    assert region.k_lo < clean_chain.k.min()
    assert region.k_hi > clean_chain.k.max()
    assert region.T_hi > clean_chain.T.max()

    k, T = region.sample(500, torch.Generator().manual_seed(0))
    assert float(k.min()) >= region.k_lo and float(k.max()) <= region.k_hi
    assert float(T.min()) >= region.T_lo and float(T.max()) <= region.T_hi


# -- data splitting -----------------------------------------------------------

def test_split_is_disjoint_and_covers_every_expiry(noisy_chain):
    train_idx, val_idx = stratified_split(noisy_chain, 0.2, seed=0)
    assert set(train_idx).isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == len(noisy_chain)
    # Every expiry must still be represented in training.
    assert set(np.unique(noisy_chain.T[train_idx])) == set(np.unique(noisy_chain.T))


def test_validation_points_are_spread_across_the_smile(noisy_chain):
    """Held-out strikes must not all sit in one wing."""
    _, val_idx = stratified_split(noisy_chain, 0.25, seed=0)
    for T in np.unique(noisy_chain.T[val_idx]):
        sel = val_idx[noisy_chain.T[val_idx] == T]
        if len(sel) >= 2:
            assert noisy_chain.k[sel].max() - noisy_chain.k[sel].min() > 0.1


def test_weights_are_renormalised_per_split(noisy_chain):
    train_idx, _ = stratified_split(noisy_chain, 0.2, seed=0)
    assert float(to_tensors(noisy_chain, train_idx).weight.mean()) == pytest.approx(1.0)


# -- training -----------------------------------------------------------------

@pytest.fixture(scope="module")
def trained(noisy_chain):
    cfg = TrainConfig(epochs=250, warmup_epochs=60, patience=250, seed=0)
    cfg.penalties.n_points = 512
    return train_surface(noisy_chain, cfg)


def test_training_improves_on_the_prior(noisy_chain, trained):
    """If the network cannot beat SSVI on its own training data, it is not working."""
    from volsurface.diagnostics import fit_report

    neural = fit_report(trained.model, noisy_chain).rmse_vol_bps
    prior = fit_report(trained.prior_surface, noisy_chain).rmse_vol_bps
    assert neural < prior


def test_trained_surface_is_arbitrage_free(noisy_chain, trained):
    report = scan_arbitrage(
        trained.model,
        k_range=(float(noisy_chain.k.min()) * 1.2, float(noisy_chain.k.max()) * 1.2),
        T_range=(0.02, float(noisy_chain.T.max()) * 1.2),
    )
    assert report.is_arbitrage_free, str(report)


def test_flat_prior_ablation_runs_and_still_fits(noisy_chain):
    """Without the SSVI prior the network must still produce a usable surface."""
    from volsurface.diagnostics import fit_report

    cfg = TrainConfig(epochs=250, warmup_epochs=60, patience=250, seed=0)
    cfg.penalties.n_points = 512
    result = train_surface(noisy_chain, cfg, prior="flat")

    assert isinstance(result.model.prior, FlatPrior)
    assert result.prior_surface is None
    assert fit_report(result.model, noisy_chain).rmse_vol_bps < 1000.0


def test_unknown_prior_is_rejected(noisy_chain):
    with pytest.raises(ValueError, match="unknown prior"):
        train_surface(noisy_chain, TrainConfig(epochs=1), prior="heston")


def test_alpha_must_keep_the_surface_positive():
    with pytest.raises(ValueError, match="alpha"):
        ModelConfig(alpha=1.5)


def test_save_and_load_roundtrip(tmp_path, noisy_chain, trained):
    path = tmp_path / "surface.pt"
    trained.model.save(path)
    reloaded = NeuralVolSurface.load(path)

    k = np.linspace(-0.4, 0.3, 25)
    T = np.full_like(k, 0.7)
    assert np.allclose(reloaded.total_variance(k, T),
                       trained.model.total_variance(k, T), rtol=1e-12)
