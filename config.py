"""
config.py
Every knob for a run, in one file.

`main.py` takes no command-line arguments on purpose: the project is a quant
study, not a CLI tool, and a run should be reproducible by reading one file
rather than by remembering which flags were passed. Change a value here and
press Run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    # ── what to do ───────────────────────────────────────────────────────────
    mode: str = "ask"
    """"ask" shows the menu (train / price / compare). Set to "train", "price"
    or "compare" to skip straight to it -- useful for a scheduled or scripted
    run. Ignored when nothing is listening for input (falls back to "train")."""

    show_plots: bool = True
    """Pop figures up in a window as well as saving them, when there is a human
    at the keyboard to look at them. Always off in a non-interactive run --
    a window nobody can close would just hang it."""

    # ── what to fit ──────────────────────────────────────────────────────────
    ticker: str = "SPY"
    """Underlying to fit. Liquid ETFs and large caps work; illiquid names will
    not survive the quote filters."""

    use_live_data: bool = True
    """True fetches today's chain from Yahoo Finance. If that fails -- no
    network, rate limit, market closed with stale quotes -- the run falls back
    to a generated chain automatically and says so, so pressing Run always
    produces a result."""

    chain_file: Path | None = None
    """Fit an already-downloaded chain (a `*_chain.npz` written by a previous
    train run) instead of fetching one. Overrides `use_live_data`. This is what
    makes an experiment reproducible: a sweep over architectures has to see the
    same quotes in every arm, and two runs a minute apart do not."""

    max_expiries: int = 8
    """Expiries are sampled evenly in sqrt(T) across the whole listed term
    structure, not taken from the front. Eight covers a week to two years."""

    max_maturity_years: float = 2.0
    min_maturity_years: float = 0.02
    """Roughly one week to two years. Anything shorter is dominated by the tick
    size; anything longer barely trades."""

    # ── quote filters ────────────────────────────────────────────────────────
    min_open_interest: int = 10
    max_relative_spread: float = 0.25
    """A quote wider than 25% of its own mid carries no usable volatility."""

    de_americanize: bool = True
    """Listed equity and ETF options are American. Leaving this off inverts them
    with a European formula, which charges the early-exercise premium to
    volatility -- on SPY that is ~13bp of vol on average and ~25bp beyond one
    year, against a fit measured in tens of bp. Costs a few seconds; see
    `volsurface.pricing.american`."""

    lattice_steps: int = 150
    """Steps in the binomial tree used for de-Americanisation. The premium is a
    difference of two prices off the same lattice, so its discretisation error
    cancels and 150 is ample."""

    # ── the network ──────────────────────────────────────────────────────────
    epochs: int = 1500
    hidden_layers: tuple[int, ...] = (64, 64, 64)
    alpha: float = 0.35
    """Maximum relative correction to the SSVI prior. 0.35 in total variance is
    about 16% in vol terms -- the model's error bound, fixed before training."""

    learning_rate: float = 3e-3
    validation_fraction: float = 0.2
    seed: int = 0

    # ── no-arbitrage penalties ───────────────────────────────────────────────
    calendar_weight: float = 10.0
    butterfly_weight: float = 1.0
    collocation_points: int = 2048
    """Points per epoch on which the constraints are enforced, drawn from a
    region wider than the quotes and resampled every step."""

    # ── what to produce ──────────────────────────────────────────────────────
    output_dir: Path = Path("results")
    """Always-overwritten pointer to the *most recent* run -- what mode [2]
    (price) reads from. Fast, but not a history."""

    models_dir: Path = Path("models")
    """Every TRAIN run is archived here under its own timestamped folder --
    weights, the chain it saw, the full training history, and metrics.json.
    Never overwritten, so mode [3] (compare) has something to compare. See
    `volsurface.evaluation.registry`."""

    make_plots: bool = True
    save_model: bool = True
    run_baselines: bool = True
    """Fitting per-slice SVI is the slowest step. It is also the point of the
    comparison, so it is on by default."""

    run_prior_ablation: bool = True
    """Retrain the same network against a flat, constant-vol prior and put it in
    the table as its own row. This is the experiment that answers the first
    question anyone asks about a prior-plus-correction model -- how much of the
    result is the network and how much is the SSVI shape it started from -- and
    without it the headline number is not interpretable. Costs one extra
    training run (roughly doubles the fit time); set to False when iterating."""

    verbose: bool = True

    # ── capacity sweep (mode "sweep") ────────────────────────────────────────
    sweep_architectures: tuple[str, ...] = ("32x2", "64x3", "128x3", "256x4", "512x4")
    """Architectures to fit, as WIDTHxDEPTH, all on the same chain with the same
    budget and the same seed. This is the experiment that answers "is the network
    too small?" with a number instead of an opinion: the surface has a few dozen
    effective degrees of freedom, so the expectation is that validation error
    flattens and then worsens well before the parameter count gets interesting.
    Run it and see, rather than arguing about it."""

    sweep_seeds: int = 1
    """Seeds per architecture. One is enough to see the shape of the curve; three
    is what you want before claiming two architectures actually differ, since
    run-to-run scatter on a real chain is a few basis points."""

    # ── pricing (mode "price", non-interactive only) ────────────────────────
    price_maturities: tuple[float, ...] = (0.25, 0.5, 1.0)
    """Maturities (years) to price when nothing is listening for input. In an
    interactive run these are just the first prompt's default."""

    price_strikes: tuple[float, ...] = ()
    """Strikes to price non-interactively. Empty means a ladder around the
    forward at each maturity in `price_maturities`."""

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.models_dir = Path(self.models_dir)
        if self.chain_file is not None:
            self.chain_file = Path(self.chain_file)
        self.hidden_layers = tuple(self.hidden_layers)


#: The configuration `main.py` runs. Edit this.
CONFIG = RunConfig()
