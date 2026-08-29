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
    # ── what to fit ──────────────────────────────────────────────────────────
    ticker: str = "SPY"
    """Underlying to fit. Liquid ETFs and large caps work; illiquid names will
    not survive the quote filters."""

    use_live_data: bool = True
    """True fetches today's chain from Yahoo Finance. If that fails -- no
    network, rate limit, market closed with stale quotes -- the run falls back
    to a generated chain automatically and says so, so pressing Run always
    produces a result."""

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
    volatility -- on SPY that is ~12bp of vol beyond one year, against a fit
    measured in tens of bp. Costs a few seconds; see `volsurface.american`."""

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
    make_plots: bool = True
    save_model: bool = True
    run_baselines: bool = True
    """Fitting per-slice SVI is the slowest step. It is also the point of the
    comparison, so it is on by default."""

    verbose: bool = True

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


#: The configuration `main.py` runs. Edit this.
CONFIG = RunConfig()
