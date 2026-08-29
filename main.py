"""
main.py
The whole study, start to finish. Press Run.

    1. load and clean an option chain
    2. calibrate the parametric baselines (per-slice SVI, joint SSVI)
    3. train the arbitrage-penalised neural surface
    4. score all three on fit *and* on static arbitrage
    5. write the report and the figures to results/

No arguments, no server, no terminal flags. Settings live in `config.py`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Windows consoles still default to cp1252, which cannot encode the characters
# a report naturally reaches for. Widen the console where the interpreter allows
# it; the report text below stays ASCII regardless, because a study that crashes
# on its own output banner is not a study anyone will run twice.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):                     # pragma: no cover
        pass

from config import CONFIG, RunConfig
from volsurface import (
    ModelConfig,
    PenaltyWeights,
    TrainConfig,
    compare,
    comparison_table,
    fetch_chain,
    plot_arbitrage_map,
    plot_fit,
    synthetic_snapshot,
    train_surface,
)


# ── presentation ─────────────────────────────────────────────────────────────

def rule(title: str = "", width: int = 92) -> str:
    if not title:
        return "=" * width
    return f"-- {title} " + "-" * max(width - len(title) - 4, 0)


class Tee:
    """Write to stdout and to the report file at once.

    The console output *is* the report -- keeping a second, differently
    formatted summary in sync with it is how the two end up disagreeing.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.file.write(text + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


# ── steps ────────────────────────────────────────────────────────────────────

def load_chain(cfg: RunConfig, say):
    """Fetch a live chain, falling back to a generated one so a run never fails.

    The fallback is loud, not silent: a study that quietly reports synthetic
    numbers under a real ticker's name is worse than one that fails.
    """
    filters = dict(
        min_open_interest=cfg.min_open_interest,
        max_rel_spread=cfg.max_relative_spread,
        min_T=cfg.min_maturity_years,
        max_T=cfg.max_maturity_years,
        de_americanize=cfg.de_americanize,
        lattice_steps=cfg.lattice_steps,
    )

    if cfg.use_live_data:
        try:
            chain = fetch_chain(cfg.ticker, max_expiries=cfg.max_expiries, **filters)
            say("source   live option chain (Yahoo Finance)")
            return chain, True
        except Exception as exc:                              # noqa: BLE001
            say(f"!  live chain unavailable for {cfg.ticker}: {exc}")
            say("!  falling back to a GENERATED chain - numbers below are synthetic")

    chain = synthetic_snapshot(ticker=cfg.ticker, noise_bps=25.0, seed=cfg.seed)
    say("source   generated from a known arbitrage-free SSVI surface")
    return chain, False


def build_train_config(cfg: RunConfig) -> TrainConfig:
    return TrainConfig(
        epochs=cfg.epochs,
        lr=cfg.learning_rate,
        val_fraction=cfg.validation_fraction,
        seed=cfg.seed,
        log_every=max(cfg.epochs // 8, 1) if cfg.verbose else 0,
        penalties=PenaltyWeights(
            calendar=cfg.calendar_weight,
            butterfly=cfg.butterfly_weight,
            n_points=cfg.collocation_points,
        ),
        model=ModelConfig(
            hidden=tuple(cfg.hidden_layers),
            alpha=cfg.alpha,
            seed=cfg.seed,
        ),
    )


# ── entry point ──────────────────────────────────────────────────────────────

def main(cfg: RunConfig = CONFIG) -> int:
    t0 = time.time()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    say = Tee(cfg.output_dir / "report.txt")

    try:
        say(rule())
        say(f"  NEURAL IMPLIED-VOLATILITY SURFACE  |  {cfg.ticker}")
        say(rule())

        # 1 ── data ----------------------------------------------------------
        say()
        say(rule("1/4  option chain"))
        chain, is_live = load_chain(cfg, say)
        say(chain.summary())
        say()
        say("  american exercise:")
        say(chain.early_exercise_summary())

        if len(chain.forwards) < 2:
            say("\nNeed at least two expiries to fit a surface. Stopping.")
            return 1

        # 2 ── fit -----------------------------------------------------------
        say()
        say(rule("2/4  training the neural surface"))
        say("  w(k,T) = w_SSVI(k,T) * [1 + alpha*tanh(net(k,T))]     "
            f"alpha = {cfg.alpha}, hidden = {tuple(cfg.hidden_layers)}")
        say("  penalties: dw/dT >= 0 and Durrleman g >= 0, on "
            f"{cfg.collocation_points} collocation points per epoch")
        say()
        result = train_surface(chain, build_train_config(cfg))
        say(result.summary())

        # 3 ── score ---------------------------------------------------------
        say()
        say(rule("3/4  scorecard"))
        scores = compare(chain, result, include_baselines=cfg.run_baselines)
        say(comparison_table(scores))
        say()
        say("  IV RMSE     vega-weighted implied-vol error against the quotes")
        say("  max err     worst single-quote vol error")
        say("  px RMSE     the same fit measured in price space")
        say("  in spread   share of quotes re-priced inside the bid-ask")
        say("  cal viol  ) share of a dense (k,T) grid - extending past the quoted")
        say("  bfly viol ) strikes - that admits calendar / butterfly arbitrage")

        # 4 ── artefacts -----------------------------------------------------
        say()
        say(rule("4/4  output"))
        if cfg.save_model:
            path = cfg.output_dir / f"{cfg.ticker.lower()}_surface.pt"
            result.model.save(path)
            say(f"  model    {path}")

        if cfg.make_plots:
            say(f"  figures  {plot_fit(chain, [s.surface for s in scores], str(cfg.output_dir / 'smiles.png'))}")
            for score in scores:
                slug = score.name.split()[0].lower()
                say(f"           {plot_arbitrage_map(score.surface, chain, str(cfg.output_dir / f'arbitrage_{slug}.png'))}")

        say(f"  report   {cfg.output_dir / 'report.txt'}")
        say()
        say(rule())
        say(f"  done in {time.time() - t0:.1f}s"
            + ("" if is_live else "   (SYNTHETIC data - see the warning above)"))
        say(rule())
        return 0

    finally:
        say.close()


if __name__ == "__main__":
    sys.exit(main())
