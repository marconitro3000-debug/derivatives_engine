"""
main.py
Press Run. It asks what you want to do.

    [1] TRAIN   fit a surface to an option chain and score it against the
                parametric baselines. Writes the report, the training curves,
                the smiles and the arbitrage maps to results/.

    [2] PRICE   load the surface trained last time and query it: give a strike
                and a maturity, get the implied vol, the price, the Greeks and
                the local no-arbitrage diagnostics -- plus the smile with your
                strike marked on it, so you can see whether the answer is an
                interpolation between real quotes or an extrapolation.

Training takes about three minutes; pricing is instant, which is the whole
reason the two are separate. A desk calibrates on a schedule and prices on every
request, and this is the same split.

    [3] COMPARE rank every archived run by validation error, so "which
                checkpoint do I use" has an answer.

    [4] SWEEP   fit several architectures to one chain and table capacity
                against error -- the experiment, not the argument.

Training takes a few minutes; pricing is instant, which is the whole reason the
two are separate. A desk calibrates on a schedule and prices on every request,
and this is the same split.

Defaults live in `config.py`. Every one of them is also a command-line flag --
`python main.py --help` generates the list from the dataclass, and
`docs/COMMANDS.md` documents every command in the repo, this file included.
Run with no arguments and you get the menu, exactly as before.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

# Windows consoles still default to cp1252, which cannot encode the characters a
# report naturally reaches for. Widen the console where the interpreter allows
# it; the report text below stays ASCII regardless, because a study that crashes
# on its own output banner is not a study anyone will run twice.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):                     # pragma: no cover
        pass

import numpy as np

from cli import config_from_args, parse_layers
from config import CONFIG, RunConfig
from volsurface import (
    ChainSnapshot,
    ModelConfig,
    NeuralVolSurface,
    PenaltyWeights,
    TrainConfig,
    compare,
    comparison_table,
    evaluate_surface,
    fetch_chain,
    greeks,
    list_runs,
    load_history,
    load_run,
    plot_arbitrage_map,
    plot_fit,
    plot_model_comparison,
    plot_quote,
    plot_training,
    price,
    runs_table,
    save_run,
    synthetic_snapshot,
    train_surface,
    worst_quote_notes,
)
from volsurface.diagnostics import butterfly_g


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

    def __init__(self, path: Path | None):
        self.file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w", encoding="utf-8")

    def __call__(self, text: str = "") -> None:
        print(text)
        if self.file:
            self.file.write(text + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.close()


def interactive() -> bool:
    """Is there a human at the other end who can answer a prompt?"""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:                                        # pragma: no cover
        return False


def ask(prompt: str, default: str = "") -> str:
    """Prompt, falling back to `default` when nothing is listening."""
    if not interactive():
        return default
    try:
        return input(prompt).strip() or default
    except (EOFError, KeyboardInterrupt):
        return default


def ask_float(prompt: str, default: float | None = None) -> float | None:
    while True:
        raw = ask(prompt, "" if default is None else str(default))
        if raw in ("", "q", "quit", "exit"):
            return None
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print(f"  '{raw}' is not a number -- try again, or blank to go back.")
            if not interactive():
                return None


def choose_mode(cfg: RunConfig) -> str:
    """Menu, unless `config.mode` already decided or nobody is listening."""
    if cfg.mode in ("train", "price", "compare", "sweep"):
        return cfg.mode
    if not interactive():
        return "train"

    print(rule())
    print("  NEURAL IMPLIED-VOLATILITY SURFACE")
    print(rule())
    print()
    print("  [1]  Train a surface        fit a chain, score it, plot everything")
    print("                              (~3 min: downloads live quotes)")
    print("  [2]  Price off a surface    query the one trained last time")
    print("                              (instant, no network)")
    print("  [3]  Compare trained models which archived run to use, and why")
    print("                              (instant, no network)")
    print("  [4]  Sweep architectures    is the network too small? fit several")
    print("                              sizes on one chain and table the answer")
    print()
    print("  (everything here is also a command: python main.py --help,")
    print("   full reference in docs/COMMANDS.md)")
    print()
    choice = ask("  choose [1]: ", "1")
    return {
        "2": "price", "price": "price", "p": "price",
        "3": "compare", "compare": "compare", "c": "compare",
        "4": "sweep", "sweep": "sweep", "s": "sweep",
    }.get(choice.lower(), "train")


def known_tickers(cfg: RunConfig) -> list[str]:
    """Every ticker with something archived, newest activity first.

    Scanned from `models/` (every past train run) union whatever `results/`
    currently points at, so the prompt below can show "here is what you have"
    instead of asking the ticker to be typed from memory.
    """
    seen: list[str] = []
    if cfg.models_dir.exists():
        for run_dir in sorted(cfg.models_dir.iterdir(),
                              key=lambda p: p.stat().st_mtime, reverse=True):
            meta = run_dir / "metrics.json"
            if not meta.exists():
                continue
            try:
                import json
                t = json.loads(meta.read_text(encoding="utf-8"))["ticker"]
            except (OSError, ValueError, KeyError):
                continue
            if t not in seen:
                seen.append(t)
    if cfg.output_dir.exists():
        for p in cfg.output_dir.glob("*_surface.pt"):
            t = p.stem.removesuffix("_surface").upper()
            if t not in seen:
                seen.append(t)
    return seen


def choose_ticker(cfg: RunConfig, mode: str) -> str:
    """Ask which underlying to work with. `config.ticker` is only the default.

    Training accepts any symbol -- it will be fetched live. Pricing and
    comparing only make sense for a ticker that has already been trained, so
    those two show what is actually on disk rather than let a typo lead
    straight to "no trained surface found".
    """
    if not interactive():
        return cfg.ticker

    print()
    if mode == "train":
        raw = ask(f"  ticker to fit [{cfg.ticker}] (any symbol -- fetched live): ")
        return raw.upper() if raw else cfg.ticker

    available = known_tickers(cfg)
    if not available:
        print(f"  no trained surfaces on disk yet -- train one first (mode [1]).")
        return cfg.ticker

    default = cfg.ticker if cfg.ticker in available else available[0]
    print(f"  available: {', '.join(available)}")
    raw = ask(f"  ticker [{default}]: ")
    return (raw.upper() if raw else default)


# ── shared steps ─────────────────────────────────────────────────────────────

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

    if cfg.chain_file is not None:
        chain = ChainSnapshot.load(cfg.chain_file)
        cfg.ticker = chain.ticker
        say(f"source   cached chain {cfg.chain_file}  (as of {chain.asof})")
        return chain, True

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


def artefact_paths(cfg: RunConfig) -> dict[str, Path]:
    stem = cfg.ticker.lower()
    return {
        "model": cfg.output_dir / f"{stem}_surface.pt",
        "chain": cfg.output_dir / f"{stem}_chain.npz",
        "report": cfg.output_dir / "report.txt",
    }


# ── mode 1: train ────────────────────────────────────────────────────────────

def run_train(cfg: RunConfig) -> int:
    t0 = time.time()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    paths = artefact_paths(cfg)
    say = Tee(paths["report"])
    show = cfg.show_plots and interactive()

    try:
        say(rule())
        say(f"  TRAIN  |  {cfg.ticker}")
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

        # The ablation: same network, same penalties, same budget, but starting
        # from a constant-vol prior instead of SSVI. It is the only honest way
        # to say how much of the headline number the network earned.
        ablation = None
        if cfg.run_prior_ablation:
            say()
            say("  ablation: same network against a flat constant-vol prior")
            ablation = train_surface(chain, build_train_config(cfg), prior="flat")
            say("  " + ablation.summary().replace("\n", "\n  "))

        # 3 ── score ---------------------------------------------------------
        say()
        say(rule("3/4  scorecard"))
        scores = compare(chain, result, include_baselines=cfg.run_baselines,
                         extra_surfaces=(ablation.model,) if ablation else ())
        say(comparison_table(scores))
        say()
        say("  IV RMSE     vega-weighted implied-vol error against the quotes")
        say("  max err     worst single-quote vol error (located below)")
        say("  px RMSE     the same fit measured in price space")
        say("  in spread   share of quotes re-priced inside the bid-ask")
        say("  cal viol  ) share of a dense (k,T) grid - extending past the quoted")
        say("  bfly viol ) strikes - that admits calendar / butterfly arbitrage")
        say()
        say(worst_quote_notes(scores))
        say("  (a large error at a weight near zero is the vega/spread weighting"
            " declining to chase a wing quote)")

        # 4 ── artefacts -----------------------------------------------------
        say()
        say(rule("4/4  output"))
        if cfg.save_model:
            result.model.save(paths["model"])
            chain.save(paths["chain"])
            say(f"  model    {paths['model']}")
            say(f"  chain    {paths['chain']}")
            say("           (both are what mode [2] loads to price without re-downloading)")

            record = save_run(cfg.models_dir, cfg, chain, result, scores)
            say(f"  archive  {record.dir}")
            say(f"           (mode [3] compares every archived run to help pick one)")

        if cfg.make_plots:
            say(f"  curves   {plot_training(result, str(cfg.output_dir / 'training.png'), show)}")
            say(f"  smiles   {plot_fit(chain, [s.surface for s in scores], str(cfg.output_dir / 'smiles.png'), show=show)}")
            for score in scores:
                # "Neural (flat prior + penalties)" -> neural_flat: the first
                # word alone would collide with the SSVI-prior run's map.
                words = score.name.replace("(", " ").split()
                slug = words[0].lower()
                if slug == "neural":
                    slug += "_" + words[1].lower()      # ssvi / flat
                out = cfg.output_dir / f"arbitrage_{slug}.png"
                say(f"  arb map  {plot_arbitrage_map(score.surface, chain, str(out), show)}")

        say(f"  report   {paths['report']}")
        say()
        say(rule())
        say(f"  done in {time.time() - t0:.1f}s"
            + ("" if is_live else "   (SYNTHETIC data - see the warning above)"))
        say(rule())

        if interactive():
            say()
            say("  Run again and choose [2] to price this surface, or")
            say("  [3] to compare it against every other archived run.")
        return 0

    finally:
        say.close()


# ── mode 4: sweep ────────────────────────────────────────────────────────────

def run_sweep(cfg: RunConfig) -> int:
    """Fit several architectures to one chain and table the result.

    The question this exists to settle is "is the network too small?", and it is
    not a question anyone should answer by reasoning about it. Every arm sees
    the same quotes, the same split, the same budget, the same seed and the same
    penalties; the only thing that changes is the shape of `net`. What comes out
    is validation error against parameter count, which is the curve that decides
    whether capacity is the binding constraint or whether the bid-ask is.

    The chain is fetched once and reused across arms. Fitting each arm to its
    own freshly downloaded chain would confound architecture with whatever the
    market did in the intervening minutes -- on a live chain that is easily
    larger than the effect being measured.
    """
    from volsurface.neural.model import ModelConfig

    t0 = time.time()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    say = Tee(cfg.output_dir / "sweep.txt")

    try:
        say(rule())
        say(f"  SWEEP  |  {cfg.ticker}")
        say(rule())
        say()
        say(rule("1/3  option chain"))
        chain, _ = load_chain(cfg, say)
        say(chain.summary())

        if len(chain.forwards) < 2:
            say("\nNeed at least two expiries to fit a surface. Stopping.")
            return 1

        # Every arm reuses these quotes. Saving them makes the whole sweep
        # replayable with --chain-file, which is the difference between an
        # experiment and an anecdote.
        chain_path = cfg.output_dir / f"{cfg.ticker.lower()}_sweep_chain.npz"
        chain.save(chain_path)

        say()
        say(rule("2/3  fitting each architecture"))
        say(f"  {len(cfg.sweep_architectures)} architectures x {cfg.sweep_seeds} seed(s), "
            f"{cfg.epochs} epochs each, identical chain and split")
        say()

        rows = []
        for spec in cfg.sweep_architectures:
            hidden = parse_layers(spec)
            for seed in range(cfg.sweep_seeds):
                arm = dataclasses.replace(cfg, hidden_layers=hidden, seed=seed,
                                          verbose=False)
                train_cfg = build_train_config(arm)
                train_cfg.log_every = 0
                # Same held-out quotes in every arm and every seed. Letting the
                # seed move the split too makes the seeds incomparable: the
                # scatter that produces is larger than the architecture effect
                # the sweep exists to measure.
                train_cfg.split_seed = cfg.seed
                result = train_surface(chain, train_cfg)
                score = evaluate_surface(result.model, chain)
                n_params = sum(p.numel() for p in result.model.net.parameters())
                rows.append({
                    "spec": spec, "seed": seed, "params": n_params,
                    "train": result.history["train_rmse_bps"][result.best_epoch],
                    "val": result.best_val_rmse_bps,
                    "chain": score.fit.rmse_vol_bps,
                    "bfly": score.arb.butterfly_pct,
                    "cal": score.arb.calendar_pct,
                    "feasible": result.best_is_feasible,
                    "secs": result.elapsed_sec,
                })
                say(f"  {spec:>7}  {n_params:>9,d} params  "
                    f"seed {seed}  train {rows[-1]['train']:6.1f}bp  "
                    f"val {rows[-1]['val']:6.1f}bp  "
                    f"bfly {rows[-1]['bfly']:5.2f}%  {rows[-1]['secs']:5.1f}s")

        say()
        say(rule("3/3  capacity vs error"))
        say(sweep_table(rows, cfg))

        if cfg.make_plots:
            path = plot_capacity(rows, str(cfg.output_dir / "sweep.png"),
                                 cfg.show_plots and interactive())
            say(f"\n  curve    {path}")
        say(f"  chain    {chain_path}")
        say(f"           (replay any arm with --chain-file {chain_path})")
        say(f"  report   {cfg.output_dir / 'sweep.txt'}")

        say()
        say(rule())
        say(f"  done in {time.time() - t0:.1f}s")
        say(rule())
        return 0
    finally:
        say.close()


def sweep_table(rows: list[dict], cfg: RunConfig) -> str:
    """The sweep as a table, with the smallest architecture as the reference.

    Reported against the *baseline* rather than in isolation, because the number
    that matters is not "512x4 scores X" but "512x4 buys you X basis points over
    the default for Y times the parameters".
    """
    import numpy as np

    by_spec: dict[str, list[dict]] = {}
    for r in rows:
        by_spec.setdefault(r["spec"], []).append(r)

    default_spec = next((s for s in by_spec
                         if parse_layers(s) == tuple(CONFIG.hidden_layers)), None)
    ref = float(np.mean([r["val"] for r in by_spec[default_spec]])) if default_spec \
        else min(float(np.mean([r["val"] for r in v])) for v in by_spec.values())

    header = (f"{'arch':>8} {'params':>10} {'train':>9} {'val':>9} {'chain':>9} "
              f"{'vs default':>11} {'cal':>7} {'bfly':>7} {'clean':>6} {'time':>7}")
    lines = [header, "-" * len(header)]
    for spec, group in by_spec.items():
        val = float(np.mean([r["val"] for r in group]))
        lines.append(
            f"{spec:>8} {group[0]['params']:>10,d} "
            f"{np.mean([r['train'] for r in group]):>8.1f}bp "
            f"{val:>8.1f}bp "
            f"{np.mean([r['chain'] for r in group]):>8.1f}bp "
            f"{val - ref:>+10.1f}bp "
            f"{np.mean([r['cal'] for r in group]):>6.2f}% "
            f"{np.mean([r['bfly'] for r in group]):>6.2f}% "
            f"{sum(r['feasible'] for r in group):>3}/{len(group):<2} "
            f"{np.mean([r['secs'] for r in group]):>6.1f}s"
        )

    lines += [
        "",
        "  train / val   vega-weighted IV RMSE at the selected epoch",
        "  chain         the same error over every quote, train and validation",
        "  vs default    validation error relative to the configured architecture;",
        "                negative means the bigger network actually bought something",
        "  cal / bfly    share of a dense (k,T) grid admitting arbitrage",
        "  clean         arms whose selected epoch was arbitrage-free on its own draw",
    ]
    return "\n".join(lines)


def plot_capacity(rows: list[dict], path: str | None = None, show: bool = False):
    """Validation error against parameter count, log x."""
    import matplotlib.pyplot as plt
    import numpy as np

    by_spec: dict[str, list[dict]] = {}
    for r in rows:
        by_spec.setdefault(r["spec"], []).append(r)

    specs = list(by_spec)
    params = np.array([by_spec[s][0]["params"] for s in specs], dtype=float)
    val = np.array([np.mean([r["val"] for r in by_spec[s]]) for s in specs])
    train = np.array([np.mean([r["train"] for r in by_spec[s]]) for s in specs])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(params, train, "o-", lw=1.3, ms=4, label="train")
    ax.plot(params, val, "o-", lw=1.6, ms=5, label="validation")
    for x, y, s in zip(params, val, specs):
        ax.annotate(s, (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters in the correction network")
    ax.set_ylabel("IV RMSE (bp)")
    ax.set_title("Capacity vs error — same chain, same budget, same split")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _finish_local(fig, path, show)


def _finish_local(fig, path, show):
    if path:
        fig.savefig(path, dpi=150)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return path if path else fig


# ── mode 3: compare ──────────────────────────────────────────────────────────

def run_compare(cfg: RunConfig) -> int:
    """Every archived TRAIN run, ranked by validation error.

    `models/` only ever grows -- each TRAIN run archives its own timestamped
    folder rather than overwriting the last one -- so after a few runs this is
    the answer to "which checkpoint do I actually use".
    """
    records = list_runs(cfg.models_dir, cfg.ticker)
    say = Tee(None)
    show = cfg.show_plots and interactive()

    say(rule())
    say(f"  COMPARE  |  {cfg.ticker}  --  {len(records)} archived run(s) in {cfg.models_dir}/")
    say(rule())
    say()

    if not records:
        say(f"  Nothing archived yet for {cfg.ticker}.")
        say("  Run again and choose [1] to train one.")
        return 1

    say(runs_table(records))

    best = min(records, key=lambda r: r.val_rmse_bps)
    say()
    say(f"  best by validation error: {best.run_id}  "
       f"({best.val_rmse_bps:.1f}bp, gap {best.generalization_gap_bps:+.1f}bp)")
    if best.baseline_svi_rmse_bps is not None:
        say(f"    vs. per-slice SVI on that same chain: "
           f"{best.baseline_svi_rmse_bps:.1f}bp")
    if best.baseline_ssvi_rmse_bps is not None:
        say(f"    vs. joint SSVI on that same chain:    "
           f"{best.baseline_ssvi_rmse_bps:.1f}bp")

    if cfg.make_plots or show:
        out = str(cfg.output_dir / "model_comparison.png") if cfg.make_plots else None
        say()
        say(f"  chart    {plot_model_comparison(records, out, show)}")

    if interactive():
        raw = ask(f"\n  overlay training curves for up to 6 runs? [y/N]: ", "n")
        if raw.lower().startswith("y"):
            _plot_overlay(records[:6], cfg, show)

    say()
    say(f"  to price off a specific run rather than the most recent one, copy its")
    say(f"  surface.pt and chain.npz from its folder into {cfg.output_dir}/ "
       f"under the")
    say(f"  <ticker>_surface.pt / <ticker>_chain.npz names mode [2] expects.")
    return 0


def _plot_overlay(records, cfg, show) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for r in records:
        h = load_history(r)
        ax.plot(h["val_rmse_bps"], lw=1.2,
               label=f"{r.run_id}  (best {r.val_rmse_bps:.1f}bp)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation IV RMSE (bp)")
    ax.set_title("Validation curves across archived runs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if cfg.make_plots:
        out = cfg.output_dir / "model_comparison_curves.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"  overlay  {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ── mode 2: price ────────────────────────────────────────────────────────────

def load_fitted(cfg: RunConfig):
    """Load the surface and the chain it was fitted to, or explain why not.

    Both are needed. The model gives the volatility; the chain gives the forward
    and discount factor those volatilities are quoted against, and those were
    fitted from the market rather than assumed -- so pricing without the chain
    would silently substitute a different forward.
    """
    paths = artefact_paths(cfg)
    if not paths["model"].exists() or not paths["chain"].exists():
        print(rule())
        print(f"  No trained surface found for {cfg.ticker} in {cfg.output_dir}/")
        print(rule())
        print()
        print(f"  expected  {paths['model']}")
        print(f"            {paths['chain']}")
        print()
        print("  Run again and choose [1] to train one first.")
        return None, None

    return NeuralVolSurface.load(paths["model"]), ChainSnapshot.load(paths["chain"])


def quote_table(surface, chain, strikes, T) -> str:
    """Implied vol, price, Greeks and the local no-arbitrage checks per strike."""
    F, DF = chain.forward_at(T)
    k = np.log(np.asarray(strikes, dtype=float) / F)
    Tv = np.full_like(k, float(T))

    iv = surface.implied_vol(k, Tv)
    dwdT = surface.dw_dT(k, Tv)
    g = butterfly_g(surface, k, Tv)

    k_lo, k_hi = chain.k.min(), chain.k.max()
    header = (f"{'strike':>9} {'k':>8} {'IV':>8} {'call':>10} {'put':>10} "
              f"{'delta':>8} {'vega':>8} {'gamma':>9} {'theta':>8} "
              f"{'dw/dT':>10} {'g':>9}  where")
    lines = [header, "-" * len(header)]

    for i, K in enumerate(np.asarray(strikes, dtype=float)):
        sig = float(iv[i])
        # S = F with r = 0 is the Black-76 forward measure; discount once.
        call = DF * price(F, K, T, 0.0, sig, "call")
        put = DF * price(F, K, T, 0.0, sig, "put")
        gk = greeks(F, K, T, 0.0, sig)
        inside = k_lo <= k[i] <= k_hi
        lines.append(
            f"{K:>9.2f} {k[i]:>+8.3f} {sig:>7.2%} {call:>10.3f} {put:>10.3f} "
            f"{gk['delta_call']:>8.3f} {gk['vega'] * DF:>8.3f} {gk['gamma']:>9.5f} "
            f"{gk['theta_call'] * DF:>8.3f} "
            f"{dwdT[i]:>+10.2e} {g[i]:>+9.2e}  "
            f"{'quoted' if inside else 'EXTRAP'}"
        )

    lines.append("")
    lines.append(f"  forward {F:.3f}   discount {DF:.5f}   "
                 f"(interpolated from the chain's fitted expiries)")
    lines.append("  delta/vega/theta are the call's; vega per 1% of vol, theta per day")
    lines.append("  dw/dT and g must both be >= 0 -- they are the no-arbitrage")
    lines.append("  conditions evaluated at exactly the point you asked about")
    return "\n".join(lines)


def default_ladder(chain, T) -> np.ndarray:
    """A strike ladder around the forward, when no strikes were given."""
    F, _ = chain.forward_at(T)
    return np.round(F * np.array([0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]), 2)


def run_price(cfg: RunConfig) -> int:
    surface, chain = load_fitted(cfg)
    if surface is None:
        return 1

    say = Tee(None)
    show = cfg.show_plots and interactive()

    say(rule())
    say(f"  PRICE  |  {chain.ticker}  surface fitted {chain.asof}")
    say(rule())
    say(chain.summary())
    say()
    say(f"  listed expiries (years): "
        + ", ".join(f"{t:.3f}" for t in chain.maturities))
    say(f"  quoted strike range at {chain.T.min():.3f}y..{chain.T.max():.3f}y: "
        f"k in [{chain.k.min():+.3f}, {chain.k.max():+.3f}]")

    # Non-interactive: use whatever config asked for and stop.
    if not interactive():
        for T in cfg.price_maturities:
            strikes = list(cfg.price_strikes) or default_ladder(chain, T)
            say()
            say(rule(f"T = {T}y"))
            say(quote_table(surface, chain, strikes, T))
        return 0

    say()
    say("  Enter a strike and a maturity. Blank strike quits; blank maturity")
    say("  gives you a ladder around the forward.")

    while True:
        say()
        T = ask_float(f"  maturity in years [{cfg.price_maturities[0]}]: ",
                      cfg.price_maturities[0])
        if T is None:
            break
        if T <= 0:
            say("  maturity must be positive.")
            continue
        if T > chain.T.max() * 2:
            say(f"  ! {T}y is more than twice the longest fitted expiry "
                f"({chain.T.max():.3f}y). The surface will extrapolate; the")
            say("    answer degrades to the SSVI prior rather than to noise, but")
            say("    it is not information the market gave you.")

        F, _ = chain.forward_at(T)
        raw = ask(f"  strike (blank = ladder around the forward {F:.2f}, q = quit): ")
        if raw.lower() in ("q", "quit", "exit"):
            break

        if raw == "":
            strikes = default_ladder(chain, T)
        else:
            try:
                strikes = [float(x.replace(",", ".")) for x in raw.split()]
            except ValueError:
                say(f"  could not read '{raw}' as strikes -- give numbers "
                    f"separated by spaces.")
                continue
            if any(K <= 0 for K in strikes):
                say("  strikes must be positive.")
                continue

        say()
        say(quote_table(surface, chain, strikes, T))

        if show:
            plot_quote(surface, chain, float(strikes[len(strikes) // 2]), T, show=True)

    say()
    say("  done.")
    return 0


# ── entry point ──────────────────────────────────────────────────────────────

def main(cfg: RunConfig | None = None) -> int:
    # Command-line arguments win over `config.py`, and `config.py` wins over
    # nothing -- with neither, the menu asks.
    if cfg is None:
        cfg, _ = config_from_args()
    mode = choose_mode(cfg)
    if cfg.chain_file is None:
        cfg.ticker = choose_ticker(cfg, mode)

    # Pick the backend once, before anything imports pyplot: an interactive run
    # wants windows, a scripted one must not block waiting for someone to close
    # them.
    import matplotlib
    if not (cfg.show_plots and interactive()):
        matplotlib.use("Agg")

    if mode == "price":
        return run_price(cfg)
    if mode == "compare":
        return run_compare(cfg)
    if mode == "sweep":
        return run_sweep(cfg)
    return run_train(cfg)


if __name__ == "__main__":
    sys.exit(main())
