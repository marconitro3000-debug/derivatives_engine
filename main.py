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

No command-line arguments. Defaults live in `config.py`; set `mode` there to
"train" or "price" to skip the menu entirely.
"""

from __future__ import annotations

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

from config import CONFIG, RunConfig
from volsurface import (
    ChainSnapshot,
    ModelConfig,
    NeuralVolSurface,
    PenaltyWeights,
    TrainConfig,
    compare,
    comparison_table,
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
    if cfg.mode in ("train", "price", "compare"):
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
    print()
    choice = ask("  choose [1]: ", "1")
    return {
        "2": "price", "price": "price", "p": "price",
        "3": "compare", "compare": "compare", "c": "compare",
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
                slug = score.name.split()[0].lower()
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

def main(cfg: RunConfig = CONFIG) -> int:
    mode = choose_mode(cfg)
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
    return run_train(cfg)


if __name__ == "__main__":
    sys.exit(main())
