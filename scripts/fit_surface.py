"""
scripts/fit_surface.py
End-to-end run: download a chain, fit every surface, print the comparison, plot.

    python -m scripts.fit_surface --ticker SPY
    python -m scripts.fit_surface --synthetic --plot out/
    python -m scripts.fit_surface --ticker AAPL --prior flat --epochs 3000

This is the script that produces the numbers quoted in the README, so it prints
the fit *and* the arbitrage diagnostics for all three surfaces side by side --
either one alone would be a misleading summary of the result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from marketdata import fetch_chain, synthetic_snapshot
from nn import TrainConfig, comparison_table, compare, plot_arbitrage_map, plot_fit, train_surface


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--ticker", default="SPY", help="underlying symbol")
    ap.add_argument("--synthetic", action="store_true",
                    help="use a generated arbitrage-free chain instead of live data")
    ap.add_argument("--noise-bps", type=float, default=25.0,
                    help="synthetic-only: IV noise injected into the generated quotes")
    ap.add_argument("--max-expiries", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--prior", choices=("ssvi", "flat"), default="ssvi")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", metavar="DIR", help="write smile and arbitrage plots here")
    ap.add_argument("--save", metavar="PATH", help="save the trained model to this path")
    ap.add_argument("--quiet", action="store_true", help="suppress the training log")
    args = ap.parse_args(argv)

    # -- data ----------------------------------------------------------------
    print(f"[1/4] loading chain ({'synthetic' if args.synthetic else 'yfinance'}) ...")
    try:
        snapshot = (
            synthetic_snapshot(ticker=args.ticker, noise_bps=args.noise_bps, seed=args.seed)
            if args.synthetic
            else fetch_chain(args.ticker, max_expiries=args.max_expiries)
        )
    except Exception as exc:                                # noqa: BLE001
        print(f"could not load a chain for {args.ticker}: {exc}", file=sys.stderr)
        print("try --synthetic to run without network access.", file=sys.stderr)
        return 1
    print(snapshot.summary())

    # -- fit -----------------------------------------------------------------
    print(f"\n[2/4] training neural surface ({args.prior} prior) ...")
    cfg = TrainConfig(
        epochs=args.epochs,
        seed=args.seed,
        log_every=0 if args.quiet else max(args.epochs // 10, 1),
    )
    result = train_surface(snapshot, cfg, prior=args.prior)
    print(result.summary())

    # -- score ---------------------------------------------------------------
    print("\n[3/4] scoring against the parametric baselines ...")
    scores = compare(snapshot, result)
    print(comparison_table(scores))
    print(
        "\n  IV RMSE / max err : vega-weighted implied-vol error against the quotes"
        "\n  px RMSE / in spread: the same fit measured in price space"
        "\n  cal / bfly viol   : share of a dense (k, T) grid -- extending past the"
        "\n                      quoted strikes -- admitting static arbitrage"
    )

    # -- artefacts -----------------------------------------------------------
    print("\n[4/4] artefacts ...")
    if args.save:
        result.model.save(args.save)
        print(f"  model -> {args.save}")

    if args.plot:
        out = Path(args.plot)
        out.mkdir(parents=True, exist_ok=True)

        # Reuse the already-calibrated surfaces from the scoring step rather
        # than refitting them: SVI calibration is the slowest part of a run.
        surfaces = [s.surface for s in scores]
        print(f"  {plot_fit(snapshot, surfaces, str(out / 'smiles.png'))}")

        # One arbitrage map per surface -- side by side, the contrast between
        # the penalised network and the naive per-slice fit is the result.
        for score in scores:
            slug = score.name.split()[0].lower()
            print(f"  {plot_arbitrage_map(score.surface, snapshot, str(out / f'arbitrage_{slug}.png'))}")
    else:
        print("  (none; pass --plot DIR for figures, --save PATH for the model)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
