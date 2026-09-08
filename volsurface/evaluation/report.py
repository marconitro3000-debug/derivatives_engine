"""
volsurface/evaluation/report.py
Scoring the neural surface against the parametric baselines, on equal terms.

Every surface -- per-slice SVI, joint SSVI, neural -- implements
`volsurface.surfaces.base.VolSurface`, so the same two reports apply to all of them:

* `volsurface.evaluation.diagnostics.fit_report`: how well it reproduces the quotes it was fitted
  to, in vol space and in price space.
* `volsurface.evaluation.diagnostics.scan_arbitrage`: whether the surface it defines *between*
  those quotes admits static arbitrage.

Reporting both together is the point of the project. Per-slice SVI wins the
first and loses the second; SSVI does the reverse; the penalised neural surface
is built to take the first without giving up the second. A comparison that
showed only RMSE would make the wrong model look best.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volsurface.evaluation.diagnostics import ArbitrageReport, FitReport, fit_report, scan_arbitrage
from volsurface.surfaces.base import VolSurface


@dataclass
class SurfaceScore:
    """One row of the comparison table.

    Carries the fitted `surface` itself, not just its numbers: calibrating the
    per-slice SVI baseline is the slowest step in a run, and a caller that wants
    to plot it afterwards should not have to fit it a second time.
    """

    name: str
    surface: VolSurface
    fit: FitReport
    arb: ArbitrageReport
    val_fit: FitReport | None = None


def evaluate_surface(surface: VolSurface, snapshot, val_snapshot=None,
                     **scan_kwargs) -> SurfaceScore:
    """Fit quality plus an arbitrage scan over the chain's own (k, T) range.

    The scan region is taken from the data and then widened, so a surface is
    also judged on the region it interpolates and extrapolates into -- not only
    where quotes pinned it down.
    """
    k_lo, k_hi = float(snapshot.k.min()), float(snapshot.k.max())
    span = max(k_hi - k_lo, 1e-3)
    defaults = dict(
        k_range=(k_lo - 0.2 * span, k_hi + 0.2 * span),
        T_range=(max(float(snapshot.T.min()) * 0.5, 1e-3), float(snapshot.T.max()) * 1.2),
    )
    defaults.update(scan_kwargs)

    return SurfaceScore(
        name=surface.name,
        surface=surface,
        fit=fit_report(surface, snapshot),
        arb=scan_arbitrage(surface, **defaults),
        val_fit=fit_report(surface, val_snapshot) if val_snapshot is not None else None,
    )


def comparison_table(scores: list[SurfaceScore]) -> str:
    """Render the scores as a fixed-width table for the terminal / README."""
    header = (
        f"{'surface':<34} {'IV RMSE':>9} {'max err':>9} {'px RMSE':>9} "
        f"{'in spread':>10} {'cal viol':>9} {'bfly viol':>10}"
    )
    lines = [header, "-" * len(header)]
    for s in scores:
        lines.append(
            f"{s.name:<34} "
            f"{s.fit.rmse_vol_bps:>8.1f}bp "
            f"{s.fit.max_vol_bps:>8.1f}bp "
            f"{s.fit.rmse_price:>9.4f} "
            f"{s.fit.pct_inside_spread:>9.1f}% "
            f"{s.arb.calendar_pct:>8.2f}% "
            f"{s.arb.butterfly_pct:>9.2f}%"
        )
    return "\n".join(lines)


def worst_quote_notes(scores: list[SurfaceScore]) -> str:
    """Where each surface's ``max err`` actually sits, and at what weight.

    The table's worst-single-quote column is the one number in it that is
    routinely misread. On a real chain it lands on a deep wing quote carrying a
    fitting weight near zero -- the vega/spread weighting deliberately declining
    to chase a vol that its own price barely identifies. Printed bare it looks
    like a blow-up; printed with its coordinates and its weight it reads as what
    it is.
    """
    lines = []
    for s in scores:
        note = s.fit.worst_quote_note()
        if note:
            lines.append(f"  {s.name:<34} {note}")
    return "\n".join(lines)


def compare(snapshot, result, include_baselines: bool = True,
            extra_surfaces: tuple = ()) -> list[SurfaceScore]:
    """Score the trained neural surface next to the parametric baselines.

    `extra_surfaces` are scored straight after it and before the baselines --
    that is where the flat-prior ablation goes, so the table reads "network with
    the prior / network without it / prior alone".

    A baseline that fails to calibrate (too few quotes on some expiry, a
    degenerate smile) is skipped with a note rather than aborting the run --
    losing the SVI column should not cost you the neural result.
    """
    from volsurface.surfaces.svi import SSVISurface, SVISliceSurface

    scores = [evaluate_surface(result.model, snapshot)]
    scores += [evaluate_surface(s, snapshot) for s in extra_surfaces]

    if include_baselines:
        for name, factory in (("SVI (per-slice)", SVISliceSurface.fit),
                              ("SSVI (joint)", SSVISurface.fit)):
            try:
                scores.append(evaluate_surface(factory(snapshot), snapshot))
            except Exception as exc:                      # noqa: BLE001 - reported
                print(f"  [skip] {name} did not calibrate: {exc}")

    return scores


# -- plots --------------------------------------------------------------------

def plot_fit(snapshot, surfaces: list[VolSurface], path: str | None = None,
             max_slices: int = 6, show: bool = False):
    """Market smiles with each surface overlaid, one panel per expiry.

    The model curves are drawn over a *wider* strike range than the quotes so
    the wings -- where the surfaces disagree and where arbitrage appears -- are
    visible rather than cropped out.
    """
    import matplotlib.pyplot as plt

    maturities = snapshot.maturities
    if len(maturities) > max_slices:
        maturities = maturities[np.linspace(0, len(maturities) - 1, max_slices).astype(int)]

    n = len(maturities)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)

    k_lo, k_hi = snapshot.k.min(), snapshot.k.max()
    pad = 0.25 * (k_hi - k_lo)
    k_plot = np.linspace(k_lo - pad, k_hi + pad, 200)

    for ax, T in zip(axes.ravel(), maturities):
        sl = snapshot.slice_at(T)
        ax.scatter(sl.k, sl.iv * 100, s=14, c="black", zorder=3, label="market")
        for surf in surfaces:
            ax.plot(k_plot, surf.implied_vol(k_plot, np.full_like(k_plot, T)) * 100,
                    lw=1.6, label=surf.name)
        ax.axvspan(k_lo - pad, sl.k.min(), color="grey", alpha=0.08)
        ax.axvspan(sl.k.max(), k_hi + pad, color="grey", alpha=0.08)
        ax.set_title(f"T = {T:.3f}y", fontsize=10)
        ax.set_xlabel("log-moneyness k")
        ax.set_ylabel("implied vol (%)")
        ax.grid(alpha=0.25)

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"{snapshot.ticker} implied vol smiles @ {snapshot.asof} "
                 f"(shaded = outside quoted strikes)", fontsize=11)
    fig.tight_layout()

    return _finish(fig, path, show)


def plot_arbitrage_map(surface: VolSurface, snapshot, path: str | None = None,
                       show: bool = False):
    """Heat map of Durrleman's ``g`` and of ``dw/dT`` over the (k, T) plane.

    Negative regions are the arbitrage; the quoted points are overlaid so it is
    immediately clear whether the violations sit inside the data or out in the
    extrapolated wings, which is a completely different severity of problem.
    """
    import matplotlib.pyplot as plt

    from volsurface.evaluation.diagnostics import butterfly_g

    k_lo, k_hi = snapshot.k.min(), snapshot.k.max()
    pad = 0.25 * (k_hi - k_lo)
    kk, TT = np.meshgrid(
        np.linspace(k_lo - pad, k_hi + pad, 160),
        np.linspace(max(snapshot.T.min() * 0.5, 1e-3), snapshot.T.max() * 1.2, 120),
        indexing="ij",
    )
    g = butterfly_g(surface, kk.ravel(), TT.ravel()).reshape(kk.shape)
    dwdT = surface.dw_dT(kk.ravel(), TT.ravel()).reshape(kk.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, field, title in ((axes[0], g, "butterfly:  g(k,T)"),
                             (axes[1], dwdT, "calendar:  dw/dT")):
        worst = float(np.nanmin(field))
        n_bad = int((field < 0).sum())

        # Scale the colour map to the *violation*, not to the global range. A
        # surface can be 20 units positive in the middle and -0.001 in one
        # corner; a symmetric scale over the full range renders that corner
        # white and hides the only thing the plot exists to show.
        lim = 3.0 * abs(worst) if worst < 0 else float(np.nanmax(np.abs(field)))
        lim = lim or 1.0

        im = ax.pcolormesh(kk, TT, field, cmap="RdBu", vmin=-lim, vmax=lim,
                           shading="auto")
        ax.contour(kk, TT, field, levels=[0.0], colors="black", linewidths=1.2)
        ax.scatter(snapshot.k, snapshot.T, s=4, c="black", alpha=0.35)
        flag = ("no violations" if not n_bad else
                f"{n_bad} violation{'s' if n_bad > 1 else ''}, worst {worst:.2e}")
        ax.set_title(f"{title}   ({flag})", fontsize=10)
        ax.set_xlabel("log-moneyness k")
        ax.set_ylabel("maturity T (years)")
        fig.colorbar(im, ax=ax, extend="max" if worst < 0 else "neither")

    fig.suptitle(f"{surface.name} — static arbitrage map", fontsize=11)
    fig.tight_layout()

    return _finish(fig, path, show)


def _finish(fig, path, show):
    """Save, display, or hand back the figure -- in whatever combination is wanted.

    A run that both writes a report and is being watched wants the figure on
    screen *and* on disk, so these are independent rather than an either/or.
    """
    import matplotlib.pyplot as plt

    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    elif path:
        plt.close(fig)
    return path if (path and not show) else fig


def plot_training(result, path: str | None = None, show: bool = False):
    """The four curves that say whether a run worked and will hold up.

    **Fit** -- training and validation IV error, with the selected epoch marked.
    The held-out strikes span the whole smile, wings included, so validation
    sitting a little above training is the expected picture; validation *below*
    training is the signature of a split that is quietly holding out only the
    easy points.

    **Generalisation gap** -- ``val - train`` explicitly, filled, because a gap
    that is small and flat is a different story from one that starts small and
    widens. The second shape is the model starting to fit the training strikes
    at the validation strikes' expense -- overfitting -- which is why validation
    error, not training error, selects the epoch (the dotted line): the selected
    epoch is where this gap is not yet a problem.

    **Worst violation** -- the smallest ``dw/dT`` and Durrleman ``g`` found on
    that epoch's collocation points. This is the curve that says whether the
    constraints are doing anything: it should cross into positive territory
    during the penalty warm-up and stay there while the fit error keeps falling.
    Hovering at zero means the constraints are binding and distorting the fit;
    deeply positive from the start means they are doing nothing.

    **Penalty terms** -- on a log scale, since a working run drives them to
    exactly zero and a linear axis would show a flat line at the bottom.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    h = result.history
    ep = np.arange(len(h["train_rmse_bps"]))
    train, val = np.asarray(h["train_rmse_bps"]), np.asarray(h["val_rmse_bps"])
    gap = val - train

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 3.8))

    axes[0].plot(ep, train, lw=1.1, label="train")
    axes[0].plot(ep, val, lw=1.1, label="validation")
    axes[0].axvline(result.best_epoch, color="k", ls=":", lw=0.9,
                    label=f"selected ({result.best_epoch})")
    axes[0].set_ylabel("IV RMSE (bp)")
    axes[0].set_title("Fit error", fontsize=10)

    axes[1].plot(ep, gap, lw=1.1, color="tab:purple")
    axes[1].fill_between(ep, 0, gap, alpha=0.15, color="tab:purple")
    axes[1].axhline(0.0, color="k", lw=0.7, ls="-")
    axes[1].axvline(result.best_epoch, color="k", ls=":", lw=0.9)
    axes[1].scatter([result.best_epoch], [gap[result.best_epoch]], color="k",
                    zorder=4, s=22,
                    label=f"at selection: {gap[result.best_epoch]:+.1f}bp")
    axes[1].set_ylabel("val - train (bp)")
    axes[1].set_title("Generalisation gap", fontsize=10)

    axes[2].plot(ep, h["worst_calendar"], lw=1.1, label="min  dw/dT")
    axes[2].plot(ep, h["worst_butterfly"], lw=1.1, label="min  g")
    axes[2].axhline(0.0, color="crimson", lw=0.9, ls="--")
    axes[2].set_title("Worst violation on collocation points", fontsize=10)

    axes[3].semilogy(ep, np.maximum(h["pen_calendar"], 1e-20), lw=1.1, label="calendar")
    axes[3].semilogy(ep, np.maximum(h["pen_butterfly"], 1e-20), lw=1.1, label="butterfly")
    axes[3].set_title("Penalty terms", fontsize=10)

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Training history - {len(ep)} epochs, best validation "
        f"{result.best_val_rmse_bps:.1f}bp, gap at selection "
        f"{gap[result.best_epoch]:+.1f}bp", fontsize=11)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_model_comparison(records, path: str | None = None, show: bool = False):
    """Validation error and generalisation gap across every archived run.

    Reads straight off `volsurface.evaluation.registry.RunRecord`, so it works whether the
    runs came from the same session or from `models/` accumulated over weeks of
    experiments -- the point of archiving every run instead of overwriting the
    last one.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not records:
        raise ValueError("no runs to compare")

    ranked = sorted(records, key=lambda r: r.val_rmse_bps)
    names = [r.run_id for r in ranked]
    val = np.array([r.val_rmse_bps for r in ranked])
    train = np.array([r.train_rmse_bps for r in ranked])
    gap = np.array([r.generalization_gap_bps for r in ranked])

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.1 * len(ranked)), 4.2))
    y = np.arange(len(ranked))

    axes[0].barh(y, train, height=0.35, label="train", alpha=0.85)
    axes[0].barh(y + 0.35, val, height=0.35, label="validation", alpha=0.85)
    axes[0].set_yticks(y + 0.175, names, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("IV RMSE (bp)")
    axes[0].set_title("Fit -- best (top) to worst", fontsize=10)
    axes[0].legend(fontsize=8)

    colors = ["tab:red" if g > 15 else "tab:purple" for g in gap]
    axes[1].barh(y, gap, height=0.5, color=colors, alpha=0.85)
    axes[1].axvline(0, color="k", lw=0.7)
    axes[1].set_yticks(y, names, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("val - train (bp)")
    axes[1].set_title("Generalisation gap  (red > 15bp)", fontsize=10)

    for ax in axes:
        ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_quote(surface: VolSurface, snapshot, strike: float, T: float,
               path: str | None = None, show: bool = False):
    """The smile at maturity ``T`` with one queried strike marked on it.

    Shows the queried point against the *quotes* nearest that maturity, so it is
    immediately obvious whether the answer is an interpolation between real
    market data or an extrapolation the model invented.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    F, _ = snapshot.forward_at(T)
    k_q = float(np.log(strike / F))

    nearest = snapshot.maturities[int(np.argmin(np.abs(snapshot.maturities - T)))]
    sl = snapshot.slice_at(nearest)
    order = np.argsort(sl.k)

    lo, hi = snapshot.k.min(), snapshot.k.max()
    pad = 0.2 * (hi - lo)
    grid = np.linspace(min(lo - pad, k_q - 0.05), max(hi + pad, k_q + 0.05), 300)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(sl.k[order], sl.iv[order] * 100, s=16, c="black", zorder=3,
               label=f"quotes at T = {nearest:.3f}y")
    ax.plot(grid, surface.implied_vol(grid, np.full_like(grid, T)) * 100,
            lw=1.8, color="tab:blue", label=f"{surface.name}, T = {T:.3f}y")

    iv_q = float(surface.implied_vol(np.array([k_q]), np.array([T]))[0])
    ax.plot([k_q], [iv_q * 100], "*", ms=16, color="crimson", zorder=4,
            label=f"K = {strike:g}  ->  {iv_q:.2%}")

    inside = sl.k.min() <= k_q <= sl.k.max()
    ax.axvspan(grid[0], sl.k.min(), color="grey", alpha=0.08)
    ax.axvspan(sl.k.max(), grid[-1], color="grey", alpha=0.08)
    ax.set_xlabel("log-moneyness  k = log(K / F)")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title(f"{snapshot.ticker}  K = {strike:g}, T = {T:.3f}y  "
                 f"({'interpolated' if inside else 'EXTRAPOLATED'})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _finish(fig, path, show)


def capacity_table(rows: list[dict], reference_spec: str | None = None) -> str:
    """The architecture sweep as a table, measured against a reference arm.

    Reported as a *difference* rather than in isolation, because the number that
    matters is not "512x4 scores X" but "512x4 buys you X basis points over the
    configured default for Y times the parameters". With no `reference_spec`, or
    one that was not swept, the best arm becomes the reference -- the caller
    knows which architecture is configured, this function does not.
    """
    import numpy as np

    by_spec: dict[str, list[dict]] = {}
    for r in rows:
        by_spec.setdefault(r["spec"], []).append(r)

    if reference_spec in by_spec:
        ref = float(np.mean([r["val"] for r in by_spec[reference_spec]]))
    else:
        ref = min(float(np.mean([r["val"] for r in v])) for v in by_spec.values())

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
    """Validation error against parameter count, log x.

    The shape is the whole argument: it flattens almost immediately, which is
    what a problem whose error floor is the bid-ask rather than the model looks
    like. Training error is drawn alongside because if capacity were the binding
    constraint that curve would be heading for zero, and it does not.
    """
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
    return _finish(fig, path, show)


def plot_run_overlay(records, path: str | None = None, show: bool = False):
    """Every archived run's validation curve on one axis.

    `plot_model_comparison` ranks the runs by their final numbers; this shows
    how they got there, which is where a run that converged smoothly is
    distinguishable from one that got lucky on the epoch it happened to select.
    """
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("no runs to overlay")

    from .registry import load_history

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
    return _finish(fig, path, show)
