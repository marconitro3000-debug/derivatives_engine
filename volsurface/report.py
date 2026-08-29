"""
volsurface/report.py
Scoring the neural surface against the parametric baselines, on equal terms.

Every surface -- per-slice SVI, joint SSVI, neural -- implements
`volsurface.surface.VolSurface`, so the same two reports apply to all of them:

* `volsurface.diagnostics.fit_report`: how well it reproduces the quotes it was fitted
  to, in vol space and in price space.
* `volsurface.diagnostics.scan_arbitrage`: whether the surface it defines *between*
  those quotes admits static arbitrage.

Reporting both together is the point of the project. Per-slice SVI wins the
first and loses the second; SSVI does the reverse; the penalised neural surface
is built to take the first without giving up the second. A comparison that
showed only RMSE would make the wrong model look best.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volsurface.diagnostics import ArbitrageReport, FitReport, fit_report, scan_arbitrage
from volsurface.surface import VolSurface


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


def compare(snapshot, result, include_baselines: bool = True) -> list[SurfaceScore]:
    """Score the trained neural surface next to the parametric baselines.

    A baseline that fails to calibrate (too few quotes on some expiry, a
    degenerate smile) is skipped with a note rather than aborting the run --
    losing the SVI column should not cost you the neural result.
    """
    from volsurface.svi import SSVISurface, SVISliceSurface

    scores = [evaluate_surface(result.model, snapshot)]

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
             max_slices: int = 6):
    """Market smiles with each surface overlaid, one panel per expiry.

    The model curves are drawn over a *wider* strike range than the quotes so
    the wings -- where the surfaces disagree and where arbitrage appears -- are
    visible rather than cropped out.
    """
    import matplotlib
    if path:
        matplotlib.use("Agg")
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

    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    return fig


def plot_arbitrage_map(surface: VolSurface, snapshot, path: str | None = None):
    """Heat map of Durrleman's ``g`` and of ``dw/dT`` over the (k, T) plane.

    Negative regions are the arbitrage; the quoted points are overlaid so it is
    immediately clear whether the violations sit inside the data or out in the
    extrapolated wings, which is a completely different severity of problem.
    """
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from volsurface.diagnostics import butterfly_g

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

    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    return fig
