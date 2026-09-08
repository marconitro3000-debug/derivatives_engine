"""
volsurface/evaluation/registry.py
A folder of trained models you can actually compare.

`main.py`'s TRAIN mode does not overwrite last run's model when you fit again --
it archives every run under `models/<ticker>_<timestamp>/`, one directory per
fit, holding the network weights, the chain it was fitted to, the full training
history, and a `metrics.json` with the numbers that matter for picking one. That
last file is the point: comparing checkpoints by eye across report.txt files
does not scale past two or three runs, and `list_runs` / `comparison_table`
below turn the folder into one table sorted by validation error.

The `results/` directory `main.py`'s PRICE mode reads from is a separate,
always-overwritten pointer to whichever run was trained *most recently* -- fast
to query, but not the place to go looking for history. `models/` is the
archive; `results/` is the cursor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class RunRecord:
    """One archived training run: enough to compare it without reloading it."""

    run_id: str
    ticker: str
    trained_at: str            # ISO 8601
    chain_asof: str
    n_quotes: int
    n_expiries: int

    epochs_run: int
    best_epoch: int
    alpha: float
    hidden: tuple[int, ...]
    prior: str
    seed: int

    train_rmse_bps: float
    val_rmse_bps: float
    generalization_gap_bps: float
    """val - train at the selected epoch. Small and stable is the sign of a run
    that will hold up on tomorrow's chain; large or still widening is a run that
    memorised this one."""

    worst_calendar: float
    worst_butterfly: float
    elapsed_sec: float

    baseline_svi_rmse_bps: float | None
    baseline_ssvi_rmse_bps: float | None

    dir: str                   # populated on load; not written into metrics.json

    ablation_flat_rmse_bps: float | None = None
    """Same network, same budget, flat constant-vol prior instead of SSVI. The
    distance between this and `train_rmse_bps` is what the parametric prior is
    worth; the distance between it and `baseline_ssvi_rmse_bps` is what the
    network is worth. Defaulted so runs archived before the ablation existed
    still load."""

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("dir", None)
        return d

    def path(self, name: str) -> Path:
        return Path(self.dir) / name


def save_run(root: Path, cfg, chain, result, scores) -> RunRecord:
    """Archive a completed training run under ``root/<ticker>_<timestamp>/``.

    Writes the model, the chain, the full per-epoch history (for
    `plot_generalization` / overlay comparisons later) and `metrics.json`.
    """
    root = Path(root)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = f"{cfg.ticker.lower()}_{stamp}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result.model.save(run_dir / "surface.pt")
    chain.save(run_dir / "chain.npz")
    np.savez_compressed(run_dir / "history.npz",
                        **{k: np.asarray(v) for k, v in result.history.items()})

    svi = next((s for s in scores if s.name.startswith("SVI")), None)
    ssvi = next((s for s in scores if s.name.startswith("SSVI")), None)
    flat = next((s for s in scores if "flat prior" in s.name), None)

    h = result.history
    record = RunRecord(
        run_id=run_id,
        ticker=cfg.ticker.upper(),
        trained_at=datetime.now().isoformat(timespec="seconds"),
        chain_asof=str(chain.asof),
        n_quotes=len(chain),
        n_expiries=len(chain.forwards),
        epochs_run=len(h["train_rmse_bps"]),
        best_epoch=result.best_epoch,
        alpha=result.model.config.alpha,
        hidden=tuple(result.model.config.hidden),
        prior=type(result.model.prior).__name__,
        seed=cfg.seed,
        train_rmse_bps=float(h["train_rmse_bps"][result.best_epoch]),
        val_rmse_bps=float(result.best_val_rmse_bps),
        generalization_gap_bps=float(
            result.best_val_rmse_bps - h["train_rmse_bps"][result.best_epoch]
        ),
        worst_calendar=float(h["worst_calendar"][-1]),
        worst_butterfly=float(h["worst_butterfly"][-1]),
        elapsed_sec=result.elapsed_sec,
        baseline_svi_rmse_bps=svi.fit.rmse_vol_bps if svi else None,
        baseline_ssvi_rmse_bps=ssvi.fit.rmse_vol_bps if ssvi else None,
        dir=str(run_dir),
        ablation_flat_rmse_bps=flat.fit.rmse_vol_bps if flat else None,
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(record.to_json(), indent=2), encoding="utf-8"
    )
    return record


def list_runs(root: Path, ticker: str | None = None) -> list[RunRecord]:
    """Every archived run under `root`, newest first.

    Skips directories with no `metrics.json` (an interrupted save) rather than
    raising, since a broken run should not stop you from seeing the good ones.
    """
    root = Path(root)
    if not root.exists():
        return []

    records = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        meta = run_dir / "metrics.json"
        if not meta.exists():
            continue
        data = json.loads(meta.read_text(encoding="utf-8"))
        data["hidden"] = tuple(data["hidden"])
        record = RunRecord(dir=str(run_dir), **data)
        if ticker is None or record.ticker == ticker.upper():
            records.append(record)
    return records


def load_run(record: RunRecord):
    """Reload the model and chain a `RunRecord` points to."""
    from volsurface.data import ChainSnapshot
    from volsurface.surfaces.neural.model import NeuralVolSurface

    model = NeuralVolSurface.load(record.path("surface.pt"))
    chain = ChainSnapshot.load(record.path("chain.npz"))
    return model, chain


def load_history(record: RunRecord) -> dict[str, np.ndarray]:
    z = np.load(record.path("history.npz"))
    return {k: z[k] for k in z.files}


def runs_table(records: list[RunRecord]) -> str:
    """Runs ranked by validation error -- the number that decides which to use."""
    if not records:
        return "  (no archived runs)"

    ranked = sorted(records, key=lambda r: r.val_rmse_bps)
    header = (f"{'run':<22} {'trained':<17} {'quotes':>7} {'val RMSE':>9} "
             f"{'train RMSE':>11} {'gap':>7} {'epochs':>7} {'alpha':>6} {'hidden':>14}")
    lines = [header, "-" * len(header)]
    for r in ranked:
        best = " *" if r is ranked[0] else "  "
        lines.append(
            f"{r.run_id:<22} {r.trained_at[:16]:<17} {r.n_quotes:>7} "
            f"{r.val_rmse_bps:>8.1f}bp {r.train_rmse_bps:>10.1f}bp "
            f"{r.generalization_gap_bps:>+6.1f}bp {r.best_epoch:>7} "
            f"{r.alpha:>6.2f} {str(r.hidden):>14}{best}"
        )
    lines.append("")
    lines.append("  gap = val - train RMSE at the selected epoch (smaller and more"
                 " stable generalises better)")
    lines.append("  * = lowest validation error among these runs")
    if any(r.generalization_gap_bps < 0 for r in ranked):
        # Validation used to be drawn from the interior of each smile only,
        # which held out the easy quotes and produced a negative gap. Those runs
        # are not comparable with later ones on this column, and silently
        # ranking them first would recommend the wrong checkpoint.
        lines.append("  !  a negative gap means the run predates the split fix:"
                     " its validation set excluded")
        lines.append("     the wings, so its val RMSE is optimistic and not"
                     " comparable with the rest. Retrain")
        lines.append("     to rank it fairly.")
    return "\n".join(lines)
