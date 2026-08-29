from __future__ import annotations

from dataclasses import replace

from structured.phoenix_autocall import PhoenixAutocallSpec, price_phoenix_autocall
from structured.shocks import with_spot_shock


def spot_vol_heatmap(
    spec: PhoenixAutocallSpec,
    spot_multipliers: list[float] | None = None,
    vol_levels: list[float] | None = None,
) -> dict:
    spot_multipliers = spot_multipliers or [0.75, 0.85, 0.95, 1.00, 1.05, 1.15, 1.25]
    vol_levels = vol_levels or [0.15, 0.20, 0.25, 0.30, 0.35, 0.45]
    grid = []
    base = replace(spec, n_paths=min(spec.n_paths, 10_000), return_paths=False, return_distributions=False)
    for vol in vol_levels:
        row = []
        for sm in spot_multipliers:
            shocked = replace(with_spot_shock(base, spec.spot * sm), volatility=vol, seed=spec.seed)
            row.append(price_phoenix_autocall(shocked)["fair_value_pct"])
        grid.append(row)
    return {"spot_multipliers": spot_multipliers, "vol_levels": vol_levels, "fair_value_pct": grid}
