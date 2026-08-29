from __future__ import annotations

from dataclasses import replace

from structured.phoenix_autocall import PhoenixAutocallSpec, price_phoenix_autocall
from structured.shocks import with_spot_shock

STANDARD_STRESSES = [
    {"name": "Spot -20%, Vol +40%", "spot_mult": 0.80, "vol_mult": 1.40, "rate_shift": 0.0},
    {"name": "Spot -10%, Vol +20%", "spot_mult": 0.90, "vol_mult": 1.20, "rate_shift": 0.0},
    {"name": "Spot flat, Vol -20%", "spot_mult": 1.00, "vol_mult": 0.80, "rate_shift": 0.0},
    {"name": "Spot +10%, Vol -10%", "spot_mult": 1.10, "vol_mult": 0.90, "rate_shift": 0.0},
    {"name": "Rates +100 bps", "spot_mult": 1.00, "vol_mult": 1.00, "rate_shift": 0.01},
    {"name": "Rates -100 bps", "spot_mult": 1.00, "vol_mult": 1.00, "rate_shift": -0.01},
]


def run_stress_scenarios(spec: PhoenixAutocallSpec, scenarios: list[dict] | None = None) -> list[dict]:
    scenarios = scenarios or STANDARD_STRESSES
    base_spec = replace(spec, n_paths=min(spec.n_paths, 20_000), return_paths=False, return_distributions=False)
    base = price_phoenix_autocall(base_spec)["fair_value_pct"]
    out = []
    for sc in scenarios:
        shocked_spot = spec.spot * float(sc.get("spot_mult", 1.0))
        shocked = replace(
            with_spot_shock(base_spec, shocked_spot),
            volatility=max(1e-8, spec.volatility * float(sc.get("vol_mult", 1.0))),
            risk_free_rate=spec.risk_free_rate + float(sc.get("rate_shift", 0.0)),
            seed=spec.seed,
        )
        price = price_phoenix_autocall(shocked)["fair_value_pct"]
        out.append({"scenario": sc["name"], "fair_value_pct": price, "delta_vs_base_pct": price - base})
    return out
