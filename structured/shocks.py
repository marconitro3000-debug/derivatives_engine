from __future__ import annotations

from dataclasses import replace

from structured.phoenix_autocall import PhoenixAutocallSpec


def with_spot_shock(spec: PhoenixAutocallSpec, shocked_spot: float) -> PhoenixAutocallSpec:
    """Shock current spot while keeping contractual trigger/barrier levels fixed.

    Phoenix terms are entered as spot multiples at inception. For risk views,
    changing spot should move the market against fixed absolute levels rather
    than rescaling every barrier with the shocked spot.
    """
    if shocked_spot <= 0:
        raise ValueError("shocked_spot must be positive")
    return replace(
        spec,
        spot=shocked_spot,
        coupon_barrier=spec.coupon_barrier * spec.spot / shocked_spot,
        capital_barrier=spec.capital_barrier * spec.spot / shocked_spot,
        autocall_level=spec.autocall_level * spec.spot / shocked_spot,
    )
