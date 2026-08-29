from __future__ import annotations

from dataclasses import replace
from typing import Any

from structured.phoenix_autocall import PhoenixAutocallSpec, price_phoenix_autocall
from structured.shocks import with_spot_shock


def finite_difference_greeks(spec: PhoenixAutocallSpec, bump_spot: float = 0.01, bump_vol: float = 0.01, bump_rate: float = 0.0001) -> dict[str, float]:
    """Finite-difference Greeks in price-percent units.

    Delta/Gamma are based on spot relative bump. Vega is per 1.00 volatility point,
    so multiply by 0.01 for a 1 vol-point move. Rho is per 1.00 rate point.
    """
    base_paths = min(spec.n_paths, 20_000)
    base_spec = replace(spec, n_paths=base_paths, return_paths=False, return_distributions=False)
    s_up = replace(with_spot_shock(base_spec, spec.spot * (1 + bump_spot)), seed=spec.seed)
    s_dn = replace(with_spot_shock(base_spec, spec.spot * (1 - bump_spot)), seed=spec.seed)
    v_up = replace(base_spec, volatility=spec.volatility + bump_vol, seed=spec.seed)
    v_dn = replace(base_spec, volatility=max(1e-8, spec.volatility - bump_vol), seed=spec.seed)
    r_up = replace(base_spec, risk_free_rate=spec.risk_free_rate + bump_rate, seed=spec.seed)
    r_dn = replace(base_spec, risk_free_rate=spec.risk_free_rate - bump_rate, seed=spec.seed)
    t_dn = replace(base_spec, maturity_years=max(1 / 252, spec.maturity_years - 1 / 252), seed=spec.seed)

    p0 = price_phoenix_autocall(base_spec)["fair_value_pct"]
    pu = price_phoenix_autocall(s_up)["fair_value_pct"]
    pd = price_phoenix_autocall(s_dn)["fair_value_pct"]
    vu = price_phoenix_autocall(v_up)["fair_value_pct"]
    vd = price_phoenix_autocall(v_dn)["fair_value_pct"]
    ru = price_phoenix_autocall(r_up)["fair_value_pct"]
    rd = price_phoenix_autocall(r_dn)["fair_value_pct"]
    td = price_phoenix_autocall(t_dn)["fair_value_pct"]

    dS = spec.spot * bump_spot
    return {
        "delta": (pu - pd) / (2 * dS),
        "gamma": (pu - 2 * p0 + pd) / (dS**2),
        "vega": (vu - vd) / (2 * bump_vol),
        "rho": (ru - rd) / (2 * bump_rate),
        "theta_1d": td - p0,
    }
