from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Any

import numpy as np

from structured.gbm import simulate_gbm_paths


@dataclass
class PhoenixAutocallSpec:
    underlying: str = "AAPL"
    spot: float = 175.0
    notional: float = 1_000_000.0
    maturity_years: float = 1.0
    coupon_pa: float = 0.08
    coupon_frequency: int = 4
    coupon_barrier: float = 0.70
    capital_barrier: float = 0.70
    autocall_level: float = 1.00
    autocall_start: float = 0.25
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.0
    volatility: float = 0.28
    n_paths: int = 50_000
    steps_per_year: int = 252
    memory: bool = True
    seed: int = 42
    return_paths: bool = False
    return_distributions: bool = True
    objectives: dict[str, Any] = field(default_factory=dict)
    observation_dates: list[float] | None = None

    @property
    def obs_dates(self) -> list[float]:
        if self.observation_dates:
            return sorted(float(d) for d in self.observation_dates if 0 < float(d) <= self.maturity_years + 1e-12)
        n_obs = max(1, int(round(self.maturity_years * self.coupon_frequency)))
        dates = [(i + 1) / self.coupon_frequency for i in range(n_obs)]
        return [d for d in dates if d >= self.autocall_start - 1e-12 and d <= self.maturity_years + 1e-12]

    def validate(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.notional <= 0:
            raise ValueError("notional must be positive")
        if not (0.0 <= self.coupon_barrier <= 1.5):
            raise ValueError("coupon_barrier should be expressed as a spot multiple, e.g. 0.70")
        if not (0.0 <= self.capital_barrier <= 1.5):
            raise ValueError("capital_barrier should be expressed as a spot multiple, e.g. 0.70")
        if self.volatility < 0:
            raise ValueError("volatility cannot be negative")
        if self.n_paths < 100:
            raise ValueError("n_paths should be at least 100 for a meaningful MC run")


def _obs_indices(spec: PhoenixAutocallSpec, n_steps: int) -> list[int]:
    indices = []
    for t in spec.obs_dates:
        idx = int(round(t / spec.maturity_years * n_steps))
        indices.append(max(1, min(n_steps, idx)))
    return sorted(set(indices))


def price_phoenix_autocall(spec: PhoenixAutocallSpec) -> dict[str, Any]:
    """Price a single-name Phoenix Autocall with GBM Monte Carlo.

    Conventions are simplified and intended for prototyping:
    - discrete coupon/autocall observations;
    - final capital barrier payoff if the product has not autocalled;
    - daily barrier touch probability is reported as a risk metric;
    - price is returned as both currency PV and percentage of notional.
    """
    spec.validate()
    n_steps = max(1, int(round(spec.maturity_years * spec.steps_per_year)))
    paths = simulate_gbm_paths(
        spot=spec.spot,
        rate=spec.risk_free_rate,
        volatility=spec.volatility,
        maturity_years=spec.maturity_years,
        n_paths=spec.n_paths,
        n_steps=n_steps,
        dividend_yield=spec.dividend_yield,
        seed=spec.seed,
    )
    obs_idx = _obs_indices(spec, n_steps)
    obs_times = np.array([i / n_steps * spec.maturity_years for i in obs_idx])
    coupon_per_period = spec.coupon_pa / spec.coupon_frequency
    disc_cf = np.zeros(spec.n_paths)
    coupon_paid = np.zeros(spec.n_paths)
    redeemed_time = np.full(spec.n_paths, spec.maturity_years)
    autocalled = np.zeros(spec.n_paths, dtype=bool)
    memory_count = np.zeros(spec.n_paths, dtype=int)
    redemption_by_obs: dict[str, int] = {f"{t:.4f}": 0 for t in obs_times}
    coupon_by_obs: dict[str, float] = {f"{t:.4f}": 0.0 for t in obs_times}

    coupon_level = spec.coupon_barrier * spec.spot
    capital_level = spec.capital_barrier * spec.spot
    autocall_level = spec.autocall_level * spec.spot

    for idx, t in zip(obs_idx, obs_times):
        alive = ~autocalled
        if not alive.any():
            break
        s_obs = paths[:, idx]
        coupon_ok = alive & (s_obs >= coupon_level)
        # Pay current coupon plus accumulated missed coupons if memory is enabled.
        if spec.memory:
            coupon_multiplier = 1 + memory_count
        else:
            coupon_multiplier = np.ones_like(memory_count)
        coupon_amount = spec.notional * coupon_per_period * coupon_multiplier
        paid_now = np.where(coupon_ok, coupon_amount, 0.0)
        disc = math.exp(-spec.risk_free_rate * float(t))
        disc_cf += paid_now * disc
        coupon_paid += paid_now
        coupon_by_obs[f"{t:.4f}"] = float(paid_now.sum() / spec.n_paths / spec.notional)
        memory_count = np.where(alive & coupon_ok, 0, memory_count)
        memory_count = np.where(alive & ~coupon_ok, memory_count + 1, memory_count)

        # Autocall can happen on observation dates except final is treated as maturity redemption.
        is_final_obs = idx == n_steps or abs(float(t) - spec.maturity_years) < 1e-9
        call_now = alive & (s_obs >= autocall_level) & (not is_final_obs)
        if call_now.any():
            disc_cf += spec.notional * call_now.astype(float) * disc
            autocalled |= call_now
            redeemed_time = np.where(call_now, float(t), redeemed_time)
            redemption_by_obs[f"{t:.4f}"] = int(call_now.sum())

    # Final redemption for paths that are still alive.
    alive = ~autocalled
    s_t = paths[:, -1]
    final_redemption = np.zeros(spec.n_paths)
    protected = alive & (s_t >= capital_level)
    loss = alive & (s_t < capital_level)
    final_redemption = np.where(protected, spec.notional, final_redemption)
    final_redemption = np.where(loss, spec.notional * (s_t / spec.spot), final_redemption)
    disc_final = math.exp(-spec.risk_free_rate * spec.maturity_years)
    disc_cf += final_redemption * disc_final
    redemption_by_obs[f"{spec.maturity_years:.4f}"] = int(alive.sum())

    price = float(disc_cf.mean())
    fair_value_pct = 100.0 * price / spec.notional
    std_error_pct = 100.0 * float(disc_cf.std(ddof=1) / math.sqrt(spec.n_paths)) / spec.notional
    ci_95_pct = 1.96 * std_error_pct

    barrier_touched = (paths.min(axis=1) <= capital_level)
    final_loss = s_t < capital_level
    terminal_return = s_t / spec.spot - 1.0

    result = {
        "product_type": "phoenix_autocall",
        "underlying": spec.underlying,
        "fair_value": price,
        "fair_value_pct": fair_value_pct,
        "mc_std_error_pct": std_error_pct,
        "mc_confidence_95_pct": ci_95_pct,
        "autocall_probability": float(autocalled.mean()),
        "barrier_touch_probability": float(barrier_touched.mean()),
        "prob_final_capital_loss": float(final_loss.mean()),
        "expected_coupon_pct": 100.0 * float(coupon_paid.mean() / spec.notional),
        "expected_coupon_pa": 100.0 * float(coupon_paid.mean() / spec.notional / max(float(redeemed_time.mean()), 1e-9)),
        "expected_life_years": float(redeemed_time.mean()),
        "terminal_return_mean": float(terminal_return.mean()),
        "terminal_return_p05": float(np.quantile(terminal_return, 0.05)),
        "terminal_return_p50": float(np.quantile(terminal_return, 0.50)),
        "terminal_return_p95": float(np.quantile(terminal_return, 0.95)),
        "event_probabilities": {
            "redemption_by_obs_date": {k: v / spec.n_paths for k, v in redemption_by_obs.items()},
            "coupon_paid_by_obs_date_pct": {k: 100.0 * v for k, v in coupon_by_obs.items()},
        },
        "cashflows": [
            {
                "time": float(t),
                "event": "coupon and autocall observation" if t < spec.maturity_years else "final coupon and redemption",
                "expected_coupon_pct": float(coupon_by_obs.get(f"{t:.4f}", 0.0) * 100.0),
                "redemption_probability": float(redemption_by_obs.get(f"{t:.4f}", 0) / spec.n_paths),
            }
            for t in obs_times
        ],
        "terms": asdict(spec) | {"observation_dates": spec.obs_dates},
    }
    if spec.return_distributions:
        result["terminal_distribution"] = {
            "sample": [float(x) for x in s_t[: min(500, len(s_t))]],
            "quantiles": {str(q): float(np.quantile(s_t, q)) for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]},
        }
    if spec.return_paths:
        result["paths"] = paths[: min(150, spec.n_paths), :].round(6).tolist()
    return result


def spec_from_dict(data: dict[str, Any]) -> PhoenixAutocallSpec:
    valid = {f.name for f in PhoenixAutocallSpec.__dataclass_fields__.values()}
    aliases = {
        "coupon_rate": "coupon_pa",
        "barrier": "capital_barrier",
        "ki_barrier": "capital_barrier",
        "autocall_trigger": "autocall_level",
        "paths": "n_paths",
        "n_sims": "n_paths",
        "S": "spot",
        "r": "risk_free_rate",
        "sigma": "volatility",
        "T": "maturity_years",
    }
    normalized = {}
    for k, v in data.items():
        normalized[aliases.get(k, k)] = v
    if "coupon_barrier" not in normalized and "capital_barrier" in normalized:
        normalized["coupon_barrier"] = normalized["capital_barrier"]
    filtered = {k: v for k, v in normalized.items() if k in valid}
    return PhoenixAutocallSpec(**filtered)
