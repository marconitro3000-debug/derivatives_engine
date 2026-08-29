from __future__ import annotations

from structured.phoenix_autocall import PhoenixAutocallSpec


def validate_phoenix_spec(spec: PhoenixAutocallSpec) -> dict:
    checks: list[dict] = []

    def add(name: str, status: str, message: str, weight: int = 10):
        checks.append({"name": name, "status": status, "message": message, "weight": weight})

    add("underlying", "ok" if spec.underlying else "error", "Underlying is defined." if spec.underlying else "Missing underlying.")
    add("coupon", "ok" if 0 < spec.coupon_pa <= 0.30 else "warning", "Coupon level is within prototype bounds.")
    add("barrier", "ok" if 0.5 <= spec.capital_barrier <= 0.85 else "warning", "Barrier is in a standard indicative range." if 0.5 <= spec.capital_barrier <= 0.85 else "Barrier is unusually aggressive/conservative.")
    add("autocall", "ok" if 0.8 <= spec.autocall_level <= 1.2 else "warning", "Autocall trigger is plausible.")
    add("volatility", "ok" if 0.05 <= spec.volatility <= 0.80 else "warning", "Volatility is plausible for the prototype range.")
    add("paths", "ok" if spec.n_paths >= 10_000 else "warning", "Monte Carlo path count is acceptable." if spec.n_paths >= 10_000 else "Use at least 10,000 paths for demo stability.")
    add("maturity", "ok" if 0.25 <= spec.maturity_years <= 5 else "warning", "Maturity is within supported bounds.")
    if spec.objectives:
        target = spec.objectives.get("target_coupon") or spec.objectives.get("target_coupon_pa")
        add("objectives", "ok", f"Design objectives provided. Target coupon={target}.", weight=15)
    else:
        add("objectives", "warning", "No explicit design objectives supplied.", weight=15)

    max_score = sum(c["weight"] for c in checks)
    earned = 0
    for c in checks:
        if c["status"] == "ok":
            earned += c["weight"]
        elif c["status"] == "warning":
            earned += c["weight"] * 0.55
    score = round(100 * earned / max_score, 1) if max_score else 0
    return {
        "score": score,
        "status": "ok" if score >= 80 else "warning" if score >= 60 else "error",
        "checks": checks,
    }
