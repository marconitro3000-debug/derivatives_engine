from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structured.greeks import finite_difference_greeks
from structured.heatmap import spot_vol_heatmap
from structured.phoenix_autocall import price_phoenix_autocall, spec_from_dict
from structured.stress import run_stress_scenarios
from structured.validation import validate_phoenix_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Price a Phoenix Autocall JSON spec.")
    parser.add_argument("spec_file")
    parser.add_argument("--paths", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = json.loads(Path(args.spec_file).read_text(encoding="utf-8"))
    if args.paths:
        payload["n_paths"] = args.paths
        payload["n_sims"] = args.paths
    spec = spec_from_dict(payload)
    result = price_phoenix_autocall(spec)
    result["greeks"] = finite_difference_greeks(spec)
    result["stress"] = run_stress_scenarios(spec)
    result["heatmap"] = spot_vol_heatmap(spec)
    result["validation"] = validate_phoenix_spec(spec)

    print(f"Phoenix Autocall {spec.underlying}")
    print(f"Fair value: {result['fair_value']:.2f} ({result['fair_value_pct']:.4f}% of notional)")
    print(f"Autocall probability: {result['autocall_probability']:.2%}")
    print(f"Barrier touch probability: {result['barrier_touch_probability']:.2%}")
    print(f"Expected life: {result['expected_life_years']:.3f} years")
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
