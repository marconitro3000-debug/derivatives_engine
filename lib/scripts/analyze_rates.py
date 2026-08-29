"""
Build a simple interest-rate curve and save rate charts.

Examples
--------
    python scripts/analyze_rates.py
    python scripts/analyze_rates.py --name usd_demo --deposit 0.25:0.052 --swap 2:0.048 --swap 5:0.046
"""

import argparse
import io
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

from rates import InterestRateSwap, bootstrap_deposit_swap_curve
from rates.charts import plot_curve


def _parse_quotes(items: list[str]) -> dict[float, float]:
    quotes = {}
    for item in items:
        try:
            maturity, rate = item.split(":", 1)
            quotes[float(maturity)] = float(rate)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"quote must be maturity:rate, got {item!r}"
            ) from exc
    return quotes


parser = argparse.ArgumentParser(description="Bootstrap a rates curve and save charts.")
parser.add_argument("--name", default="demo_curve", help="Curve/run label.")
parser.add_argument("--deposit", action="append", default=[],
                    help="Deposit quote as maturity:rate, e.g. 0.5:0.042")
parser.add_argument("--swap", action="append", default=[],
                    help="Swap quote as maturity:rate, e.g. 5:0.047")
parser.add_argument("--fixed-freq", type=int, default=1,
                    help="Fixed leg payments per year.")
parser.add_argument("--notional", type=float, default=1_000_000,
                    help="Swap notional for summary PV.")
args = parser.parse_args()

deposit_quotes = _parse_quotes(args.deposit) or {
    0.25: 0.0410,
    0.50: 0.0420,
    1.00: 0.0430,
}
swap_quotes = _parse_quotes(args.swap) or {
    2.0: 0.0440,
    3.0: 0.0450,
    5.0: 0.0470,
    10.0: 0.0490,
}

curve = bootstrap_deposit_swap_curve(
    deposit_quotes=deposit_quotes,
    swap_quotes=swap_quotes,
    fixed_freq=args.fixed_freq,
    name=args.name,
)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(base_dir, "output", "rates", args.name, run_id)
os.makedirs(out_dir, exist_ok=True)

saved = plot_curve(curve, out_dir)

log = io.StringIO()
log.write(f"Curve: {curve.name}\n")
log.write(f"Output: output/rates/{args.name}/{run_id}/\n\n")
log.write("Pillars\n")
for t, df in zip(curve.times, curve.discount_factors):
    log.write(f"  {t:6.3f}Y  DF={df:.8f}  zero={curve.zero_rate(float(t)):.4%}\n")

log.write("\nDerived rates\n")
for start, end in [(0.0, 0.25), (0.25, 0.5), (1.0, 2.0), (2.0, 5.0)]:
    if end <= curve.times[-1]:
        log.write(f"  forward {start:.2f}Y->{end:.2f}Y = {curve.forward_rate(start, end):.4%}\n")

log.write("\nPar swap rates\n")
for maturity in [1.0, 2.0, 5.0, 10.0]:
    if maturity <= curve.times[-1]:
        log.write(f"  {maturity:4.1f}Y = {curve.par_swap_rate(maturity, args.fixed_freq):.4%}\n")

if curve.times[-1] >= 5.0:
    par = curve.par_swap_rate(5.0, args.fixed_freq)
    swap = InterestRateSwap(args.notional, par, 5.0, pay_freq=args.fixed_freq)
    log.write(f"\n5Y par swap PV at par = {swap.pv(curve):.2f}\n")

with open(os.path.join(out_dir, "results.txt"), "w", encoding="utf-8") as fh:
    fh.write(log.getvalue())

print(log.getvalue())
print("Saved charts:")
for filename in saved:
    size_kb = os.path.getsize(os.path.join(out_dir, filename)) // 1024
    print(f"  {size_kb:4d} KB  {filename}")
