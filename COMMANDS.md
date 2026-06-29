# Commands

## Pricing

```bash
python scripts/price_option.py TICKER --strike K --expiry T --model MODEL --type call|put
```

`--expiry` is in years. Examples: `0.083` = 1 month, `0.25` = 3 months,
`0.5` = 6 months, `1.0` = 1 year.

`--model` can be: `all` (default), `black-scholes`, `heston`, `svi`,
`monte-carlo`, `binomial`.

By default the CLI fetches spot and ATM IV from Yahoo Finance. Override them
with `--spot` and `--vol`.

## Examples

```bash
python scripts/price_option.py AAPL --strike 185 --expiry 0.25
python scripts/price_option.py SPY --strike 550 --expiry 0.5 --model heston
python scripts/price_option.py TSLA --strike 250 --expiry 1.0 --model svi
python scripts/price_option.py NVDA --strike 900 --expiry 0.25 --model black-scholes
python scripts/price_option.py GS --strike 520 --expiry 0.75 --model monte-carlo
python scripts/price_option.py MSFT --strike 420 --expiry 0.5 --model binomial
python scripts/price_option.py AAPL --strike 185 --expiry 0.25 --type put --model heston
python scripts/price_option.py SPY --strike 550 --expiry 0.5 --vol 0.18
python scripts/price_option.py SPY --strike 550 --expiry 0.5 --spot 540 --vol 0.20
python scripts/price_option.py TSLA --strike 250 --expiry 1.0 --type put --style american
```

Option output path:

```text
output/<TICKER>/options/<run_id>/
```

## Rates

```bash
python scripts/analyze_rates.py
python scripts/analyze_rates.py --name usd_demo --deposit 0.25:0.052 --deposit 1:0.050 --swap 5:0.046
```

Rates output path:

```text
output/rates/<curve_name>/<run_id>/
```

## Tests

```bash
pytest
pytest tests/test_options.py
pytest tests/test_calibration.py
pytest tests/test_forwards_futures.py
pytest tests/test_rates.py
pytest -k "heston"
```

## Install

```bash
pip install -e ".[dev]"
pip install -e .
```
