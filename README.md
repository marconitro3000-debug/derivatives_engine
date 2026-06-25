# options-pricer

> Modular Python library for derivatives pricing, implied-volatility calibration, and stochastic-volatility modelling.

- **[COMMANDS.md](COMMANDS.md)** — all terminal commands with examples
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — module breakdown, math, API reference, implementation notes
- **[paper.ipynb](paper.ipynb)** — paper-style walkthrough: theory, derivations, and runnable code

---

## Features

| Module | Capability |
|--------|-----------|
| `core/black_scholes` | European pricing, all Greeks, put-call parity |
| `core/implied_vol` | Newton-Raphson + Brent solver; IV surface |
| `core/monte_carlo` | GBM paths, 5 exotic payoffs, antithetic + control-variate |
| `core/binomial_tree` | CRR European & American; early-exercise premium |
| `models/heston` | Char-function pricing (Gil-Pelaez), Feller check |
| `models/svi` | Per-maturity smile, butterfly-arbitrage check |
| `calibration/` | WLS fit to live chains, warm start, EWMA smoothing, SQLite audit trail |
| `surface/` | Spline / RBF non-parametric IV interpolation |
| `data/` | YFinanceLoader (live) — no API key needed |

---

## Install

```bash
pip install -e ".[dev]"   # with pytest
pip install -e .           # runtime only
```

Python >= 3.11. Network access required for live data.

---

## CLI — `price_option.py`

Fetches live spot and ATM IV. Heston and SVI are calibrated to the real option chain before pricing.

```bash
python price_option.py AAPL --strike 185 --expiry 0.25
python price_option.py SPY  --strike 550 --expiry 0.5  --model heston
python price_option.py TSLA --strike 250 --expiry 1.0  --type put --model all
python price_option.py NVDA --strike 900 --expiry 0.25 --model black-scholes
python price_option.py GS   --strike 520 --expiry 0.75 --model svi
```

| Argument | Default | Description |
|----------|---------|-------------|
| `ticker` | required | Yahoo Finance ticker |
| `--strike` / `-K` | required | Strike price |
| `--expiry` / `-T` | required | Years to expiry (`0.25` = 3 months) |
| `--type` | `call` | `call` or `put` |
| `--model` | `all` | `all`, `black-scholes`, `heston`, `svi`, `monte-carlo`, `binomial` |
| `--vol` / `-v` | live ATM IV | Override implied vol |
| `--spot` / `-S` | live price | Override spot |
| `--rate` / `-r` | `0.045` | Risk-free rate |
| `--expiries` | `5` | Expiry dates for Heston/SVI calibration |

---

## Tests

```bash
pytest                           # 61 tests
pytest tests/test_pricer.py      # pricing + surface
pytest tests/test_calibration.py # calibration recovery
```

---

## Structure

```
options_pricer/
├── engine.py            # PricingEngine — self-calibrating, SQLite-persistent
├── core/                # BS · IV · Monte Carlo · Binomial
├── models/              # Heston · SVI
├── calibration/         # WLS calibrator · store
├── surface/             # VolSurface spline/RBF
└── data/                # YFinanceLoader · SyntheticLoader
```
