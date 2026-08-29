# Commands Reference

Quick reference for every CLI and Python API in the engine.

---

## Install

```bash
pip install -e ".[dev]"     # editable + dev extras (pytest, etc.)
pip install -e .             # editable, no extras
```

Python ≥ 3.11. Live data commands require network access (Yahoo Finance).

---

## Options (live market data)

```bash
# ATM call — all models vs real data
python scripts/price_option.py AAPL --strike 185 --expiry 0.25

# Specific model
python scripts/price_option.py SPY  --strike 550 --expiry 0.5  --model heston
python scripts/price_option.py TSLA --strike 250 --expiry 1.0  --model svi
python scripts/price_option.py NVDA --strike 900 --expiry 0.25 --model black-scholes
python scripts/price_option.py GS   --strike 520 --expiry 0.75 --model monte-carlo
python scripts/price_option.py MSFT --strike 420 --expiry 0.5  --model binomial

# Put / American
python scripts/price_option.py AAPL --strike 185 --expiry 0.25 --type put --model heston
python scripts/price_option.py TSLA --strike 250 --expiry 1.0  --type put --style american

# Override live data
python scripts/price_option.py SPY --strike 550 --expiry 0.5 --vol 0.18
python scripts/price_option.py SPY --strike 550 --expiry 0.5 --spot 540 --vol 0.20 --rate 0.05
```

| Flag | Default | Description |
|------|---------|-------------|
| `ticker` | required | Yahoo Finance ticker |
| `--strike / -K` | required | Strike price |
| `--expiry / -T` | required | Years to expiry (0.25 = 3M, 0.5 = 6M, 1.0 = 1Y) |
| `--type` | call | call or put |
| `--style` | european | european or american |
| `--model` | all | all · black-scholes · heston · svi · monte-carlo · binomial |
| `--vol / -v` | live ATM IV | Override implied vol (e.g. 0.20) |
| `--spot / -S` | live price | Override spot |
| `--rate / -r` | 0.045 | Risk-free rate |
| `--expiries` | 5 | Expiry dates used for calibration |

Output folder: `output/<TICKER>_<timestamp>/`

---

## Exotic Options

```bash
# Barrier — down-and-out call
python scripts/price_exotic.py AAPL barrier --strike 185 --expiry 0.25 --barrier 170 --barrier-type down-out
python scripts/price_exotic.py SPY  barrier --strike 500 --expiry 1.0  --barrier 450 --barrier-type up-out

# Asian (arithmetic MC + geometric closed-form + Kemna-Vorst)
python scripts/price_exotic.py SPY  asian --strike 500 --expiry 1.0
python scripts/price_exotic.py NVDA asian --strike 900 --expiry 0.5  --type put

# Lookback floating
python scripts/price_exotic.py TSLA lookback --expiry 0.5 --type put
python scripts/price_exotic.py AAPL lookback --expiry 1.0 --type call

# Digital
python scripts/price_exotic.py NVDA digital --strike 900 --expiry 0.25 --digital-type cash-or-nothing
python scripts/price_exotic.py GS   digital --expiry 0.25 --barrier 480 --digital-type one-touch
```

---

## Rates

```bash
# Basic curve bootstrap
python scripts/analyze_rates.py
python scripts/analyze_rates.py --name usd_demo --deposit 0.25:0.052 --deposit 1:0.050 --swap 5:0.046

# Advanced: swaption price + vol surface
python scripts/analyze_rates_advanced.py swaption --expiry 1 --tenor 5 --strike 0.05 --vol 0.20
python scripts/analyze_rates_advanced.py swaption --expiry 2 --tenor 10 --strike 0.055 --model bachelier

# Cap/floor with caplet breakdown
python scripts/analyze_rates_advanced.py capfloor --maturity 3 --strike 0.05 --vol 0.20
python scripts/analyze_rates_advanced.py capfloor --maturity 5 --strike 0.055 --type floor

# SABR smile calibration
python scripts/analyze_rates_advanced.py sabr --forward 0.05 --expiry 1 --beta 0.5
python scripts/analyze_rates_advanced.py sabr --forward 0.04 --expiry 2 --beta 0.0   # normal SABR

# Nelson-Siegel / Svensson yield curve fitting
python scripts/analyze_rates_advanced.py ns
```

---

## Credit

```bash
# CDS pricing
python scripts/price_credit.py IBM  cds --maturity 5 --spread 150
python scripts/price_credit.py AAPL cds --maturity 3 --spread 80  --position seller

# Hazard curve bootstrap from CDS quotes
python scripts/price_credit.py GS   curve --quotes 1:50 3:90 5:120 7:145 10:170
python scripts/price_credit.py MSFT curve --quotes 1:30 5:60 10:90 --recovery 0.30

# Risky corporate bond (Duffie-Singleton)
python scripts/price_credit.py GS   bond --face 1000 --coupon 4.5 --maturity 5
python scripts/price_credit.py IBM  bond --face 1000 --coupon 5.0 --maturity 10 --spread 120

# CVA on option exposure
python scripts/price_credit.py MSFT cva --strike 420 --expiry 1.0
python scripts/price_credit.py AAPL cva --strike 185 --expiry 0.5 --type put
```

---

## Volatility

```bash
# Full analysis: realized estimators + GARCH + VRP
python scripts/analyze_vol.py AAPL
python scripts/analyze_vol.py SPY  --window 30 --horizon 60
python scripts/analyze_vol.py NVDA --no-garch
python scripts/analyze_vol.py TSLA --window 21 --horizon 30

# With vol override (use instead of fetching ATM IV)
python scripts/analyze_vol.py AAPL --implied-vol 0.28
```

---

## Structured Products

```bash
# CDO: Gaussian copula tranching
python scripts/price_structured.py cdo --pd 0.02 --rho 0.20 --maturity 5
python scripts/price_structured.py cdo --pd 0.03 --rho 0.30 --maturity 3 --recovery 0.30

# Autocallable note
python scripts/price_structured.py autocall AAPL --coupon 0.08 --ki-barrier 0.70
python scripts/price_structured.py autocall SPY  --coupon 0.10 --ki-barrier 0.65 --autocall-level 1.05

# MBS / PSA prepayment model
python scripts/price_structured.py mbs --face 1000000 --wac 6.5 --wam 360 --psa 100
python scripts/price_structured.py mbs --face 1000000 --wac 7.0 --wam 300 --psa 200
```

---

## Portfolio & Risk

```bash
# Full risk report: Greeks + VaR + stress
python scripts/analyze_portfolio.py

# Custom portfolio
python scripts/analyze_portfolio.py --spot 100 --rate 0.05

# Subcommands
python scripts/analyze_portfolio.py greeks    # delta/gamma/vega/theta/rho aggregation
python scripts/analyze_portfolio.py var       # VaR/CVaR: historical, parametric, CF, MC
python scripts/analyze_portfolio.py stress    # 11 historical stress scenarios
python scripts/analyze_portfolio.py grid      # spot × vol P&L grid
```

---

## ML / Quantitative Finance

```bash
# American option pricing (Longstaff-Schwartz)
python scripts/ml_pricing.py lsm --spot 100 --strike 100 --expiry 1 --vol 0.20 --type put
python scripts/ml_pricing.py lsm --spot 100 --strike 110 --expiry 2 --vol 0.30 --type put --sims 100000

# SVI / SSVI smile calibration
python scripts/ml_pricing.py svi   --forward 100 --expiry 1
python scripts/ml_pricing.py ssvi  --forward 100 --expiry 1

# Volatility forecasting: HAR vs GBM vs LSTM
python scripts/ml_pricing.py volforecast --ticker AAPL
python scripts/ml_pricing.py volforecast --ticker SPY  --horizon 22
python scripts/ml_pricing.py volforecast --ticker NVDA --horizon 5
```

---

## FX Options

```bash
# Garman-Kohlhagen pricing + Greeks
python scripts/analyze_fx.py gk
python scripts/analyze_fx.py gk --spot 1.08 --strike 1.10 --tenor 0.5 --vol 0.09 --rd 0.05 --rf 0.03

# FX smile from 25-delta RR/BF market quotes
python scripts/analyze_fx.py smile
python scripts/analyze_fx.py smile --spot 1.08 --atm 0.080 --rr25 0.010 --bf25 0.003
python scripts/analyze_fx.py smile --spot 1.08 --atm 0.080 --rr25 0.012 --bf25 0.004 --rr10 0.022 --bf10 0.007

# FX barrier option with Vanna-Volga smile adjustment
python scripts/analyze_fx.py barrier
python scripts/analyze_fx.py barrier --spot 1.08 --strike 1.10 --barrier 1.02 --btype down-out
python scripts/analyze_fx.py barrier --spot 1.08 --strike 1.06 --barrier 1.12 --btype up-out

# Vol surface: smile term structure across tenors
python scripts/analyze_fx.py surface --spot 1.08
```

---

## Commodities

```bash
# Futures curve: term structure + convenience yields + calendar spreads
python scripts/analyze_commodities.py curve
python scripts/analyze_commodities.py curve --spot 80 --commodity WTI --storage 0.02
python scripts/analyze_commodities.py curve --spot 1900 --commodity Gold --backwardation

# Schwartz 1F mean-reversion model: calibration + option pricing
python scripts/analyze_commodities.py schwartz
python scripts/analyze_commodities.py schwartz --spot 80 --kappa 1.2 --vol 0.35
python scripts/analyze_commodities.py schwartz --spot 1900 --kappa 0.3 --vol 0.15

# Spread options: Margrabe (K=0) and Kirk (K>0)
python scripts/analyze_commodities.py spread
python scripts/analyze_commodities.py spread --f1 102 --f2 82 --sig1 0.25 --sig2 0.22 --rho 0.75
python scripts/analyze_commodities.py spread --f1 50 --f2 42 --sig1 0.30 --sig2 0.28 --rho 0.60

# Commodity option via Schwartz MC
python scripts/analyze_commodities.py option --spot 80 --kappa 0.8 --vol 0.30 --tenor 1.0
python scripts/analyze_commodities.py option --spot 1900 --kappa 0.5 --vol 0.18 --tenor 0.5
```

---

## Tests

```bash
# Full suite
pytest
pytest -v                                    # verbose
pytest --tb=short                            # condensed tracebacks

# Per module
pytest tests/test_options.py -v             # 37 tests: BS, IV, MC, Binomial, Surface
pytest tests/test_calibration.py -v         # 24 tests: Heston, SVI, store, engine
pytest tests/test_exotics.py -v             # 32 tests: Barrier, Asian, Lookback, Digital
pytest tests/test_forwards_futures.py -v    #  9 tests: forwards, futures
pytest tests/test_rates.py -v               #  8 tests: discount curve
pytest tests/test_rates_advanced.py -v      # 44 tests: swaptions, caps, SABR, NS/Svensson
pytest tests/test_credit.py -v              # 30 tests: HazardCurve, CDS, Bond, CVA
pytest tests/test_volatility.py -v          # 37 tests: realized vol, GARCH, variance swap
pytest tests/test_structured.py -v          # 38 tests: CDO, Autocall, MBS
pytest tests/test_portfolio.py -v           # 35 tests: Greeks, VaR, stress
pytest tests/test_ml.py -v                  # 30 tests: LSM, SSVI, HAR, vol forecast
pytest tests/test_fx.py -v                  # 26 tests: GK pricing, smile, delta/strike
pytest tests/test_commodities.py -v         # 31 tests: futures curve, Schwartz, spreads

# Filters
pytest -k "heston"
pytest -k "barrier or asian"
pytest -k "swaption"
pytest -k "var"
```

---

## Structure

```text
core/
  models/          Heston, SVI
  calibration/     MLE calibrator, warm starts, SQLite audit trail
  data/            YFinanceLoader, SyntheticLoader

options/           Black-Scholes, greeks, implied vol, MC, binomial, vol surface, engine, charts
exotics/           Barrier (Reiner-Rubinstein), Asian, Lookback, Digital + charts
forwards/          Fair price, contract value, implied carry
futures/           Futures pricing, basis, mark-to-market PnL
rates/             Discount curves, bootstrap, swaps, swaptions (Black/Bachelier),
                   caps/floors, SABR, Nelson-Siegel/Svensson, charts
credit/            Hazard curves, CDS, risky bonds, CVA/DVA, charts
volatility/        Realized estimators (CC/Parkinson/GK/RS/YZ/EWMA), GARCH(1,1),
                   variance swaps, VRP, charts
structured/        CDO (Gaussian copula/LHP), autocallable notes, MBS/PSA, charts
portfolio/         Position & Portfolio, Greeks aggregation, VaR/CVaR (4 methods),
                   Kupiec backtest, stress scenarios, spot×vol grid, charts
ml/                Longstaff-Schwartz LSM, SVI/SSVI calibration,
                   HAR + Gradient Boosting + LSTM vol forecasting
fx/                Garman-Kohlhagen, vanna/volga greeks, 25/10-delta smile (RR/BF),
                   vanna-volga exotic pricing, delta↔strike mapping
commodities/       Futures term structure, convenience yield, calendar spreads,
                   crack/spark spreads, Schwartz 1F, spread options (Margrabe/Kirk)

scripts/
  price_option.py          Options CLI (live Yahoo Finance data)
  price_exotic.py          Exotic options CLI
  price_credit.py          Credit (CDS, bond, CVA) CLI
  price_structured.py      Structured products CLI
  analyze_rates.py         Basic rates curve CLI
  analyze_rates_advanced.py  Swaptions, caps/floors, SABR, Nelson-Siegel CLI
  analyze_vol.py           Volatility analysis CLI
  analyze_portfolio.py     Portfolio risk CLI
  ml_pricing.py            ML pricing CLI (LSM, SSVI, vol forecast)
  analyze_fx.py            FX options CLI
  analyze_commodities.py   Commodities CLI

tests/             381 tests across 13 test files
derivatives_engine.ipynb   62-cell theory + examples notebook
```
# Phoenix Workbench Commands

```bash
pip install -e ".[dev]"
python scripts/init_db.py
python scripts/ingest_market.py AAPL --source yfinance --period 2y
python scripts/ingest_crypto.py bitcoin ethereum
python scripts/price_autocall.py examples/phoenix_autocall_aapl.json --paths 50000
python scripts/run_api.py
```

Open `apps/workbench/index.html` and point the API field to `http://localhost:8000`.
