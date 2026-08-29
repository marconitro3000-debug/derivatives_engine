# Derivatives Engine & Visual Structuring Workbench

A modular Python derivatives pricing engine with an experimental visual workbench for designing, pricing and explaining structured products such as Phoenix Autocall Notes.

## Overview

The repository has two layers:

- Quant Engine: Python modules for derivatives pricing, volatility, calibration, risk analytics and structured-product models.
- Visual Workbench: a real drag/connect node-canvas frontend (React + React Flow, in `frontend/`) backed by FastAPI, SQLite audit storage and public market-data ingestion.

Main demo: Phoenix Autocall pricing with GBM Monte Carlo, autocall/barrier event detection, memory coupons, confidence intervals, finite-difference Greeks, stress scenarios, spot-vol heatmaps, validation flags and JSON workflow serialization.

A second product on the same canvas: vanilla call/put pricing with a selectable model — Black-Scholes, CRR binomial tree (European/American), Monte Carlo (antithetic + control variate), and a market-calibrated Heston/SVI (real option chain, live least-squares fit, persisted for warm-starting) — plus a one-click comparison across all models.

## Why This Project Exists

The project is intended as a serious financial-engineering prototype: small enough to inspect, modular enough to extend, and explicit about model assumptions.

## Installation

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
python scripts/init_db.py
python scripts/price_autocall.py examples/phoenix_autocall_aapl.json --paths 50000
python scripts/run_api.py
```

In a second terminal, start the visual workbench:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Drag nodes from the palette, connect them, edit params, and hit Run — it fetches live market data, compiles the graph, and prices it through the FastAPI backend. If the API is not running it uses a labelled demo fallback instead.

## API Usage

```bash
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/price/phoenix \
  -H "Content-Type: application/json" \
  -d @examples/phoenix_autocall_aapl.json
```

Core endpoints:

- `GET /api/health`
- `POST /api/price/phoenix`
- `POST /api/validate/workflow`
- `POST /api/stress/phoenix`
- `POST /api/term-sheet/phoenix`
- `GET /api/market/quote/{symbol}`
- `GET /api/market/history/{symbol}`
- `GET /api/market/calibration/{symbol}`
- `GET /api/market/rates/{currency}`
- `GET /api/market/option-chain/{symbol}`
- `GET /api/market/vol-surface/{symbol}`
- `POST /api/price/option` (Black-Scholes, binomial tree, Monte Carlo, Heston, SVI)
- `POST /api/price/option/compare`
- `POST /api/calibrate/{ticker}` / `GET /api/calibration/{ticker}`
- `GET /api/market/dividend-yield/{symbol}` (real trailing-twelve-month yield)
- `GET /api/market/rates/{currency}?maturity_years=...` (interpolated discount curve, not just a flat tenor pick)

Vanilla option models now also include `merton` (jump-diffusion, closed-form) and `local_vol` (Dupire local vol from a jointly-calibrated, calendar-consistent SSVI surface — `POST /api/calibrate/{ticker}?model=local_vol`).

## CLI Usage

```bash
python scripts/init_db.py
python scripts/ingest_market.py AAPL --source yfinance --period 2y
python scripts/ingest_crypto.py bitcoin ethereum
python scripts/price_autocall.py examples/phoenix_autocall_aapl.json --paths 50000
python scripts/run_api.py
```

## Architecture

Pricing logic lives in `structured/`. API routes live in `api/routes/` and call the engine. Market data adapters live under `data/sources/`. SQLite schema and repositories live in `db/`. The frontend is isolated under `frontend/` (React + React Flow, its own `npm` project — see `docs/visual_workbench.md`).

Legacy modules that were already present under `lib/` are preserved; root-level modules expose the current active engine surface.

## Data Sources

Initial public sources:

- yfinance for equities, ETFs and historical OHLCV.
- Stooq as a historical-price fallback.
- CoinGecko for crypto spot prices.
- FRED for rates and macro series using optional `FRED_API_KEY`.
- SEC EDGAR helpers using `SEC_USER_AGENT`.

No API keys are hardcoded.

## Pricing Methodology

The Phoenix Autocall implementation uses risk-neutral GBM paths, discrete observation dates, memory coupon accounting, early redemption detection, final conditional capital redemption, Monte Carlo standard error, finite-difference Greeks, stress scenarios and spot-vol grid repricing.

See `docs/phoenix_autocall_methodology.md`.

## Tests

```bash
python -m compileall .
pytest
```

## Scope And Limitations

This is a research-oriented educational prototype. It is not an institutional pricing library, trading system, execution platform, or investment recommendation engine. Models are simplified and intended for experimentation, learning and prototyping.

## Roadmap

- Multi-asset autocalls and worst-of baskets.
- Dividend curves (rates curve is now live via FRED).
- Calibrated local/stochastic volatility hooks (SSVI vol-surface calibration from live option chains is now live).
- Persisted workflow save/load in the node canvas and richer term-sheet exports.
- Expanded tests for market-data fallbacks and API error handling.
