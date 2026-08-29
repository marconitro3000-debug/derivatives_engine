# Architecture

```text
+---------------------------------+
| frontend/ (React + React Flow)   |
| Draggable/connectable node canvas|
| Objectives + per-node config     |
+---------------+-------------------+
                |
                | live market-data fetches (quote, rates, calibration, vol-surface)
                | + compiled JSON product specification
                v
+---------------+-------------------+
| FastAPI api.main                  |
| /api/price/phoenix                |
| /api/validate/workflow            |
| /api/market/quote, /history        |
| /api/market/rates, /option-chain,  |
|              /vol-surface          |
+---------------+-------------------+
                |
                v
+---------------+-------------------+
| Pricing Engine                    |
| GBM MC (Phoenix payoff)           |
| Greeks / stress / heatmap         |
| SSVI vol-surface calibration      |
+---------------+-------------------+
                |
                v
+---------------+-------------------+
| SQLite                            |
| ohlcv, market_snapshots, rates,    |
| option_chains, pricing_runs,       |
| workflows                          |
+-----------------------------------+
```

## Design principle

The frontend is not the source of truth for valuation. It's a real node-canvas editor (drag/connect blocks — Underlying, Rates Curve, Volatility Forecast, Implied Vol Surface, Barrier, Coupon, Memory, Autocall, Pricing Output, Stress Test, Term Sheet, etc.) that fetches live market data per node and compiles the graph into a flat pricing specification client-side (`frontend/src/lib/compileGraph.ts`), using the exact field names `api/schemas.py` expects. The backend owns pricing, validation and market-data ingestion; it never trusts the frontend's numbers.

## Real vs. offline data

Every "live" node hits a real public data source through the backend:

- **Underlying** → yfinance spot + history (`/api/market/quote`, `/api/market/history`)
- **Rates Curve** → live FRED Treasury constant-maturity series (`/api/market/rates`), falling back to a labeled prototype curve if `FRED_API_KEY` is unset or the request fails
- **Volatility Forecast** → realized volatility from real OHLCV history (`/api/market/calibration`)
- **Implied Vol Surface** → a real live option chain (yfinance) calibrated to SSVI per expiry, with an arbitrage-free check (`/api/market/vol-surface`)

If the backend is unreachable, the frontend falls back to a simplified local formula and shows an explicit "offline demo mode" banner — it never silently presents mock numbers as real pricing.
