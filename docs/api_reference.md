# API Reference

Run with:

```bash
python scripts/run_api.py
```

Endpoints:

- `GET /api/health`: service health.
- `POST /api/price/phoenix`: prices a Phoenix Autocall and optionally includes Greeks, stress and heatmap.
- `POST /api/validate/workflow`: validates workflow metadata and product parameters.
- `POST /api/stress/phoenix`: returns standard scenario repricing.
- `POST /api/term-sheet/phoenix`: returns an indicative term-sheet preview.
- `GET /api/market/quote/{symbol}`: downloads and caches a quote.
- `GET /api/market/history/{symbol}`: downloads and caches OHLCV history.
- `GET /api/market/calibration/{symbol}`: spot + realized volatility (21d/63d/252d) from real history.
- `GET /api/market/rates/{currency}`: live FRED Treasury curve (falls back to a labeled prototype curve without `FRED_API_KEY`).
- `GET /api/market/option-chain/{symbol}`: real option chain (calls + puts) via yfinance, cached to SQLite.
- `GET /api/market/vol-surface/{symbol}`: SSVI calibration per expiry from a live option chain, with an arbitrage-free flag.
- `POST /api/price/option`: prices a vanilla call/put with a selectable model (`black_scholes`, `binomial`, `monte_carlo`, `heston`, `svi`). Heston/SVI require a prior calibration (409 if missing).
- `POST /api/price/option/compare`: prices the same contract across all models at once; Heston/SVI rows are flagged `calibrated: false` instead of erroring if uncalibrated.
- `POST /api/calibrate/{ticker}?model=heston|svi`: calibrates to a live option chain (yfinance) and persists to SQLite for warm-starting later calls.
- `GET /api/calibration/{ticker}?model=heston|svi`: latest calibration status for a ticker/model.
