# Public data sources

The project supports several public/free data connectors in a deliberately defensive way.

## Yahoo Finance via yfinance

Used for equities/ETF spot + history, and for real option chains (`GET /api/market/option-chain/{symbol}`, `GET /api/market/vol-surface/{symbol}`) which back the live SSVI calibration and arbitrage checks. `yfinance` is convenient, but it is not affiliated with Yahoo and should be treated as a research/education connector rather than a production market-data feed.

## Stooq

Useful fallback for daily OHLCV CSV data. Symbols often require suffixes such as `aapl.us`.

## CoinGecko

Used for crypto spot and market-chart data. CoinGecko's simple price endpoints are convenient for crypto snapshots.

## FRED

Used for the live USD risk-free curve (`GET /api/market/rates/{currency}`), pulling Treasury constant-maturity series (1M/3M/1Y/2Y). Requires `FRED_API_KEY`; without it (or on request failure) the endpoint falls back to a clearly labeled prototype curve (`source: "prototype_fallback"`) rather than failing the whole pricing run.

## SEC EDGAR

Used for company submissions and XBRL company facts. The SEC requires a clear User-Agent header.

## Data-quality layer to add later

- timestamp freshness checks;
- stale price detection;
- duplicate handling;
- cross-source reconciliation;
- data lineage/audit trail;
- source-specific rate limit handling.
