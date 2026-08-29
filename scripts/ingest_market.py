from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.ingestion import get_history, get_quote


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest public market data into SQLite cache.")
    parser.add_argument("symbol")
    parser.add_argument("--source", default="auto", choices=["auto", "yfinance", "stooq"])
    parser.add_argument("--period", default="1y")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    quote = get_quote(args.symbol, source=args.source)
    history = get_history(args.symbol, start=args.start, end=args.end, source=args.source, period=args.period)
    print(f"{args.symbol}: quote={quote['price']:.4f} source={quote['source']}")
    print(f"Cached {len(history)} OHLCV rows")


if __name__ == "__main__":
    main()
