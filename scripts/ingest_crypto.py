from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.ingestion import get_crypto_price


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CoinGecko crypto spot prices.")
    parser.add_argument("assets", nargs="+")
    args = parser.parse_args()

    for asset in args.assets:
        quote = get_crypto_price(asset)
        print(f"{asset}: {quote['price']:.4f} {quote.get('currency') or ''} source={quote['source']}")


if __name__ == "__main__":
    main()
