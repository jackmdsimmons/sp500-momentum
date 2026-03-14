"""
Download iShares ETF constituent holdings.

Thin wrapper around quant_tools.constituents for this project.

Usage
-----
    python fetch_constituents.py              # downloads ACWI
    python fetch_constituents.py --ticker URTH
    python fetch_constituents.py --all
"""

import argparse
from quant_tools.constituents import fetch, save, FUNDS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download iShares ETF holdings")
    parser.add_argument("--ticker", default="ACWI", help=f"Fund ticker: {list(FUNDS)}")
    parser.add_argument("--all", action="store_true", help="Download all registered funds")
    args = parser.parse_args()

    tickers = list(FUNDS) if args.all else [args.ticker]
    for t in tickers:
        df = fetch(t)
        save(df, t, "data")
        print(f"\nSample ({t}):")
        print(df[["ticker", "name", "sector", "country", "weight_pct"]].head(10).to_string(index=False))
        print()
