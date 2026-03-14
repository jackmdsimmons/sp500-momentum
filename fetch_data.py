"""
Fetch SPY total return data via yfinance.
SPY is used as an S&P 500 proxy — adjusted close prices account for dividends.

Output: data/spy_daily.csv
"""

import os
import pandas as pd
import yfinance as yf

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "spy_daily.csv")
TICKER = "SPY"
START  = "1993-01-01"   # SPY inception


def fetch():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Downloading {TICKER} from {START}...")
    df = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
    df = df[["Close"]].rename(columns={"Close": "price"})
    df.index.name = "date"
    df.to_csv(OUTPUT_FILE)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    fetch()
