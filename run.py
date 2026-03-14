"""
Run the full momentum analysis pipeline.

  python run.py
"""

import pandas as pd
from fetch_data import fetch
from momentum import compute_all
from backtest import run, plot_cumulative, plot_sharpe_comparison


def main():
    # 1. Fetch data
    prices_df = fetch()
    prices = prices_df["price"].squeeze()

    # 2. Compute momentum metrics
    print("\nComputing momentum metrics...")
    metrics = compute_all(prices)
    print(f"Metrics computed: {list(metrics.columns)}")

    # 3. Backtest
    print("\nRunning backtests...")
    summary = run(prices, metrics)

    print("\n-- Results --")
    print(summary.to_string())

    # 4. Charts
    print("\nGenerating charts...")
    plot_cumulative(prices, metrics, top_n=4)
    plot_sharpe_comparison(summary)


if __name__ == "__main__":
    main()
