"""
Run the full momentum analysis pipeline.

  python run.py
"""

from fetch_data import fetch
from momentum import compute_all
from backtest import (
    spread_table,
    plot_quintile_bars,
    plot_spread_heatmap,
    plot_rolling_spread,
    plot_yearly_heatmap,
)


def main():
    # 1. Fetch data (SPY from 1993)
    prices_df = fetch()
    prices = prices_df["price"].squeeze()

    # 2. Compute momentum metrics
    print("\nComputing momentum metrics...")
    metrics = compute_all(prices)
    print(f"Metrics: {list(metrics.columns)}")

    # 3. Full-period quintile spreads
    print("\nComputing Q5-Q1 spreads across forward horizons...")
    spreads = spread_table(prices, metrics)
    print("\n-- Q5-Q1 Forward Return Spread (%) --")
    print(spreads.to_string())

    # 4. Charts
    print("\nGenerating charts...")

    # Full-period quintile bars
    plot_quintile_bars(prices, metrics, horizon="fwd_1m")
    plot_quintile_bars(prices, metrics, horizon="fwd_3m")
    plot_spread_heatmap(spreads)

    # Time dependency
    plot_rolling_spread(prices, metrics, horizon="fwd_1m", window=36)
    plot_rolling_spread(prices, metrics, horizon="fwd_3m", window=36)
    plot_yearly_heatmap(prices, metrics, horizon="fwd_1m")
    plot_yearly_heatmap(prices, metrics, horizon="fwd_3m")

    print("\nAll charts saved to data/")


if __name__ == "__main__":
    main()
