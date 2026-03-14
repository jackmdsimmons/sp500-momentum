"""
Run the full momentum analysis pipeline.

  python run.py
"""

from fetch_data import fetch
from momentum import compute_all
from backtest import spread_table, plot_quintile_bars, plot_spread_heatmap


def main():
    # 1. Fetch data
    prices_df = fetch()
    prices = prices_df["price"].squeeze()

    # 2. Compute momentum metrics
    print("\nComputing momentum metrics...")
    metrics = compute_all(prices)
    print(f"Metrics: {list(metrics.columns)}")

    # 3. Quintile analysis
    print("\nComputing Q5-Q1 spreads across forward horizons...")
    spreads = spread_table(prices, metrics)

    print("\n-- Q5-Q1 Forward Return Spread (%) --")
    print(spreads.to_string())

    # 4. Charts
    print("\nGenerating charts...")
    plot_quintile_bars(prices, metrics, horizon="fwd_1m")
    plot_quintile_bars(prices, metrics, horizon="fwd_3m")
    plot_spread_heatmap(spreads)


if __name__ == "__main__":
    main()
