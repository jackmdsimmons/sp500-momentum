"""
Backtest each momentum metric as a simple timing signal.

Signal logic:
  - If metric > 0  → long SPY next month (invested)
  - If metric <= 0 → cash next month (flat, 0% return)

This isolates each metric's ability to predict positive vs. negative months.

Output metrics per strategy:
  - Annualised return
  - Annualised volatility
  - Sharpe ratio
  - Max drawdown
  - Win rate (% of months correctly called)
  - vs. buy-and-hold benchmark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR  = 252
RISK_FREE_RATE         = 0.04   # approximate annual risk-free rate


def monthly_returns(prices: pd.Series) -> pd.Series:
    """Resample daily prices to month-end, compute monthly returns."""
    monthly = prices.resample("ME").last()
    return monthly.pct_change().dropna()


def build_strategy_returns(
    signal: pd.Series,
    fwd_returns: pd.Series,
) -> pd.Series:
    """
    Given a daily signal and monthly forward returns,
    align signal to month-end and shift forward one period.
    """
    # Resample signal to month-end (use last value of month)
    sig_monthly = signal.resample("ME").last()

    # Shift signal forward: today's signal → next month's position
    position = (sig_monthly.shift(1) > 0).astype(float)

    # Align to forward returns index
    position, fwd = position.align(fwd_returns, join="inner")

    strategy_returns = position * fwd
    strategy_returns.name = signal.name
    return strategy_returns


def performance_stats(returns: pd.Series) -> dict:
    """Compute annualised performance stats from monthly returns."""
    ann_return = (1 + returns.mean()) ** 12 - 1
    ann_vol    = returns.std() * np.sqrt(12)
    sharpe     = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_dd = drawdown.min()

    win_rate = (returns > 0).mean()

    return {
        "ann_return":  round(ann_return * 100, 2),
        "ann_vol":     round(ann_vol * 100, 2),
        "sharpe":      round(sharpe, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate":    round(win_rate * 100, 1),
        "n_months":    len(returns),
    }


def run(prices: pd.Series, metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Run backtest for all metrics. Returns a summary DataFrame.
    """
    fwd = monthly_returns(prices)

    results = {}

    # Benchmark: buy and hold
    results["buy_and_hold"] = performance_stats(fwd)

    for col in metrics.columns:
        strat_returns = build_strategy_returns(metrics[col], fwd)
        results[col] = performance_stats(strat_returns)

    summary = pd.DataFrame(results).T
    summary.index.name = "strategy"
    return summary


def plot_cumulative(prices: pd.Series, metrics: pd.DataFrame, top_n: int = 4):
    """Plot cumulative returns of top strategies vs. buy-and-hold."""
    fwd = monthly_returns(prices)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Buy and hold
    bah = (1 + fwd).cumprod()
    ax.plot(bah.index, bah.values, label="Buy & Hold", linewidth=2, color="black")

    # Top N strategies by Sharpe
    summary = run(prices, metrics)
    top = (
        summary.drop("buy_and_hold")
        .sort_values("sharpe", ascending=False)
        .head(top_n)
        .index.tolist()
    )

    colors = sns.color_palette("tab10", top_n)
    for i, col in enumerate(top):
        strat = build_strategy_returns(metrics[col], fwd)
        cum = (1 + strat).cumprod()
        sharpe = summary.loc[col, "sharpe"]
        ax.plot(cum.index, cum.values, label=f"{col} (SR={sharpe})", color=colors[i])

    ax.set_title("Cumulative Returns: Top Momentum Strategies vs. Buy & Hold")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/cumulative_returns.png", dpi=150)
    plt.show()
    print("Chart saved to data/cumulative_returns.png")


def plot_sharpe_comparison(summary: pd.DataFrame):
    """Bar chart of Sharpe ratios across all strategies."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_summary = summary.sort_values("sharpe", ascending=True)
    colors = ["black" if i == "buy_and_hold" else "steelblue"
              for i in sorted_summary.index]
    bars = ax.barh(sorted_summary.index, sorted_summary["sharpe"], color=colors)
    ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_title("Sharpe Ratio by Momentum Strategy")
    ax.set_xlabel("Sharpe Ratio")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/sharpe_comparison.png", dpi=150)
    plt.show()
    print("Chart saved to data/sharpe_comparison.png")
