# sp500-momentum

Time-series momentum analysis on the S&P 500 (SPY ETF) over 10 years.

## What it does

1. Downloads 10 years of SPY daily price data via yfinance
2. Computes 8 momentum metrics
3. Backtests each as a market timing signal (long SPY vs. cash)
4. Compares by Sharpe ratio, return, drawdown, and win rate

## Momentum metrics

| Metric | Description |
|---|---|
| `mom_12_1` | 12-month return skipping last month (Jegadeesh-Titman classic) |
| `mom_6_1` | 6-month return skipping last month |
| `mom_3_1` | 3-month return skipping last month |
| `mom_1` | Raw 1-month return |
| `ma_ratio` | Price vs. 200-day moving average |
| `high_52w` | Price proximity to 52-week high |
| `risk_adj` | 12-1 momentum / realised volatility |
| `tsmom` | Binary sign of 12-1 momentum |

## Setup

```
pip install -r requirements.txt
python run.py
```

## Results (2015-2025)

| Strategy | Ann. Return | Sharpe | Max Drawdown |
|---|---|---|---|
| Buy & Hold | 15.49% | 0.769 | -23.93% |
| mom_12_1 | 12.01% | 0.605 | -19.45% |
| risk_adj | 12.01% | 0.605 | -19.45% |
| high_52w | 8.25% | 0.463 | **-7.95%** |
| mom_6_1 | 10.15% | 0.491 | -19.45% |

Buy & hold wins on raw return in this bull market period. 12-1 momentum offers a better risk-adjusted profile with lower drawdown.
