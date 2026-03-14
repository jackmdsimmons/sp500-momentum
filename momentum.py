"""
Momentum metrics for a daily price series.

Metrics implemented:
  - mom_12_1   : 12-month return skipping last month (classic Jegadeesh-Titman)
  - mom_6_1    : 6-month return skipping last month
  - mom_3_1    : 3-month return skipping last month
  - mom_1      : 1-month return (raw, no skip)
  - ma_ratio   : price / 200-day moving average
  - high_52w   : price / 52-week high
  - risk_adj   : 12-1 momentum divided by 12-month realised volatility
  - tsmom      : sign of 12-1 momentum (+1 long, -1 short)
"""

import pandas as pd
import numpy as np


TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR  = 252


def compute_all(prices: pd.Series) -> pd.DataFrame:
    """
    Given a daily price series, return a DataFrame of all momentum metrics.
    Index aligned to prices.
    """
    p = prices.copy()

    metrics = pd.DataFrame(index=p.index)

    # ── Return-based momentum ─────────────────────────────────────────────────

    # 12-1: return from 12 months ago to 1 month ago (skip last month)
    metrics["mom_12_1"] = (
        p.shift(TRADING_DAYS_PER_MONTH) /
        p.shift(12 * TRADING_DAYS_PER_MONTH) - 1
    )

    # 6-1
    metrics["mom_6_1"] = (
        p.shift(TRADING_DAYS_PER_MONTH) /
        p.shift(6 * TRADING_DAYS_PER_MONTH) - 1
    )

    # 3-1
    metrics["mom_3_1"] = (
        p.shift(TRADING_DAYS_PER_MONTH) /
        p.shift(3 * TRADING_DAYS_PER_MONTH) - 1
    )

    # 1-month (no skip)
    metrics["mom_1"] = p / p.shift(TRADING_DAYS_PER_MONTH) - 1

    # ── Trend-based metrics ───────────────────────────────────────────────────

    # Price relative to 200-day moving average
    metrics["ma_ratio"] = p / p.rolling(200).mean() - 1

    # Price relative to 52-week high (positive = near high, negative = far below)
    # Signal: positive when price is within 5% of 52-week high
    ratio = p / p.rolling(TRADING_DAYS_PER_YEAR).max() - 1
    metrics["high_52w"] = ratio + 0.05  # shift so >0 means within 5% of high

    # ── Risk-adjusted momentum ────────────────────────────────────────────────

    # Annualised realised volatility over past 12 months
    daily_returns = p.pct_change()
    vol_12m = daily_returns.rolling(TRADING_DAYS_PER_YEAR).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    metrics["risk_adj"] = metrics["mom_12_1"] / vol_12m.replace(0, np.nan)

    # ── Time-series momentum signal ───────────────────────────────────────────

    # +1 if 12-1 momentum is positive (go long), -1 if negative (go short/cash)
    metrics["tsmom"] = np.sign(metrics["mom_12_1"])

    return metrics
